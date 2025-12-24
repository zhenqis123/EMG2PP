from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl
import os

from hydra.utils import to_absolute_path


@dataclass
class PianoKPMSessionData:
    """单个 Pianokpm performance 的只读封装。
    模式类似 emg2pose 的 Emg2PoseSessionData，只是数据来源是若干 pkl 文件。
    """

    data_root: Path
    basename: str  # 不含 .pkl

    def __post_init__(self) -> None:
        root = Path(to_absolute_path(str(self.data_root)))

        # keystroke: [T, K?]
        ks_path = root / "keystroke_data" / f"{self.basename}.pkl"
        with open(ks_path, "rb") as f:
            ks = pkl.load(f)
        self.keystroke = torch.as_tensor(ks, dtype=torch.float32)

        # emg: [T, C]
        emg_path = root / "emg_data" / f"{self.basename}.pkl"
        with open(emg_path, "rb") as f:
            emg = pkl.load(f)
        self.emg = torch.as_tensor(emg, dtype=torch.float32)

        # pose_top: [T, 21, 2] -> [T, 42]
        pose_top_path = root / "hand_data" / "keyp_top" / f"{self.basename}.pkl"
        with open(pose_top_path, "rb") as f:
            pose_top = pkl.load(f)
        pose_top = torch.as_tensor(pose_top, dtype=torch.float32)[:, :, :2]
        self.pose_top = pose_top.reshape(pose_top.shape[0], -1)

        # pose_right: [T, 21, 2] -> [T, 42]
        pose_right_path = root / "hand_data" / "keyp_right" / f"{self.basename}.pkl"
        with open(pose_right_path, "rb") as f:
            pose_right = pkl.load(f)
        pose_right = torch.as_tensor(pose_right, dtype=torch.float32)[:, :, :2]
        self.pose_right = pose_right.reshape(pose_right.shape[0], -1)

        # 一致性检查
        T = self.emg.shape[0]
        if not (self.keystroke.shape[0] == T == self.pose_top.shape[0] == self.pose_right.shape[0]):
            raise ValueError(
                f"[{self.basename}] length mismatch: "
                f"emg={T}, ks={self.keystroke.shape[0]}, "
                f"pose_top={self.pose_top.shape[0]}, pose_right={self.pose_right.shape[0]}"
            )

        self.length = T

    def __len__(self) -> int:
        return self.length

    def get_window(self, start: int, end: int) -> dict[str, torch.Tensor]:
        """按帧范围返回一个窗口 [start, end) 的数据."""
        sl = slice(start, end)
        return {
            "emg": self.emg[sl],              # [t, C]
            "keystroke": self.keystroke[sl],  # [t, K]
            "pose_top": self.pose_top[sl],    # [t, 42]
            "pose_right": self.pose_right[sl] # [t, 42]
        }


@dataclass
class MultiSessionWindowedPianoKPMDataset(Dataset):
    """多 session Pianokpm 数据的 windowed dataset，风格对齐 MultiSessionWindowedEmgDataset.

    关键输出字段：
      - emg: [C, T]（与 emg2pose/emg2qwerty 保持一致）
      - label_valid_mask: [T]，这里暂时全 True（没有 IK failure 信息）
      - pose_top / pose_right: [42, T]
      - keystroke: [T, K]
      - window_start_idx / window_end_idx / session_idx
    """

    data_root: Path
    basenames: list[str]           # 每个 performance 的 basename（不含扩展名）
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False

    def __post_init__(self) -> None:
        assert self.window_length > 0
        self.stride = self.stride or self.window_length
        assert self.stride > 0

        self.left_padding, self.right_padding = self.padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        # session cache（按需加载）
        self._sessions: dict[int, PianoKPMSessionData] = {}

        (
            self._block_session_idx,
            self._block_start,
            self._block_end,
            self._block_cumsum,
        ) = self._build_blocks_index()

    def _get_session(self, si: int) -> PianoKPMSessionData:
        if si not in self._sessions:
            self._sessions[si] = PianoKPMSessionData(
                data_root=self.data_root,
                basename=self.basenames[si],
            )
        return self._sessions[si]

    def _session_len(self, si: int) -> int:
        # 用 PianoKPMSessionData 的长度；为减少 IO，也可以预先读一遍长度再缓存
        session = self._get_session(si)
        return len(session)

    def _build_blocks_index(self):
        block_session_idx: list[int] = []
        block_start: list[int] = []
        block_end: list[int] = []
        block_lengths: list[int] = []

        for si, _ in enumerate(self.basenames):
            slen = self._session_len(si)
            if slen < self.window_length:
                continue
            # 简化：每个 session 是一个 block [0, slen)
            start = 0
            end = slen
            n = (slen - self.window_length) // self.stride + 1
            if n <= 0:
                continue
            block_session_idx.append(si)
            block_start.append(start)
            block_end.append(end)
            block_lengths.append(n)

        if not block_lengths:
            # 没有任何可用窗口
            self._empty = True
            block_session_idx = np.asarray([], dtype=np.int32)
            block_start = np.asarray([], dtype=np.int64)
            block_end = np.asarray([], dtype=np.int64)
            block_cumsum = np.asarray([0], dtype=np.int64)
        else:
            self._empty = False
            block_session_idx = np.asarray(block_session_idx, dtype=np.int32)
            block_start = np.asarray(block_start, dtype=np.int64)
            block_end = np.asarray(block_end, dtype=np.int64)
            block_cumsum = np.cumsum(np.asarray([0] + block_lengths, dtype=np.int64))

        return block_session_idx, block_start, block_end, block_cumsum

    def __len__(self) -> int:
        return int(self._block_cumsum[-1])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # idx -> block
        bi = int(np.searchsorted(self._block_cumsum, idx, side="right") - 1)
        si = int(self._block_session_idx[bi])
        start_idx = int(self._block_start[bi])
        end_idx = int(self._block_end[bi])
        rel = int(idx - self._block_cumsum[bi])

        offset = start_idx + rel * self.stride

        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")

        if leftover > 0 and self.jitter:
            offset += int(np.random.randint(0, min(self.stride, leftover)))

        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding

        session = self._get_session(si)
        window = session.get_window(window_start, window_end)
        # window["emg"]: [t, C]；其余也是 [t, ...]
        emg = window["emg"].transpose(0, 1)           # -> [C, T]
        pose_top = window["pose_top"].transpose(0, 1) # -> [42, T]
        pose_right = window["pose_right"].transpose(0, 1)
        keystroke = window["keystroke"]               # [T, K]

        T = emg.shape[-1]
        label_valid_mask = torch.ones(T, dtype=torch.bool)

        return {
            "emg": emg,                      # [C, T]，与 emg2pose/emg2qwerty 对齐
            "pose_top": pose_top,            # [42, T]
            "pose_right": pose_right,        # [42, T]
            "keystroke": keystroke,          # [T, K]
            "label_valid_mask": label_valid_mask,  # [T]，这里全 True
            "window_start_idx": window_start,
            "window_end_idx": window_end,
            "session_idx": si,
            "basename": self.basenames[si],
        }