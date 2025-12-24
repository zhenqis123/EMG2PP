from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl

from hydra.utils import to_absolute_path


@dataclass
class PianoKPMSessionData:
    """单个 PianokPM performance 的只读接口。
    数据来自若干 pkl 文件，而不是 HDF5：
      - keystroke_data/{basename}.pkl    -> keystroke
      - emg_data/{basename}.pkl          -> emg
      - hand_data/keyp_top/{basename}.pkl   -> pose_top
      - hand_data/keyp_right/{basename}.pkl -> pose_right
    """

    KEYPST_TOP_DIR: ClassVar[str] = "hand_data/keyp_top"
    KEYPST_RIGHT_DIR: ClassVar[str] = "hand_data/keyp_right"
    KEYSTROKE_DIR: ClassVar[str] = "keystroke_data"
    EMG_DIR: ClassVar[str] = "emg_data"

    data_root: Path
    basename: str  # 不含 .pkl

    def __post_init__(self) -> None:
        root = Path(to_absolute_path(str(self.data_root)))

        # keystroke: [T, K?]
        ks_path = root / self.KEYSTROKE_DIR / f"{self.basename}.pkl"
        with open(ks_path, "rb") as f:
            ks = pkl.load(f)
        self.keystroke = torch.as_tensor(ks, dtype=torch.float32)

        # emg: [T, C]
        emg_path = root / self.EMG_DIR / f"{self.basename}.pkl"
        with open(emg_path, "rb") as f:
            emg = pkl.load(f)
        self.emg = torch.as_tensor(emg, dtype=torch.float32)

        # pose_top: [T, 21, 2] -> [T, 42]
        pose_top_path = root / self.KEYPST_TOP_DIR / f"{self.basename}.pkl"
        with open(pose_top_path, "rb") as f:
            pose_top = pkl.load(f)
        pose_top = torch.as_tensor(pose_top, dtype=torch.float32)[:, :, :2]
        self.pose_top = pose_top.reshape(pose_top.shape[0], -1)

        # pose_right: [T, 21, 2] -> [T, 42]
        pose_right_path = root / self.KEYPST_RIGHT_DIR / f"{self.basename}.pkl"
        with open(pose_right_path, "rb") as f:
            pose_right = pkl.load(f)
        pose_right = torch.as_tensor(pose_right, dtype=torch.float32)[:, :, :2]
        self.pose_right = pose_right.reshape(pose_right.shape[0], -1)

        # 一致性检查
        T = self.emg.shape[0]
        if not (
            self.keystroke.shape[0] == T
            == self.pose_top.shape[0]
            == self.pose_right.shape[0]
        ):
            raise ValueError(
                f"[{self.basename}] length mismatch: "
                f"emg={T}, ks={self.keystroke.shape[0]}, "
                f"pose_top={self.pose_top.shape[0]}, "
                f"pose_right={self.pose_right.shape[0]}"
            )

        self.length = T

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key: slice) -> dict[str, torch.Tensor]:
        """按时间 slice 返回一个 dict，类似 Emg2PoseSessionData.__getitem__."""
        if not isinstance(key, slice):
            raise TypeError("PianoKPMSessionData only supports slice indexing")
        return {
            "emg": self.emg[key],              # [t, C]
            "keystroke": self.keystroke[key],  # [t, K]
            "pose_top": self.pose_top[key],    # [t, 42]
            "pose_right": self.pose_right[key] # [t, 42]
        }


@dataclass
class WindowedPianoKPMEmgDataset(Dataset):
    """单 session PianokPM 的 windowed Dataset，风格对齐 WindowedEmgDataset.

    输出：
      - emg: [C, T]
      - pose_top / pose_right: [42, T]
      - keystroke: [T, K]
      - label_valid_mask: [T]（这里没有 IK 信息，先全 True）
      - window_start_idx / window_end_idx
    """

    data_root: Path
    basename: str

    window_length: InitVar[int | None] = 10_000
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False

    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ) -> None:
        # 懒加载 session，以便 __len__ / blocks 使用
        self._session: PianoKPMSessionData | None = None

        self.session_length = len(self.session)
        self.window_length = (
            window_length if window_length is not None else self.session_length
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        self._blocks: list[tuple[int, int]] | None = None
        self._cumsum_blocks: np.ndarray | None = None
        self.windows: list[tuple[int, int]] = self.precompute_windows()

    @property
    def session(self) -> PianoKPMSessionData:
        if self._session is None:
            self._session = PianoKPMSessionData(self.data_root, self.basename)
        return self._session

    @property
    def blocks(self) -> list[tuple[int, int]]:
        """和 WindowedEmgDataset 一样，用 blocks 表示连续有效区间。
        这里没有 IK 掩码，直接用一个 block: (0, len(session)).
        """
        if self._blocks is None:
            if len(self.session) < self.window_length:
                self._blocks = []
            else:
                self._blocks = [(0, len(self.session))]
        return self._blocks

    def _get_block_len(self, block: tuple[int, int]) -> int:
        start, end = block
        return (end - start - self.window_length) // self.stride + 1

    def precompute_windows(self) -> list[tuple[int, int]]:
        block_lengths = [self._get_block_len(b) for b in self.blocks]
        self._cumsum_blocks = np.cumsum([0] + block_lengths)
        return []  # 不提前存每个窗口

    def __len__(self) -> int:
        return int(self._cumsum_blocks[-1])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # idx -> block & offset（同 WindowedEmgDataset）
        block_idx = int(np.searchsorted(self._cumsum_blocks, idx, side="right") - 1)
        start_idx, end_idx = self.blocks[block_idx]
        relative_idx = int(idx - self._cumsum_blocks[block_idx])
        offset = start_idx + relative_idx * self.stride

        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds, leftover {leftover}")
        if leftover > 0 and self.jitter:
            offset += int(np.random.randint(0, min(self.stride, leftover)))

        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding

        window = self.session[slice(window_start, window_end)]
        # window["emg"]: [t, C]
        emg = window["emg"].transpose(0, 1)          # -> [C, T]
        pose_top = window["pose_top"].transpose(0, 1)# -> [42, T]
        pose_right = window["pose_right"].transpose(0, 1)
        keystroke = window["keystroke"]              # [T, K]

        T = emg.shape[-1]
        label_valid_mask = torch.ones(T, dtype=torch.bool)

        return {
            "emg": emg,                      # CT
            "pose_top": pose_top,            # [42, T]
            "pose_right": pose_right,        # [42, T]
            "keystroke": keystroke,          # [T, K]
            "label_valid_mask": label_valid_mask,  # [T]
            "window_start_idx": window_start,
            "window_end_idx": window_end,
        }