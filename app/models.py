"""Lightweight data classes shared across modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    n_frames: int


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int, int]) -> "ROI":
        return cls(int(t[0]), int(t[1]), int(t[2]), int(t[3]))

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def clip_to(self, width: int, height: int) -> "ROI":
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return ROI(x, y, w, h)
