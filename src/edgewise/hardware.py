"""Detect the local hardware so every benchmark result is tied to the device it ran on."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

from pydantic import BaseModel

GB = 1024**3

# Memory macOS and everyday apps need before a model can load. Rough, conservative guess.
SYSTEM_RESERVE_GB = 3.5


class HardwareInfo(BaseModel):
    os: str
    machine: str
    chip: str
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    python: str

    @property
    def apple_silicon(self) -> bool:
        return self.os == "Darwin" and self.machine == "arm64"

    @property
    def model_budget_gb(self) -> float:
        """Memory we can spend on weights + KV cache without pushing the system into swap."""
        return max(0.0, round(self.ram_gb - SYSTEM_RESERVE_GB, 1))


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _ram_bytes() -> int:
    if value := _sysctl("hw.memsize"):
        return int(value)
    if hasattr(os, "sysconf"):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return 0


def detect() -> HardwareInfo:
    return HardwareInfo(
        os=platform.system(),
        machine=platform.machine(),
        chip=_sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        cpu_cores=os.cpu_count() or 0,
        ram_gb=round(_ram_bytes() / GB, 1),
        disk_free_gb=round(shutil.disk_usage(os.path.expanduser("~")).free / GB, 1),
        python=platform.python_version(),
    )
