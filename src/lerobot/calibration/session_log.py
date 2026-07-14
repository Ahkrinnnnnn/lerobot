# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mirror stdout/stderr to a session log file for post-run debugging."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


class _TeeStream:
    """Write to terminal and log file; delegate other attrs to the original stream."""

    def __init__(self, original: TextIO, log_file: TextIO) -> None:
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_file.write(data)
        self._original.flush()
        self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class SessionLog:
    """Tee ``stdout`` and ``stderr`` to ``log_path`` for the duration of a calibration session."""

    def __init__(self, log_path: Path, *, header_lines: tuple[str, ...] = ()) -> None:
        self.log_path = log_path
        self.header_lines = header_lines
        self._log_file: TextIO | None = None
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_file.write(f"=== calibration session log {ts} ===\n")
        for line in self.header_lines:
            self._log_file.write(f"{line}\n")
        self._log_file.write("\n")
        self._log_file.flush()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._log_file)
        sys.stderr = _TeeStream(self._orig_stderr, self._log_file)
        self._active = True

    def stop(self, *, exc: BaseException | None = None) -> None:
        if not self._active:
            return
        if exc is not None and self._log_file is not None:
            self._log_file.write(f"\n=== session ended with error: {type(exc).__name__}: {exc} ===\n")
            self._log_file.flush()

        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr

        if self._log_file is not None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log_file.write(f"\n=== session log closed {ts} ===\n")
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

        self._orig_stdout = None
        self._orig_stderr = None
        self._active = False
