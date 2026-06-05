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

"""Lazy loader for the third-party ``CrpRobotPy`` native SDK."""

from __future__ import annotations

from typing import Any

_sdk_loaded = False


def ensure_crp_sdk_loaded() -> None:
    """Configure ``sys.path`` / ``LD_LIBRARY_PATH`` and preload helper ``.so`` files once."""
    global _sdk_loaded
    if _sdk_loaded:
        return
    from lerobot.tools.lib_loader import load_CrpRobotPy

    load_CrpRobotPy()
    _sdk_loaded = True


def import_crp_robot_py() -> tuple[Any, Any]:
    """Return ``(CrpRobotPy, RobotMode)`` after ensuring the SDK is on the import path."""
    ensure_crp_sdk_loaded()
    from CrpRobotPy import CrpRobotPy, RobotMode

    return CrpRobotPy, RobotMode
