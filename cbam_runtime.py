"""Runtime-only CBAM injection for Ultralytics (same pattern as yolo_test.py / ripvis_vis_pred.py)."""

from __future__ import annotations

import importlib
import inspect
import textwrap


def apply_runtime_cbam_patch() -> None:
    import ultralytics.nn.tasks as tasks_module
    from ultralytics.nn.modules import CBAM

    tasks_module.__dict__["CBAM"] = CBAM
    source = inspect.getsource(tasks_module.parse_model)
    if "elif m is CBAM:" in source:
        return

    anchor = "        else:\n            c2 = ch[f]"
    cbam_branch = (
        "        elif m is CBAM:\n"
        "            c2 = ch[f]\n"
        "            args = [c2]\n"
        "        else:\n"
        "            c2 = ch[f]"
    )
    if anchor not in source:
        raise RuntimeError("Unable to apply runtime patch: parse_model anchor block not found.")

    patched_source = source.replace(anchor, cbam_branch, 1)
    exec(textwrap.dedent(patched_source), tasks_module.__dict__)


def reload_and_inject_cbam_runtime() -> None:
    import ultralytics.nn.modules
    import ultralytics.nn.tasks
    import ultralytics.nn.tasks as tasks_module
    from ultralytics.nn.modules import CBAM

    importlib.reload(ultralytics.nn.modules)
    importlib.reload(ultralytics.nn.tasks)
    tasks_module.__dict__["CBAM"] = CBAM
    apply_runtime_cbam_patch()
