"""Keep `train <task> ... --hf-jobs` working, whoever owns the `train` script.

The flag used to live in a `train` console script of our own, declared in
`[project.scripts]` and documented as "shadowing" mjlab's. It shadows nothing:
mjlab 1.3.0 declares `train` too, two distributions declaring the SAME script
name is last-writer-wins at install time, and mjlab won — `uv sync` left
`mjlab.scripts.train:main` in `.venv/bin/train`, so our wrapper was never
invoked and `uv run train ... --hf-jobs` died on tyro's
`Unrecognized options: --hf-jobs` (2026-08-31). Nothing warns about it: the
install succeeds and the flag silently disappears.

So the flag is not implemented in a console script at all any more. It is
intercepted here, from the `mjlab.tasks` plugin entry point: mjlab's own
`mjlab/__init__.py` calls `_import_registered_packages()` at module scope,
which imports `mjlab_microduck.tasks` — and mjlab's `train` reaches that while
executing `from mjlab.scripts.train import main`, i.e. before its two-stage
tyro parse ever sees argv. That path is mjlab's own, so no install order can
take it away from us.

`uv run scripts/hf/train_hf.py <task> ...` calls `submit()` directly and stays
the escape hatch if this interception ever stops firing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FLAG = "--hf-jobs"

#: Set on the job's environment by ``hf_jobs.submit`` — inside the job,
#: ``uv run train`` must always mean "train locally".
_IN_JOB_ENV = "MICRODUCK_IN_HF_JOB"

#: ``/path/to/file.py:function`` — called as ``function(task_id, env_cfg, play)``
#: on every env cfg ``load_env_cfg`` returns, to edit it in place.
CFG_HOOK_ENV = "MICRODUCK_CFG_HOOK"

#: ``/path/to/file.py:function`` — called as ``function(task_id, agent_cfg)`` on every
#: rsl_rl cfg ``load_rl_cfg`` returns. Set it on every process that loads a checkpoint
#: trained with an edited network, or the runner is built with the wrong layer sizes.
RL_CFG_HOOK_ENV = "MICRODUCK_RL_CFG_HOOK"


def _invoked_as_train() -> bool:
    """True when argv[0] is mjlab's trainer (console script or `-m`).

    `play --hf-jobs` must NOT submit a training job; let that command's own
    parser reject the flag instead.
    """
    prog = Path(sys.argv[0]).name
    return prog.removesuffix(".py").removesuffix("-script") == "train"


def maybe_init_checkpoint() -> None:
    """Consume `--init-checkpoint <spec>` (checkpoints.py) on a local `train`; a no-op without it.

    Runs AFTER the HF Jobs check, so a `--hf-jobs` submission forwards the flag untouched and the
    job's own `train` consumes it remotely (with the job's HF token for private bundles).
    """
    if not _invoked_as_train() or not any(a.split("=", 1)[0] == "--init-checkpoint" for a in sys.argv[1:]):
        return
    from mjlab_microduck.checkpoints import consume_init_flag, install_init_checkpoint_patch

    sys.argv[1:] = consume_init_flag(sys.argv[1:])
    install_init_checkpoint_patch()


def maybe_install_cfg_hook() -> None:
    """Route `load_env_cfg` / `load_rl_cfg` through `$MICRODUCK_CFG_HOOK` / `$MICRODUCK_RL_CFG_HOOK`.

    Each is a no-op when its variable is unset. Lets an orchestrator (e.g. reward and experiment
    designs from rl-roboracer) edit a task's cfgs without a fork-side task per variant. Wrapping
    the loaders rather than the registered cfgs means the hook sees a fresh copy each time, before
    tyro's CLI overrides, so `--env.*` / `--agent.*` flags still win and `params/*.yaml` record the
    edited cfgs. Must run before `mjlab.scripts.train` does `from mjlab.tasks.registry import
    load_env_cfg, load_rl_cfg`, which is why it lives on this import path.

    A broken hook exits the process: mjlab's plugin loader swallows ordinary exceptions, which
    would leave every microduck task unregistered and the real error buried in a warning.
    """
    for env_var, loader in ((CFG_HOOK_ENV, "load_env_cfg"), (RL_CFG_HOOK_ENV, "load_rl_cfg")):
        spec = os.environ.get(env_var)
        if not spec:
            continue
        try:
            _install_hook(env_var, loader, spec)
        except Exception as exc:
            sys.exit(f"{env_var}={spec!r}: {type(exc).__name__}: {exc}")


def _install_hook(env_var: str, loader: str, spec: str) -> None:
    import importlib.util

    import mjlab.tasks.registry as registry

    original = getattr(registry, loader)
    if getattr(original, "_microduck_cfg_hook", None) == spec:
        return
    path, sep, func_name = spec.rpartition(":")
    if not sep or not path or not func_name:
        raise ValueError(f"{env_var} must be /path/to/file.py:function, got {spec!r}")
    module_spec = importlib.util.spec_from_file_location(f"microduck_{loader}_hook", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"{env_var}: cannot load {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    hook = getattr(module, func_name)

    if loader == "load_env_cfg":
        def wrapped(task_name: str, play: bool = False):
            cfg = original(task_name, play=play)
            hook(task_name, cfg, play)
            return cfg
    else:
        def wrapped(task_name: str):
            cfg = original(task_name)
            hook(task_name, cfg)
            return cfg

    wrapped._microduck_cfg_hook = spec
    setattr(registry, loader, wrapped)


def maybe_submit_to_hf_jobs() -> None:
    """Consume `--hf-jobs` and exit the process; a no-op without the flag.

    Called at import time of `mjlab_microduck.tasks`, so it runs inside mjlab's
    plugin loader. `SystemExit` is a `BaseException`, so it propagates through
    the loader's `except Exception` and out of `import mjlab` — the local
    trainer never starts.
    """
    if _FLAG not in sys.argv[1:]:
        return
    if os.environ.get(_IN_JOB_ENV):
        return
    if not _invoked_as_train():
        return

    from mjlab_microduck.hf_jobs import submit

    sys.exit(submit([a for a in sys.argv[1:] if a != _FLAG]))
