"""`MICRODUCK_CFG_HOOK` must reach the cfg mjlab's `train` actually builds.

`mjlab.scripts.train` binds `load_env_cfg` by name at import, so the hook only
works if train_hook wraps it first, from the `mjlab.tasks` plugin entry point.
A refactor that moved the install later would make the hook a silent no-op,
hence the fresh interpreters: they exercise the real import order.
"""

import os
import subprocess
import sys

_TASK = "Mjlab-Velocity-Flat-MicroDuck"

_HOOK = """
def apply(task_id, env_cfg, play):
    env_cfg.rewards["upright"].weight = 123.0
    env_cfg.rewards.pop("foot_slip")
    print("HOOK", task_id, play)
"""

_PROBE = f"""
from mjlab.scripts.train import load_env_cfg
cfg = load_env_cfg({_TASK!r})
print("WEIGHT", cfg.rewards["upright"].weight, "foot_slip" in cfg.rewards)
"""


def _run(hook_spec, code, env_var="MICRODUCK_CFG_HOOK"):
    env = {**os.environ, env_var: hook_spec}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)


def test_rl_hook_edits_the_agent_cfg_train_loads(tmp_path):
    hook = tmp_path / "rl_hook.py"
    hook.write_text(
        "def apply_agent(task_id, agent_cfg):\n"
        "    agent_cfg.actor.hidden_dims = (64, 32)\n"
        "    agent_cfg.algorithm.learning_rate = 3e-4\n")
    code = (
        "from mjlab.scripts.train import load_rl_cfg\n"
        f"cfg = load_rl_cfg({_TASK!r})\n"
        "print('AGENT', tuple(cfg.actor.hidden_dims), cfg.algorithm.learning_rate)\n"
    )
    out = _run(f"{hook}:apply_agent", code, env_var="MICRODUCK_RL_CFG_HOOK")
    assert out.returncode == 0, out.stderr
    assert "AGENT (64, 32) 0.0003" in out.stdout


def test_hook_edits_the_cfg_train_loads(tmp_path):
    hook = tmp_path / "hook.py"
    hook.write_text(_HOOK)
    out = _run(f"{hook}:apply", _PROBE)
    assert out.returncode == 0, out.stderr
    assert f"HOOK {_TASK} False" in out.stdout
    assert "WEIGHT 123.0 False" in out.stdout


def test_registered_cfg_is_untouched(tmp_path):
    hook = tmp_path / "hook.py"
    hook.write_text(_HOOK)
    code = (
        "import mjlab.tasks\n"
        "from mjlab.tasks.registry import _REGISTRY, load_env_cfg\n"
        f"load_env_cfg({_TASK!r})\n"
        f"print('REGISTERED', _REGISTRY[{_TASK!r}].env_cfg.rewards['upright'].weight)\n"
    )
    out = _run(f"{hook}:apply", code)
    assert out.returncode == 0, out.stderr
    assert "REGISTERED 123.0" not in out.stdout


def test_broken_hook_exits_instead_of_unregistering_tasks(tmp_path):
    out = _run(f"{tmp_path / 'missing.py'}:apply", "import mjlab.tasks\nprint('IMPORTED')")
    assert out.returncode != 0
    assert "IMPORTED" not in out.stdout
    assert "MICRODUCK_CFG_HOOK" in out.stderr
