"""CLI execution must use its checkout even with another installation present."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(path.name for path in (ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("checkout_already_on_path", [False, True])
def test_cli_loads_its_own_config_with_another_checkout_on_path(
    tmp_path, script, checkout_already_on_path
):
    foreign = tmp_path / "another_installation"
    package = foreign / "qwen_lora_experiment"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('another checkout was imported')\n", encoding="utf-8"
    )
    code = """
import importlib
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
sys.path = [entry for entry in sys.path if entry != str(root)]
sys.path.insert(0, sys.argv[2])
if sys.argv[4] == 'True':
    sys.path.append(str(root))
runpy.run_path(str(root / 'scripts' / sys.argv[3]), run_name='cli_import_test')
module = importlib.import_module('qwen_lora_experiment.workflows.config')
assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
config = module.load_pilot_config(root / 'configs/lora_qwen_n4096_template.json')
assert config.method == 'grouped_quadratic'
assert config.tuning_mode == 'lora'
assert config.group_asset_sha256 is None
print(module.__file__)
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(ROOT),
            str(foreign),
            script,
            str(checkout_already_on_path),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid, pgid = process.pid, os.getpgid(process.pid)
    stdout, stderr = process.communicate()
    assert process.returncode == 0, (
        f"{script} pid={pid} pgid={pgid} exited={process.returncode}\n"
        f"{stdout}{stderr}"
    )
