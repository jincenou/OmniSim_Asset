"""Common entry for NEW project generation adapters; no changes to frozen runs."""
import argparse
import os
from pathlib import Path
import sys
from gate import check_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    check_root(a.root)  # Before importing torch/loading weights/creating outputs.
    if not a.adapter.is_file():
        raise FileNotFoundError(a.adapter)
    env = os.environ.copy()
    env['FDANYONE_WAYMO_ROOT'] = str(a.root.resolve())
    os.execve(sys.executable, [sys.executable, '-u', str(a.adapter), *a.args], env)

if __name__ == '__main__':
    main()
