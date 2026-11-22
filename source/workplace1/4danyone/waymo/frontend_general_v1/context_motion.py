"""Run existing raw-context inference with explicit 101-frame serialization."""
import os
from pathlib import Path
import sys

G = Path(__file__).resolve().parent
sys.path[:0] = [str(G.parents[1] / "repo"), str(G.parent / "scene20/r20261019/scripts")]
import motion
from context_motion_result import result_type

if __name__ == "__main__":
    motion.MotionResult = result_type(int(os.environ["FDANYONE_FRAMES"]))
    motion.main()
