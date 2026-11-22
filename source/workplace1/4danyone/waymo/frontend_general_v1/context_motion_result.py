"""Explicit clip-length adapter; preserve every original motion check."""
from fdanyone.motion.result import MotionResult


def result_type(frame_count):
    if not isinstance(frame_count, int) or frame_count < 2:
        raise ValueError("Invalid context length")

    class ContextMotionResult(MotionResult):
        def validate(self, expected_frames=None):
            expected = frame_count if expected_frames is None else expected_frames
            if expected != frame_count:
                raise ValueError("Context clip length differs from frozen protocol")
            super().validate(expected_frames=expected)

    return ContextMotionResult
