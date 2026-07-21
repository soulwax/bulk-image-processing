import numpy as np

from process import _apply_tone_curve


def test_apply_tone_curve_increases_midtone_contrast() -> None:
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    mapped = _apply_tone_curve(values)

    assert mapped[0] == 0.0
    assert mapped[-1] == 1.0
    assert mapped[2] > 0.5
    assert mapped[1] > 0.25
    assert mapped[3] < 0.75
