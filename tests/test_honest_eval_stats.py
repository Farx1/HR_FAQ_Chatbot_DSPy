"""Checks for the interval and test helpers in evaluation/honest_eval.py."""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation")
)

from honest_eval import mcnemar_exact, paired_bootstrap, wilson  # noqa: E402


def test_wilson_known_values():
    lo, hi = wilson(8, 10)  # Wilson (1927) score interval, z = 1.96
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3
    assert wilson(0, 10)[0] == 0.0 and abs(wilson(10, 10)[1] - 1.0) < 1e-12


def test_mcnemar_exact():
    # 6 discordant pairs all in favour of the first system: p = 2 * 0.5**6
    assert abs(mcnemar_exact([True] * 6, [False] * 6)["p_two_sided"] - 0.03125) < 1e-9
    assert mcnemar_exact([True, False], [True, False])["p_two_sided"] == 1.0


def test_paired_bootstrap_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.random(40)
    res = paired_bootstrap(a, a + 0.1, rng)
    assert abs(res["delta"] - 0.1) < 1e-9 and res["p_one_sided"] < 0.01
    assert paired_bootstrap(a, a, rng)["p_one_sided"] == 1.0
