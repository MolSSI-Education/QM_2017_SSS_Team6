"""
Testing for the math.py module.
"""

import qm6
import pytest
import numpy as np

def test_HF():
    assert np.allclose(qm6.HF.psi4_energy(), qm6.HF.SCF())
# testdata  = [
#     (2, 5, 10),
#     (1, 2, 2),
#     (11, 9, 99),
#     (11, 0, 0),
#     (0, 0, 0),
# ]
# @pytest.mark.parametrize("a,b,expected", testdata)
# def test_mult(a, b, expected):
#     assert fcm.math.mult(a, b) == expected
#     assert fcm.math.mult(b, a) == expected
#
#     Contact GitHub API Training Shop Blog About
