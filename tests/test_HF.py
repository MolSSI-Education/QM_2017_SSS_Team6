"""
Testing for the math.py module.
"""

import qm6
import psi4
import pytest
import numpy as np

def test_HF():
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    mol.update_geometry()
    calc = qm6.HF.HFcalc(mol)
    assert np.allclose(qm6.HF.psi4_energy(mol), calc.SCF(mol))

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
