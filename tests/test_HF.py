"""
Testing for the math.py module.
"""

import friendly_computing_machine as fcm
import pytest

def test_HF():
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)

# Build a molecule
mol.update_geometry()
mol.print_out()

# Build a basis
bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
bas.print_out()

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
