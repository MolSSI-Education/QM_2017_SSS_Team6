import psi4
import qm6

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")
mol.update_geometry()

calc = qm6.HF.HFcalc(mol, DIIS_=True)
print(calc.SCF())
print(qm6.HF.psi4_energy(mol))
