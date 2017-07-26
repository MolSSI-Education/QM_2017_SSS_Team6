import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)

def mol():
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)

    # Build a molecule
    mol.update_geometry()
    mol.print_out()
    return mol

def basis(mol):
    # Build a basis
    bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
    return bas

def mints(bas):
    # Build a MintsHelper
    mints = psi4.core.MintsHelper(bas)
    return mints

def core_hamiltonian(mints):
    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())
    return T + V

def get_JK(g, D):
    J = np.einsum("pqrs,rs->pq", g, D)
    K = np.einsum("prqs,rs->pq", g, D)
    return J, K

mol = mol()
basis = basis(mol)
mints = mints(basis)

H = core_hamiltonian(mints)

e_conv = 1.e-6
d_conv = 1.e-6
nel = 5
damp_value = 0.20
damp_start = 3
nbf = mints.nbf()



if (nbf > 100):
    raise Exception("More than 100 basis functions!")


S = np.array(mints.ao_overlap())
g = np.array(mints.ao_eri())

# print(S.shape)
# print(I.shape)

A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)

# print(A @ S @ A)


# Diagonalize Core H
def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


eps, C = diag(H, A)
Cocc = C[:, :nel]
D = Cocc @ Cocc.T

E_old = 0.0
F_old = None
count_iter = 0
E_diff = -1.0
for iteration in range(25):
    J, K = get_JK(g, D)

    F_new = H + 2.0 * J - K

    if(E_diff > 0.0):
        count_iter += 1

    # conditional iteration > start_damp
    if count_iter >= damp_start:
        F = damp_value * F_old + (1.0 - damp_value) * F_new
    else:
        F = F_new

    F_old = F_new
    # F = (damp_value) Fold + (??) Fnew

    # Build the AO gradient
    grad = F @ D @ S - S @ D @ F

    grad_rms = np.mean(grad ** 2) ** 0.5

    # Build the energy
    E_electric = np.sum((F + H) * D)
    E_total = E_electric + mol.nuclear_repulsion_energy()

    E_diff = E_total - E_old
    E_old = E_total
    print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
            (iteration, E_total, E_diff, grad_rms))

    # Break if e_conv and d_conv are met
    if (E_diff < e_conv) and (grad_rms < d_conv):
        break

    eps, C = diag(F, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

print("SCF has finished!\n")

def psi4_energy():
    psi4.set_output_file("output.dat")
    psi4.set_options({"scf_type": "pk"})
    return psi4.energy("SCF/aug-cc-pVDZ", molecule=mol)

psi4_energy = psi4_energy()
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))
