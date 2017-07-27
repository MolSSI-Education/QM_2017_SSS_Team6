import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=8)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

mol.update_geometry()
mol.print_out()

e_conv = 1.e-8
e_diff = 1.e-7
nel = 5
damp_value = 0.20
damp_start = 5

# Build a basis set
bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
bas.print_out()
# help(psi4.core.BasisSet.build)

# Build a MintsHelper
mints = psi4.core.MintsHelper(bas)
nbf = mints.nbf()

if(nbf > 100):
    raise Exception("Too large basis set")

# print(mints.ao_potential())
V = np.array(mints.ao_potential())
T = np.array(mints.ao_kinetic())

# Core HAmiltonian
H = T + V

S = np.array(mints.ao_overlap())
g = np.array(mints.ao_eri())
# print (S.shape)
# print (I.shape)

A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)

# print( A @ S @ A )

Fp = A.T @ H @ A
diag = np.linalg.eigh(Fp)

eps, Cp = diag

# print(eps)
# print(Cp)

C = A @ Cp
Cocc = C[:, :nel]
D = Cocc @ Cocc.T


# Diagonalize Core H
def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


eps, C = diag(H, A)
Cocc = C[:, :nel]
D = Cocc @ Cocc.T

# print(np.allclose(tmp_C, C))

# raise Exception("Breakpoint")

E_old = 0.0
F_old = None
count_iter = 0
E_diff = -1.0

fock_list = []
r_list = []

for iteration in range(100):
    # F_pq = H_pq + 2 * g_pqrs D_rs - g_prqs D_rs
    # g = (7,7,7,7)
    # D = (1,1,7,7)

    # Jsum = np.sum(g * D, axis = (2,3) )
    J = np.einsum(" pqrs, rs -> pq ", g, D)
    K = np.einsum(" prqs, rs -> pq ", g, D)

    # print(Jsum)
    # print(Jein)

    F = H + 2.0 * J - K
    F_new = F
    # conditional iteration > start_damp
    # F = (damp_value) F_old + (??) F_new


    """ if(E_diff > 0.0):
        count_iter += 1

    if(count_iter >= damp_start):
        F = damp_value * F_old + (1.0 - damp_value) * F_new
        count_iter = 0

    F_old = F_new """

    # Build the AO gradient
    grad = F @ D @ S - S @ D @ F
    grad_rms = np.mean(grad ** 2) ** 0.5   # Every element is squared here

    E_electric = np.sum((F + H) * D)
    E_total = E_electric + mol.nuclear_repulsion_energy()
    E_diff = E_total - E_old
    print("% d % 16.12f % 8.8f % 8.8f" % (iteration, E_total, E_diff, grad_rms))
    E_old = E_total

    # break if convergence is met
    if(E_diff < e_conv) and (grad_rms < e_diff):
        break

    r = A.T @ grad @ A
    r_list.append(r)
    fock_list.append(F)

    if (iteration > 5):
        fock_list.pop(0)
        r_list.pop(0)
        B = np.dot(r.T,r)
        B = np.c_[B,-np.ones(len(B[0,:]))]
        B = np.r_[B,-np.ones(B.shape[1])[None,:]]
        B[-1,-1] = 0.
        # print(B.shape)
        vec = np.zeros(B.shape[1])  # [:,None]])
        # vec.pop(0)
        vec[-1] = -1
        # print(vec)
        coeff =  np.linalg.solve(B, vec) 
        # print(np.sum(coeff))
        # print()
        # print(coeff)
        # print(np.sum(coeff))
        coeff = coeff[:-1]
        print(coeff.shape)
        # print(coeff)
        # print(coeff[0])
        # print(fock_list[0])
        # print(coeff[0] * fock_list[0])
        # coeff_len = len(coeff) * len(coeff)
        F_new2 = np.zeros((len(coeff),len(coeff)))
        for j in range(len(coeff)-1):
            F_new2 += coeff[j] * fock_list[j]
        F = F_new2

    eps, C = diag(F, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T


print("SCF done. \n")

# print(fock_list)
# print()
# print(r_list)

# Exercise
# np.sum(np.diag(A@B)) == np.sum(A*B)

# psi4.set_options({"scf_type": "pk"})
# psi4_energy = psi4.energy("SCF/aug-cc-pVDZ", molecule=mol)

# print("Energy matchees psi4 %s" % np.allclose(psi4_energy, E_total))
