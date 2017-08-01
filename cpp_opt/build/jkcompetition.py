import numpy as np
import psi4
import jk 
import time
# Make sure we get the same random array
np.random.seed(0)

# A hydrogen molecule
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a ERI tensor
basis = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
mints = psi4.core.MintsHelper(basis)
I = np.array(mints.ao_eri())
# Symmetric random density
nbf = I.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
ta = time.time()
J_ref = np.einsum("pqrs,rs->pq", I, D)
K_ref = np.einsum("prqs,rs->pq", I, D)
td = time.time()

# Your implementation
t0 = time.time()
J, K = jk.form_JK(I,D)  
t1 = time.time()

print(J-J_ref)
print(np.linalg.norm(J-J_ref))

# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
print("K is correct: %s" % np.allclose(K, K_ref))


print("our JK took ", t1-t0)
print("psi JK took ", td-ta)
