import numpy as np

def mp2(g, eps, C, mints):
    """
    Function that computes conventional MP2 correlation energy
    Requires:
    g, the 4d 2e- integral tensor
    eps, the eigenvalues of the last HF Fock matrix, and
    C, the eigenvectors of the HF Fock matrix
    mints, the psi4 mints helper object
    """
    nocc = 5
    nbf = mints.nbf()
    nvirt = nbf - nocc
    O = slice(None, nocc)
    V = slice(nocc, None)
    # Transform 2-e integral tensor
    g_mo_1 = np.einsum('pQRS, pP -> PQRS',
        np.einsum('pqRS, qQ -> pQRS',
        np.einsum('pqrS, rR -> pqRS',
        np.einsum('pqrs, sS -> pqrS', g, C[:,V]), C[:,O]), C[:,V]), C[:,O])
    # Loop over our tensor and orbital energies and add energy contributions together 
    E_MP2 = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    e_denom = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b]
                    iajb = g_mo_1[i, a, j, b]
                    ibja = g_mo_1[i, b, j, a]
                    E_MP2 += ((iajb**2) - ((iajb - ibja) * iajb)) / e_denom
    return E_MP2

def df_mp2():
    """
    Function that computes density-fitted MP2 correlation energy
    """
    pass
