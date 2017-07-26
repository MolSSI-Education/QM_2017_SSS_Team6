import numpy as np
import psi4

def build_dfJK(mol, C, D):
    """
    Function that builds the density-fitted coulomb and exchange integral tensors
    """
    # Prevent psi4 from printing basis information every iteration
    psi4.core.be_quiet()    
    # Get basis set
    bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
    # Build the complementary JKFIT basis for the aug-cc-pVDZ basis (for example)
    aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other="aug-cc-pVDZ")
    # The zero basis set
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()    
    # Build instance of MintsHelper
    mints = psi4.core.MintsHelper(bas)
    # Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
    Qls_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
    Qls_tilde = np.squeeze(Qls_tilde) # remove the 1-dimensions
    # Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
    metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    metric.power(-0.5, 1.e-14)
    metric = np.squeeze(metric) # remove the 1-dimensions
    #TODO convert all einsums into np.dots to make code faster
    # Compute (P|ls)
    Pls = np.dot(metric, Qls_tilde.transpose(1,0,2))
    # Compute J
    ChiP = np.dot(Pls.reshape(Pls.shape[0],-1), D.reshape(-1))
    J = np.dot(Pls.transpose(1,2,0), ChiP)
    # Compute K 
    Eta1_Pmup = np.dot(Pls.transpose(2,0,1), C[:,:5]).transpose(1,0,2)
	Eta1_Pmup = Eta1_Pmup.transpose(1,0,2).reshape(Eta1_Pmup.shape[1],-1) # make it become a mu x Pp matrix
    Eta2_Pnup = np.dot(Pls, C[:,:5])
	Eta2_Pnup.transpose(1,0,2).reshape(Eta2_Pnup.shape[1],-1).T 		  # make it become a Pp x nu matrix
    K = np.dot(Eta1_Pmup, Eta2_Pnup)

    return J, K
