import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)

class HFcalc:
    def __init__(self, mol_, basis_ = "aug-cc-PVDZ", DIIS_ = False, dfJK = False):
        self.DIIS = DIIS_
        self.dfJK = dfJK
        if self.DIIS:
            self.diis_vectors = 7
        self.mol = mol_
        self.basis = psi4.core.BasisSet.build(self.mol, target=basis_)
        self.mints = psi4.core.MintsHelper(self.basis)
        self.nbf = self.mints.nbf()
        if (self.nbf > 100):
            raise Exception("More than 100 basis functions!")
        self.H = self.core_hamiltonian(self.mints)

        self.e_conv = 1.e-6
        self.d_conv = 1.e-6
        self.damp_value = 0.20
        self.damp_start = 3
        self.nel = 5
        self.S = np.array(self.mints.ao_overlap())
        self.g = np.array(self.mints.ao_eri())

        # Needed for Density fitting
        self.aux = psi4.core.BasisSet.build(self.mol, fitrole="JKFIT", other=basis_)
        self.zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
        Qls_tilde = self.mints.ao_eri(self.zero_bas, self.aux, self.basis, self.basis)
        Qls_tilde = np.squeeze(Qls_tilde) # remove the 1-dimensions
        metric = self.mints.ao_eri(self.zero_bas, self.aux, self.zero_bas, self.aux)
        metric.power(-0.5, 1.e-14)
        metric = np.squeeze(metric) # remove the 1-dimensions
        self.Pls = np.einsum('pq, qls->pls', metric, Qls_tilde)

        # Build Orthogonalizer
        self.A = self.mints.ao_overlap()
        self.A.power(-0.5, 1.e-14)
        self.A = np.array(self.A)
        self.E_old = 0.0
        # self.F_old = None
        self.count_iter = 0
        # self.E_diff = -1.0

    def core_hamiltonian(self, mints):
        # Build the core hamiltonian
        V = np.array(mints.ao_potential())
        T = np.array(mints.ao_kinetic())
        return T + V

    def get_JK(self, D):
        # Build the coloumb-repulsion and exchange integral tensors
        J = np.einsum("pqrs,rs->pq", self.g, D)
        K = np.einsum("prqs,rs->pq", self.g, D)
        return J, K

    def get_dfJK(self, Cocc, D):
        """
        Function that builds the density-fitted coulomb and exchange integral tensors
        """
        Pls = self.Pls
        # Compute J
        ChiP = np.einsum('pls,ls->p', Pls, D)
        J = np.einsum('pmn,p->mn', Pls, ChiP)
        # Compute K
        Eta1_Pmup = np.einsum('Pms,sp->Pmp', Pls, Cocc)
        Eta2_Pnup = np.einsum('Pnl,lp->Pnp', Pls, Cocc)
        K = np.einsum('Pmp,Pnp->mn', Eta1_Pmup, Eta2_Pnup)
        return J, K

    # Diagonalize Core H
    def diag(self, F):
        Fp = self.A.T @ F @ self.A
        eps, Cp = np.linalg.eigh(Fp)
        C = self.A @ Cp
        return eps, C

    def DIIS_step(self):
        B = -1*np.ones((self.diis_vectors+1, self.diis_vectors+1))
        B[-1,-1] = 0
        for i in range(self.diis_vectors):
            for j in range(i, self.diis_vectors):
                ri_dot_rj = np.vdot(self.r_array[i],self.r_array[j])
                # print(i,j,ri_dot_rj)
                B[i,j] = B[j,i] = ri_dot_rj
        # print(B)
        vec = np.zeros((self.diis_vectors+1))  # [:,None]])
        vec[-1] = -1
        coeff =  np.linalg.solve(B, vec)
        coeff = coeff[:-1]
        return np.einsum("i,ijk->jk", coeff, self.fock_array)

    def SCF(self):
        if self.DIIS:
            self.fock_array = np.zeros([self.diis_vectors]+list(self.H.shape))
            self.r_array = np.zeros([self.diis_vectors]+list(self.H.shape))
        eps, C = self.diag(self.H)
        Cocc = C[:, :self.nel]
        D = Cocc @ Cocc.T
        E_diff = -1.0
        for iteration in range(25):
            if self.dfJK:
                J, K = self.get_dfJK(Cocc, D)
            else:    
                J, K = self.get_JK(D)

            F_new = self.H + 2.0 * J - K

            if(E_diff > 0.0):
                self.count_iter += 1

            # conditional iteration > start_damp
            # if self.count_iter >= self.damp_start:
            #     F = self.damp_value * F_old + (1.0 - self.damp_value) * F_new
            # else:
            F = F_new

            F_old = F_new
            # F = (damp_value) Fold + (??) Fnew

            # Build the AO gradient
            grad = F @ D @ self.S - self.S @ D @ F

            grad_rms = np.mean(grad ** 2) ** 0.5

            if self.DIIS:
                if iteration < self.diis_vectors:
                    self.fock_array[iteration] = F
                    # r = self.A.T @ grad @ self.A
                    # print(r)
                    self.r_array[iteration] = self.A.T @ grad @ self.A
                else:
                    self.fock_array = np.roll(self.fock_array, -1, axis=0)
                    self.r_array = np.roll(self.r_array, -1, axis=0)
                    self.fock_array[-1] = F
                    self.r_array[-1] = self.A.T @ grad @ self.A
                if iteration > 4:
                    # print(self.r_array[-1])
                    F = self.DIIS_step()

            # Build the energy
            E_electric = np.sum((F + self.H) * D)
            E_total = E_electric + self.mol.nuclear_repulsion_energy()

            E_diff = E_total - self.E_old
            self.E_old = E_total
            print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
                    (iteration, E_total, E_diff, grad_rms))

            # Break if e_conv and d_conv are met
            if (E_diff < self.e_conv) and (grad_rms < self.d_conv):
                break

            eps, C = self.diag(F)
            Cocc = C[:, :self.nel]
            D = Cocc @ Cocc.T
        print("SCF has finished!\n")
        return E_total



def psi4_energy(mol):
    psi4.set_output_file("output.dat")
    psi4.set_options({"scf_type": "pk"})
    return psi4.energy("SCF/aug-cc-PVDZ", molecule=mol)
