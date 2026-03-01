import numpy as np
from itertools import product
from copy import deepcopy
from HartreeFockTrigonal import TrigonalDFT2
from dataclasses import dataclass
from pathlib import Path
from HartreeFock import HartreeFock


GAMMA_TRUNCATION = [(-6, -3), (-5, -4), (-5, -3), (-5, -2), (-5, -1), 
                    (-4, -4), (-4, -3), (-4, -2), (-4, -1), (-4, 0), (-4, 1), 
                    (-3, -5), (-3, -4), (-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), 
                    (-2, -5), (-2, -4), (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3), 
                    (-1, -4), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), 
                    (0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), 
                    (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), 
                    (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), 
                    (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), 
                    (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), 
                    (5, 3), (5, 4), (6, 3)]

def IP(arr1, arr2):
    return np.sum(arr1 * arr2)
def dagger(arr):
    return np.transpose(np.conjugate(arr))
def similarity_transformation(transformed, transformation):
    return dagger(transformation) @ transformed @ transformation
def is_hermitian(arr):
    return np.allclose(arr, dagger(arr))
def assert_real(a):
    assert np.allclose(np.imag(a), 0)
    return np.real(a)

pi = np.pi


@dataclass
class HFSolution:
    converged: bool
    iterations: int
    energy_tol: float
    density_tol: float  # stores the final relative Frobenius deltaC (Eq. 5.7)
    chemical_potential: float | None
    eigenvalues: np.ndarray  # shape (N**2, dim)
    occupancies: np.ndarray  # shape (N**2, dim)
    Ck: np.ndarray  # shape (N**2, dim, dim)
    htb_k: np.ndarray  # shape (N**2, dim, dim)
    h_k: np.ndarray  # shape (N**2, dim, dim)
    rho_local: np.ndarray  # shape (dim, dim), averaged over k
    total_energy: float  # Trigonal-style: per-filled-band mean energy (or FD-weighted sum / numFilledBands)
    energy_per_cell: float
    energy_per_particle: float | None
    hf_identity_energy_total: float  # Eq. (4.2) c-number term (system total)
    hf_identity_energy_per_cell: float
    total_energy_with_identity_per_cell: float  # band-energy-per-cell + Eq. (4.2) identity term
    total_energy_with_identity_per_particle: float | None

class HF:
    def __init__(self, path, nu, N, U0=0, metal=True, kT=0.005):
        self.path = path
        self.nu = nu
        self.U0 = U0
        self.N = N
        self.metal = metal
        self.kT = kT

        self.Nocc = self.nu * self.N**2
        self.dim = 4

        self.read_path()
        self.define_convention()
        self.build_hopping()
        self.build_Htb()

    def define_convention(self):
        self.A = {}
        self.G = {}

        self.A[1] = np.array([1, 0])
        self.A[2] = np.array([np.cos(2 * pi / 3), np.sin(2 * pi / 3)])
        prefac = 4 * pi / np.sqrt(3)
        self.G[1] = prefac * np.array([np.cos(pi / 6), np.sin(pi / 6)])
        self.G[2] = prefac * np.array([0, 1])

        for i1, i2 in product(*([[1, 2]]*2)):
            assert np.isclose(IP(self.A[i1], self.G[i2]), 2 * pi * int(i1==i2))
        
        self.kGridToIndex = {}
        for n1, n2 in product(*([range(self.N)]*2)):
            self.kGridToIndex[n1, n2] = n1 * self.N + n2
        self.indexToKGrid = {v: k for k, v in self.kGridToIndex.items()}

    @staticmethod
    def load_csv_hermitian(path, delimiter, dtype):
        m = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return (m + dagger(m)) / 2

    def read_path(self):
        self.C3z = np.loadtxt(f'{self.path}/C3z.csv', delimiter=",", dtype=complex)
        self.Inverse = np.loadtxt(f'{self.path}/Inverse.csv', delimiter=",", dtype=complex)
        self.H0 = self.load_csv_hermitian(f'{self.path}/H0.csv', delimiter=",", dtype=complex)
        self.H1 = self.load_csv_hermitian(f'{self.path}/H1.csv', delimiter=",", dtype=complex)
        self.H2 = self.load_csv_hermitian(f'{self.path}/H2.csv', delimiter=",", dtype=complex)
        self.H3 = self.load_csv_hermitian(f'{self.path}/H3.csv', delimiter=",", dtype=complex)
        assert np.all(np.array(self.C3z.shape) == self.dim)
        assert np.all(np.array(self.Inverse.shape) == self.dim)
        assert np.all(np.array(self.H0.shape) == self.dim)
        assert np.all(np.array(self.H1.shape) == self.dim)
        assert np.all(np.array(self.H2.shape) == self.dim)
        assert np.all(np.array(self.H3.shape) == self.dim)

    @staticmethod
    def rotate3(j1, j2):
        return (-j2, j1-j2)
    @staticmethod
    def inversion(j1, j2):
        return (-j1, -j2)
    def build_hopping(self):
        self.hopping = {}
        self.hopping[0, 0] = self.H0
        self.hopping[1, 0] = self.H1
        self.hopping[1, 2] = self.H2
        self.hopping[2, 0] = self.H3
        initial_hopping_keys = deepcopy(list(self.hopping.keys()))
        for k in initial_hopping_keys:
            if not k == (0, 0):
                new_k = self.rotate3(*k)
                self.hopping[new_k] = similarity_transformation(self.hopping[k], self.C3z)
                new_k = self.rotate3(*self.rotate3(*k))
                self.hopping[new_k] = similarity_transformation(self.hopping[k], dagger(self.C3z))
        initial_hopping_keys = deepcopy(list(self.hopping.keys()))
        for k in initial_hopping_keys:
            if not k == (0, 0):
                new_k = self.inversion(*k)
                self.hopping[new_k] = similarity_transformation(self.hopping[k], self.Inverse)

    def HKtb(self, k):
        htb = np.zeros(shape=(self.dim, self.dim), dtype=complex)
        for key, val in self.hopping.items():
            DeltaR = key[0] * self.A[1] + key[1] * self.A[2]
            htb += np.exp(-1j * IP(k, DeltaR)) * val
        assert np.allclose(htb, dagger(htb))
        return htb

    def build_Htb(self):
        self.Htb = np.zeros(shape=(self.N**2, self.dim, self.dim), dtype=complex)
        for idx, grid in self.indexToKGrid.items():
            n1, n2 = grid
            self.Htb[idx, :, :] = self.HKtb(n1 * self.G[1] / self.N + n2 * self.G[2] / self.N)

    def find_mu(self, vals):
        low_bound = np.min(vals) - 1
        high_bound = np.max(vals) + 1
        it_ = 0
        while it_ < 100:
            it_ += 1
            mu = (low_bound + high_bound) / 2
            Nocc = np.sum(1 / (1 + np.exp((vals - mu)/self.kT)))
            if Nocc > self.Nocc:
                high_bound = mu
            else:
                low_bound = mu
            if ((high_bound - low_bound) < np.std(vals) * 1e-10) and (np.abs(Nocc - self.Nocc) < 1e-5):
                break
        mu = (low_bound + high_bound) / 2
        assert np.isclose(np.imag(mu), 0)
        return np.real(mu)

    def occupancies_from_energies(self, eigvals):
        if self.metal:
            vals = eigvals.reshape(-1)
            mu = self.find_mu(vals)
            occ = 1.0 / (np.exp((vals - mu) / self.kT) + 1)
            return occ.reshape(eigvals.shape), float(mu)
        else:
            vals = eigvals.reshape(-1)
            filled = np.argsort(vals)[:self.Nocc]
            occ = np.zeros_like(vals, dtype=float)
            occ[filled] = 1.0
            assert np.sum(occ) == self.Nocc
            occ = occ.reshape(eigvals.shape)
            return occ, None

    def density_from_eigensystem(self, eigvecs, occ):
        Ck = np.zeros(shape=eigvecs.shape, dtype=complex)
        for i in range(eigvecs.shape[0]):
            v = eigvecs[i, :, :]
            Ck[i, :, :] = v.conj() @ np.diag(occ[i]) @ v.T
            Ck[i] = 0.5 * (Ck[i] + Ck[i].conj().T)
        return Ck

    def build_Ck(self, Ck0=None, random_seed: int | None = None):
        if not random_seed == None:
            rng = np.random.default_rng(random_seed)
            noise = rng.normal(scale=1e-6, size=(self.N**2, self.dim, self.dim))
            noise = 0.5 * (noise + noise.swapaxes(-1, -2).conj())
        else:
            noise = 0
        if Ck0 is None:
            eigvals, eigvecs = self.diagonalize_blocks(self.Htb + noise)
            occ, _ = self.occupancies_from_energies(eigvals)
            return self.density_from_eigensystem(eigvecs, occ)
        else:
            return Ck0

    def diagonalize_blocks(self, h):
        eigvals = np.zeros(shape=(h.shape[0], self.dim))
        eigvecs = np.zeros(shape=(h.shape[0], self.dim, self.dim), dtype=complex)
        assert np.allclose(np.conjugate(np.transpose(h, axes=[0, 2, 1])), h)
        for i in range(h.shape[0]):
            vals, vecs = np.linalg.eigh(h[i])
            eigvals[i, :] = vals
            eigvecs[i, :, :] = vecs
        return eigvals, eigvecs

    def nearest_neighbor_electron_repulsion(self, Ck: np.ndarray) -> np.ndarray:
        # rho_{sigma,sigma'} = (1/N^2) sum_k C_k[sigma,sigma']
        rho = np.mean(Ck, axis=0)
        # Exact on-site U0 Hartree-Fock single-particle shift from Eq. (4.2):
        #   h_HF = U0 * Tr(rho) * I - U0 * rho^T
        # using the note convention C_{σσ'} = <c^†_σ c_{σ'}>.
        nbar = float(np.trace(rho).real)
        h_hf = self.U0 * nbar * np.eye(self.dim, dtype=np.complex128) - self.U0 * rho.T
        h_hf = 0.5 * (h_hf + h_hf.conj().T)
        return h_hf, np.sum(h_hf * rho) / 2

    @staticmethod
    def relative_frobenius_delta(new: np.ndarray, old: np.ndarray) -> float:
        num = float(np.linalg.norm((new - old).ravel()))
        den = float(np.linalg.norm(old.ravel()))
        if den == 0.0:
            return num
        return num / den

    def solve(
        self,
        max_iter: int = 200,
        alpha: float = 0.2,
        tol_dC: float = 1e-6,
        random_seed: int | None = None,
        Ck0: np.ndarray | None = None,
        verbose: bool = False,
    ) -> HFSolution:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("mix must satisfy 0 < mix <= 1.")

        if Ck0 is None:
            Ck = self.build_Ck(random_seed=random_seed)
        else:
            Ck = np.array(Ck0, dtype=np.complex128, copy=True)
            if Ck.shape != (self.N**2, self.dim, self.dim):
                raise ValueError(f"Ck0 must have shape {(self.N**2, self.dim, self.dim)}.")

        for it in range(1, max_iter + 1):
            h_hf, e_hf = self.nearest_neighbor_electron_repulsion(Ck)
            h_k = self.Htb + h_hf[None, :, :]
            evals, vecs = self.diagonalize_blocks(h_k)
            evals = evals - e_hf
            evals = assert_real(evals)
            occ, mu = self.occupancies_from_energies(evals)
            occ = assert_real(occ)
            C_new = self.density_from_eigensystem(vecs, occ)

            C_mixed = (1.0 - alpha) * Ck + alpha * C_new
            C_mixed = 0.5 * (C_mixed + C_mixed.swapaxes(-1, -2).conj())

            dC = self.relative_frobenius_delta(C_mixed, Ck)
            e_mean = np.sum(evals * occ) / self.N**2
            e_mean = assert_real(e_mean)

            Ck_sum = np.sum(Ck, axis=0)
            n_sum = np.diag(Ck_sum)
            spin_components = assert_real(np.sort([n_sum[0]+n_sum[2], n_sum[1]+n_sum[3]]))


            if verbose:
                print(
                    f"e_mean={np.round(e_mean, 12):<20} dC_rel={dC:.3e}   {np.round(spin_components[1], 1)}/{self.Nocc}"
                )

            Ck = C_mixed

            if dC < tol_dC:
                converged = True
                break

        return h_k, Ck, converged

    def build_effective_hopping(self, h_k):
        effective_hopping = {}
        for j1, j2 in GAMMA_TRUNCATION:
            DeltaR = self.A[1] * j1 + self.A[2] * j2
            tilde_gamma = np.zeros((self.dim, self.dim), dtype=complex)
            for idx, grid in self.indexToKGrid.items():
                n1, n2 = grid[0], grid[1]
                k = n1 * self.G[1] / self.N + n2 * self.G[2] / self.N
                exponent = np.exp(1j * IP(k, DeltaR))
                tilde_gamma += exponent * h_k[idx, :, :]
            effective_hopping[j1, j2] = tilde_gamma / self.N**2
        return effective_hopping
    
    def HKtbEff(self, k, effective_hopping):
        htb = np.zeros(shape=(self.dim, self.dim), dtype=complex)
        for key, val in effective_hopping.items():
            DeltaR = key[0] * self.A[1] + key[1] * self.A[2]
            htb += np.exp(-1j * IP(k, DeltaR)) * val
        assert np.allclose(htb, dagger(htb))
        return htb


if __name__ == '__main__':
    U0_ = 0.1
    metal_ = True

    model = HF(path='TightBindingModel/Re2CoO8/withSOCwannier-dim2', 
               nu=1, U0=U0_, N=12, metal=metal_)

    # hf_dft = TrigonalDFT2(path='TightBindingModel/Re2CoO8/withSOCwannier-dim2', nu=1, U0=U0_, N=12, metal=metal_)

    # tb_path = Path("TightBindingModel/Re2CoO8/withSOCwannier-dim2")
    # if tb_path.exists():
    #     hf = HartreeFock(tb_path, nu=1, U0=U0_, N=12, metal=metal_)

    # ############################################################
    # for k, v in model.hopping.items():
    #     assert np.allclose(v, hf_dft.Hamiltonian[k]['H']), (k, v - hf_dft.Hamiltonian[k]['H'])
    # for k, v in hf_dft.Hamiltonian.items():
    #     assert np.allclose(model.hopping[k], v['H']), (k, v - hf_dft.Hamiltonian[k]['H'])
    
    # for _ in range(30):
    #     r1, r2 = np.random.rand(2)
    #     assert np.allclose(model.HKtb(r1*model.G[1]+r2*model.G[2]), hf_dft.Hk(r1*hf_dft.G[1]+r2*hf_dft.G[2]))
    
    # hf_dft.genInitialTwoPointCorrelation()
    # Ck = hf_dft.twoPointCorrelation
    # h_dft = hf_dft.U0 * (hf_dft.genHamiltonianHartreeU0() + hf_dft.genHamiltonianFockU0())
    # h_hf, e_hf = model.nearest_neighbor_electron_repulsion(Ck)
    # assert np.allclose(h_hf, h_dft[0, :, :])

    # h_k = model.Htb + h_hf[None, :, :]
    # evals, vecs = model.diagonalize_blocks(h_k)
    # evals = evals - e_hf
    # hf_dft.calculateSpectrum()
    # assert np.allclose(np.sort(evals.reshape(-1)), hf_dft.eigvals)

    # h_hf_ = hf._hartree_fock_shift(Ck)
    # h_k_ = hf.htb_k + h_hf[None, :, :]
    # evals_, vecs_ = hf._diagonalize_blocks(h_k_)
    # occ_, mu_ = hf._occupancies_from_energies(evals_)
    # C_new_ = hf._density_from_eigensystem(vecs_, occ_)

    
    # ############################################################

    # occ, mu = model.occupancies_from_energies(evals)
    # C_new = model.density_from_eigensystem(vecs, occ)

    # assert np.allclose(vecs, vecs_)
    # assert np.allclose(occ, occ_)
    # assert np.allclose(C_new, C_new_)

    # hf_dft.updateTwoPointCorrelation(0)
    # assert np.allclose(C_new, hf_dft.tildeD)
    # assert np.allclose(np.sort(occ.reshape(-1).real), np.sort(hf_dft.FD_distribution(hf_dft.eigvals, hf_dft.mu).real))
    
    h_k, Ck, converged = model.solve(max_iter=3000, alpha=1, verbose=True)
    print(f'convergence: {converged}')

    effective_hopping = model.build_effective_hopping(h_k)
    for idx, grid in model.indexToKGrid.items():
        k = grid[0] * model.G[1] / model.N + grid[1] * model.G[2] / model.N
        eigvals1 = np.linalg.eigh(h_k[idx, :, :])[0]
        eigvals2 = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        assert np.allclose(eigvals1, eigvals2)

    
