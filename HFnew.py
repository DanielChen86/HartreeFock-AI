import numpy as np
from itertools import product
from copy import deepcopy
from HartreeFockTrigonal import TrigonalDFT2
from dataclasses import dataclass
from pathlib import Path
from HartreeFock import HartreeFock
import datetime
import matplotlib.pyplot as plt


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
def interpolation(a, b, N, i):
    return a + (b - a) / N * i

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
    def __init__(self, path, nu, N, U0=0, Un=0, Vupdown=0, metal=True, kT=0.005):
        self.path = path
        self.nu = nu
        self.U0 = U0
        self.Un = Un
        self.Vupdown = Vupdown
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

        self.Gamma = 0 * self.G[1]
        self.K = (1/3) * self.G[1] + (1/3) * self.G[2]
        self.M = (1/2) * self.G[2]

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

    def build_Ck(
        self,
        Ck0=None,
        random_seed: int | None = None,
        use_reference_filling: bool = False,
        deviation: float = 1e-2,
    ):
        if Ck0 is not None:
            return Ck0

        rng = np.random.default_rng(random_seed)
        Ck = np.zeros((self.N**2, self.dim, self.dim), dtype=np.complex128)

        if use_reference_filling:
            reference_diag = np.diag(np.sum(self.reference_Ck(), axis=0)).real / self.Nocc
        else:
            reference_diag = np.full(self.dim, self.nu / self.dim, dtype=float)

        for idx in range(self.N**2):
            filling_deviation = deviation * rng.random(self.dim)
            filling_deviation = filling_deviation - np.mean(filling_deviation)
            filling = reference_diag + filling_deviation
            if not np.all(filling > 0):
                raise ValueError("Negative filling number in initial Ck.")

            Ck[idx, np.arange(self.dim), np.arange(self.dim)] = filling

            off_diagonal = (
                rng.random((self.dim, self.dim)) + 1j * rng.random((self.dim, self.dim))
            ) * deviation
            np.fill_diagonal(off_diagonal, 0)
            off_diagonal = 0.5 * (off_diagonal + off_diagonal.conj().T)
            Ck[idx] += off_diagonal

            Ck[idx] = 0.5 * (Ck[idx] + Ck[idx].conj().T)
            if not np.all(np.linalg.eigvalsh(Ck[idx]) > 0):
                raise ValueError("Initial Ck must be positive definite.")
            if not np.all(np.linalg.eigvalsh(np.eye(self.dim) - Ck[idx]) > 0):
                raise ValueError("Initial Ck must satisfy Ck < I.")

        total_trace = np.trace(Ck, axis1=1, axis2=2).sum().real
        if not np.isclose(total_trace, self.N**2 * self.nu):
            raise ValueError("Initial Ck violates the particle-number constraint.")
        return Ck

    def diagonalize_blocks(self, h):
        eigvals = np.zeros(shape=(h.shape[0], self.dim))
        eigvecs = np.zeros(shape=(h.shape[0], self.dim, self.dim), dtype=complex)
        assert np.allclose(np.conjugate(np.transpose(h, axes=[0, 2, 1])), h)
        for i in range(h.shape[0]):
            vals, vecs = np.linalg.eigh(h[i])
            eigvals[i, :] = vals
            eigvecs[i, :, :] = vecs
        return eigvals, eigvecs

    def onsite_density_density(self, Ck: np.ndarray) -> np.ndarray:
        # rho_{sigma,sigma'} = (1/N^2) sum_k C_k[sigma,sigma']
        rho = np.mean(Ck, axis=0)
        # Exact on-site U0 Hartree-Fock single-particle shift from Eq. (4.2):
        #   h_HF = U0 * Tr(rho) * I - U0 * rho^T
        # using the note convention C_{σσ'} = <c^†_σ c_{σ'}>.
        nbar = float(np.trace(rho).real)
        h_hf = self.U0 * nbar * np.eye(self.dim, dtype=np.complex128) - self.U0 * rho.T
        h_hf = 0.5 * (h_hf + h_hf.conj().T)
        return h_hf, np.sum(h_hf * rho) / 2 / self.nu

    def nearest_neighbor_density_density(self, Ck: np.ndarray) -> np.ndarray:
        # Section 4.3: U_N/2 sum_{R,R'} n_R n_R' delta^{(N)}_{R-R'} on the
        # triangular-lattice nearest-neighbor shell
        # {(±1,0), (0,±1), (±1,±1)}.
        if np.isclose(self.Un, 0.0):
            h_hf = np.zeros((self.N**2, self.dim, self.dim), dtype=np.complex128)
            return h_hf, 0.0

        rho = np.mean(Ck, axis=0)
        nbar = float(np.trace(rho).real)

        nn_displacements = np.array([
            self.A[1],
            -self.A[1],
            self.A[2],
            -self.A[2],
            self.A[1] + self.A[2],
            -(self.A[1] + self.A[2]),
        ])
        k_vectors = np.zeros((self.N**2, 2), dtype=float)
        for idx, (n1, n2) in self.indexToKGrid.items():
            k_vectors[idx] = n1 * self.G[1] / self.N + n2 * self.G[2] / self.N

        phases = np.exp(-1j * np.einsum('kd,ad->ka', k_vectors, nn_displacements))
        rho_delta = np.einsum('ka,kij->aij', phases, Ck) / self.N**2

        hartree = self.Un * len(nn_displacements) * nbar * np.eye(self.dim, dtype=np.complex128)
        fock = -self.Un * np.einsum('ka,aij->kij', phases.conj(), rho_delta.swapaxes(-1, -2))
        h_hf = hartree[None, :, :] + fock
        h_hf = 0.5 * (h_hf + h_hf.swapaxes(-1, -2).conj())

        e_hf = np.sum(h_hf * Ck) / (2 * self.N**2)
        e_hf = assert_real(e_hf)
        return h_hf, e_hf / self.nu

    def on_site_hubbard_up_down(self, Ck: np.ndarray) -> np.ndarray:
        # Section 4.2: V_{up,down}/2 sum_{R,alpha,alpha'} n_{R alpha up} n_{R alpha' down}
        # with the local basis ordered as [alpha1 up, alpha1 down, alpha2 up, alpha2 down].
        rho = np.mean(Ck, axis=0)
        up_idx = np.arange(0, self.dim, 2)
        down_idx = np.arange(1, self.dim, 2)

        n_up = float(np.trace(rho[np.ix_(up_idx, up_idx)]).real)
        n_down = float(np.trace(rho[np.ix_(down_idx, down_idx)]).real)

        h_hf = np.zeros((self.dim, self.dim), dtype=np.complex128)
        h_hf[np.ix_(up_idx, up_idx)] += 0.5 * self.Vupdown * n_down * np.eye(len(up_idx))
        h_hf[np.ix_(down_idx, down_idx)] += 0.5 * self.Vupdown * n_up * np.eye(len(down_idx))

        # Fock exchange only mixes opposite-spin sectors for this interaction.
        h_hf[np.ix_(up_idx, down_idx)] -= 0.5 * self.Vupdown * rho[np.ix_(down_idx, up_idx)].T
        h_hf[np.ix_(down_idx, up_idx)] -= 0.5 * self.Vupdown * rho[np.ix_(up_idx, down_idx)].T

        h_hf = 0.5 * (h_hf + h_hf.conj().T)
        return h_hf, np.sum(h_hf * rho) / 2 / self.nu

    def hartree_fock_terms(self, Ck: np.ndarray) -> tuple[np.ndarray, float]:
        h_u0, e_u0 = self.onsite_density_density(Ck)
        h_un, e_un = self.nearest_neighbor_density_density(Ck)
        h_vud, e_vud = self.on_site_hubbard_up_down(Ck)
        h_hf = h_un + h_u0[None, :, :] + h_vud[None, :, :]
        e_hf = assert_real(e_u0 + e_un + e_vud)
        return h_hf, e_hf

    def reference_Ck(self) -> np.ndarray:
        """
        Section 6 reference density C^(0): diagonalize h_tb(k) and fill it using the
        same occupancy rule as the self-consistent iteration.
        """
        eigvals, eigvecs = self.diagonalize_blocks(self.Htb)
        occ, _ = self.occupancies_from_energies(eigvals)
        return self.density_from_eigensystem(eigvecs, occ)

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
        subtract_reference: bool = False,
        verbose: bool = False,
    ) -> HFSolution:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("mix must satisfy 0 < mix <= 1.")

        if Ck0 is None:
            Ck = self.build_Ck(
                random_seed=random_seed,
                use_reference_filling=subtract_reference,
            )
        else:
            Ck = np.array(Ck0, dtype=np.complex128, copy=True)
            if Ck.shape != (self.N**2, self.dim, self.dim):
                raise ValueError(f"Ck0 must have shape {(self.N**2, self.dim, self.dim)}.")

        if subtract_reference:
            reference_Ck = self.reference_Ck()
            h_ref, e_ref = self.hartree_fock_terms(reference_Ck)
        else:
            h_ref = np.zeros((self.N**2, self.dim, self.dim), dtype=np.complex128)
            e_ref = 0.0

        converged = False
        for it_ in range(1, max_iter + 1):
            h_hf, e_hf = self.hartree_fock_terms(Ck)
            h_k = self.Htb - h_ref + h_hf
            evals, vecs = self.diagonalize_blocks(h_k)
            evals = evals - e_hf + e_ref
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

        return h_k, e_hf - e_ref, Ck, converged, it_

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
    Un_ = 0.2
    Vupdown_ = 0.3
    metal_ = True
    nu_ = 1
    C0_modify_ = True

    model = HF(path='TightBindingModel/Re2CoO8/withSOCwannier-dim2', 
               nu=nu_, U0=U0_, Un=Un_, Vupdown=Vupdown_, N=12, metal=metal_)
    reference_Ck = model.reference_Ck()

    now_int = int(np.round(datetime.datetime.now().timestamp() * 1e6))
    h_k, e_hf, Ck, converged, it_ = model.solve(max_iter=5000, alpha=0.5, verbose=True, random_seed=now_int, subtract_reference=C0_modify_)
    print(f'convergence: {converged} / iteration: {it_}')

    effective_hopping = model.build_effective_hopping(h_k)
    for idx, grid in model.indexToKGrid.items():
        k = grid[0] * model.G[1] / model.N + grid[1] * model.G[2] / model.N
        eigvals1 = np.linalg.eigh(h_k[idx, :, :])[0]
        eigvals2 = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        assert np.allclose(eigvals1, eigvals2)
    print(np.round(np.trace(np.sum(Ck, axis=0)), 2))

    N_high_symmetry = 100
    band_structure = np.zeros((3*N_high_symmetry+1, model.dim))
    for i in range(N_high_symmetry):
        k = interpolation(model.Gamma, model.K, N_high_symmetry, i)
        eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        band_structure[i, :] = eigvals - e_hf

        k = interpolation(model.K, model.M, N_high_symmetry, i)
        eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        band_structure[N_high_symmetry+i, :] = eigvals - e_hf

        k = interpolation(model.M, model.Gamma, N_high_symmetry, i)
        eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        band_structure[2*N_high_symmetry+i, :] = eigvals - e_hf
    k = model.Gamma
    eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
    band_structure[3*N_high_symmetry, :] = eigvals - e_hf
    for bnd in range(model.dim):
        plt.plot(np.arange(3*N_high_symmetry+1), band_structure[:, bnd], color='k')
    plt.show()

