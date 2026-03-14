import argparse
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from HFnew import HF
import datetime
from HartreeFockTrigonalCDW import TrigonalCDWDFT2


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


def similarity_transformation(transformed, transformation):
    return dagger(transformation) @ transformed @ transformation


class HFsuper:
    """
    Non-interacting tight-binding supercell model (Section 8).
    Class layout follows HFnew.py style:
    __init__ -> read_path -> define_convention -> build_hopping -> build_Htb/build_Htb_super
    """

    def __init__(self, path, nu, N, U0=0.0, Un=0.0, Vupdown=0.0, metal=True, kT=0.005):
        self.path = path
        self.nu = nu
        self.N = N
        self.U0 = U0
        self.Un = Un
        self.Vupdown = Vupdown
        self.metal = metal
        self.kT = kT

        self.dim = 4
        self.numSub = 3
        self.dimSuper = self.numSub * self.dim
        self.nuSuper = self.numSub * self.nu
        self.Nocc = self.nuSuper * self.N**2

        self.read_path()
        self.define_convention()
        self.build_hopping()
        self.build_Htb()
        self.build_Htb_super()

    @staticmethod
    def load_csv_hermitian(path, delimiter, dtype):
        m = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return (m + dagger(m)) / 2

    def read_path(self):
        self.C3z = np.loadtxt(f"{self.path}/C3z.csv", delimiter=",", dtype=complex)
        self.Inverse = np.loadtxt(f"{self.path}/Inverse.csv", delimiter=",", dtype=complex)
        self.H0 = self.load_csv_hermitian(f"{self.path}/H0.csv", delimiter=",", dtype=complex)
        self.H1 = self.load_csv_hermitian(f"{self.path}/H1.csv", delimiter=",", dtype=complex)
        self.H2 = self.load_csv_hermitian(f"{self.path}/H2.csv", delimiter=",", dtype=complex)
        self.H3 = self.load_csv_hermitian(f"{self.path}/H3.csv", delimiter=",", dtype=complex)
        assert np.all(np.array(self.C3z.shape) == self.dim)
        assert np.all(np.array(self.Inverse.shape) == self.dim)
        assert np.all(np.array(self.H0.shape) == self.dim)
        assert np.all(np.array(self.H1.shape) == self.dim)
        assert np.all(np.array(self.H2.shape) == self.dim)
        assert np.all(np.array(self.H3.shape) == self.dim)

    def define_convention(self):
        pi = np.pi
        self.A = {}
        self.G = {}
        self.As = {}
        self.Gs = {}

        self.A[1] = np.array([1.0, 0.0])
        self.A[2] = np.array([np.cos(2 * pi / 3), np.sin(2 * pi / 3)])

        prefac = 4 * pi / (np.sqrt(3))
        self.G[1] = prefac * np.array([np.cos(pi / 6), np.sin(pi / 6)])
        self.G[2] = prefac * np.array([0.0, 1.0])

        for i1, i2 in product(*([[1, 2]] * 2)):
            assert np.isclose(IP(self.A[i1], self.G[i2]), 2 * pi * int(i1 == i2))

        # Section 8: sqrt(3) x sqrt(3) supercell
        self.As[1] = 2 * self.A[1] + self.A[2]
        self.As[2] = self.A[1] + 2 * self.A[2]
        self.Gs[1] = (2 / 3) * self.G[1] - (1 / 3) * self.G[2]
        self.Gs[2] = (-1 / 3) * self.G[1] + (2 / 3) * self.G[2]

        self.q = {}
        self.q[0] = self.As[2] / 3
        self.q[1] = - self.As[1] / 3
        self.q[2] = (self.As[1] - self.As[2]) / 3

        self.GammaSuper = 0 * self.Gs[1]
        self.Ksuper = (1/3) * self.Gs[1] + (1/3) * self.Gs[2]
        self.Msuper = (1/2) * self.Gs[2]

        for i1, i2 in product(*([[1, 2]] * 2)):
            assert np.isclose(IP(self.As[i1], self.Gs[i2]), 2 * pi * int(i1 == i2))

        self.kGridToIndex = {}
        self.kGridToIndexS = {}
        for n1, n2 in product(*([range(self.N)] * 2)):
            self.kGridToIndex[n1, n2] = n1 * self.N + n2
            self.kGridToIndexS[n1, n2] = n1 * self.N + n2
        self.indexToKGrid = {v: k for k, v in self.kGridToIndex.items()}
        self.indexToKGridS = {v: k for k, v in self.kGridToIndexS.items()}

        # q_j - q_j' in (A1, A2) integer coordinates for j,j' = 1,2,3.
        self.dq = {
            (1, 1): (0, 0),
            (1, 2): (1, 1),
            (1, 3): (0, 1),
            (2, 1): (-1, -1),
            (2, 2): (0, 0),
            (2, 3): (-1, 0),
            (3, 1): (0, -1),
            (3, 2): (1, 0),
            (3, 3): (0, 0),
        }

    @staticmethod
    def rotate3(j1, j2):
        return (-j2, j1 - j2)

    @staticmethod
    def inversion(j1, j2):
        return (-j1, -j2)

    def build_hopping(self):
        self.hopping = {}
        self.hopping[0, 0] = self.H0
        self.hopping[1, 0] = self.H1
        self.hopping[1, 2] = self.H2
        self.hopping[2, 0] = self.H3

        initial_hopping_keys = list(self.hopping.keys())
        for k in initial_hopping_keys:
            if k != (0, 0):
                new_k = self.rotate3(*k)
                self.hopping[new_k] = similarity_transformation(self.hopping[k], self.C3z)
                new_k = self.rotate3(*self.rotate3(*k))
                self.hopping[new_k] = similarity_transformation(self.hopping[k], dagger(self.C3z))

        initial_hopping_keys = list(self.hopping.keys())
        for k in initial_hopping_keys:
            if k != (0, 0):
                new_k = self.inversion(*k)
                self.hopping[new_k] = similarity_transformation(self.hopping[k], self.Inverse)

    def HKtb(self, k):
        htb = np.zeros(shape=(self.dim, self.dim), dtype=complex)
        for key, val in self.hopping.items():
            DeltaR = key[0] * self.A[1] + key[1] * self.A[2]
            htb += np.exp(-1j * IP(k, DeltaR)) * val
        htb = (htb + dagger(htb)) / 2
        assert np.allclose(htb, dagger(htb))
        return htb

    @staticmethod
    def supercell_indices_from_original(m, n):
        # [m, n]^T = [[2,1],[1,2]] [u, v]^T
        u_num = 2 * m - n
        v_num = -m + 2 * n
        if (u_num % 3) != 0 or (v_num % 3) != 0:
            return None
        return (u_num // 3, v_num // 3)

    def HKtb_super(self, k):
        # Basis ordering: (j=1,2,3) x (internal 4 states).
        htb = np.zeros(shape=(self.dimSuper, self.dimSuper), dtype=complex)
        for j, jp in product(range(1, self.numSub + 1), repeat=2):
            dq1, dq2 = self.dq[j, jp]
            block = np.zeros(shape=(self.dim, self.dim), dtype=complex)
            for key, val in self.hopping.items():
                m, n = key
                c1 = m - dq1
                c2 = n - dq2
                if self.supercell_indices_from_original(c1, c2) is None:
                    continue
                DeltaR = m * self.A[1] + n * self.A[2]
                block += np.exp(-1j * IP(k, DeltaR)) * val
            i0 = (j - 1) * self.dim
            j0 = (jp - 1) * self.dim
            htb[i0 : i0 + self.dim, j0 : j0 + self.dim] = block
        htb = (htb + dagger(htb)) / 2
        assert np.allclose(htb, dagger(htb))
        return htb

    def build_Htb(self):
        self.Htb = np.zeros(shape=(self.N**2, self.dim, self.dim), dtype=complex)
        for idx, grid in self.indexToKGrid.items():
            n1, n2 = grid
            k = n1 * self.G[1] / self.N + n2 * self.G[2] / self.N
            self.Htb[idx, :, :] = self.HKtb(k)

    def build_Htb_super(self):
        self.HtbSuper = np.zeros(shape=(self.N**2, self.dimSuper, self.dimSuper), dtype=complex)
        for idx, grid in self.indexToKGridS.items():
            n1, n2 = grid
            k = n1 * self.Gs[1] / self.N + n2 * self.Gs[2] / self.N
            self.HtbSuper[idx, :, :] = self.HKtb_super(k)

    def diagonalize_super(self):
        eigvals = np.zeros((self.N**2, self.dimSuper))
        eigvecs = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=complex)
        for i in range(self.N**2):
            vals, vecs = np.linalg.eigh(self.HtbSuper[i])
            eigvals[i] = vals
            eigvecs[i] = vecs
        return eigvals, eigvecs

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
        Ck = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=np.complex128)

        if use_reference_filling:
            reference_diag = np.diag(np.sum(self.reference_Ck(), axis=0)).real / self.Nocc
        else:
            reference_diag = np.full(self.dimSuper, self.nuSuper / self.dimSuper, dtype=float)

        for idx in range(self.N**2):
            filling_deviation = deviation * rng.random(self.dimSuper)
            filling_deviation = filling_deviation - np.mean(filling_deviation)
            filling = reference_diag + filling_deviation
            if not np.all(filling > 0):
                raise ValueError("Negative filling number in initial Ck.")

            Ck[idx, np.arange(self.dimSuper), np.arange(self.dimSuper)] = filling

            off_diagonal = (
                rng.random((self.dimSuper, self.dimSuper)) + 1j * rng.random((self.dimSuper, self.dimSuper))
            ) * deviation
            np.fill_diagonal(off_diagonal, 0)
            off_diagonal = 0.5 * (off_diagonal + off_diagonal.conj().T)
            Ck[idx] += off_diagonal

            Ck[idx] = 0.5 * (Ck[idx] + Ck[idx].conj().T)
            if not np.all(np.linalg.eigvalsh(Ck[idx]) > 0):
                raise ValueError("Initial Ck must be positive definite.")
            if not np.all(np.linalg.eigvalsh(np.eye(self.dimSuper) - Ck[idx]) > 0):
                raise ValueError("Initial Ck must satisfy Ck < I.")

        total_trace = np.trace(Ck, axis1=1, axis2=2).sum().real
        if not np.isclose(total_trace, self.N**2 * self.nuSuper):
            raise ValueError("Initial Ck violates the particle-number constraint.")
        return Ck

    def onsite_density_density(self, Ck):
        # Supercell generalization of HFnew.onsite_density_density:
        # rho = (1/N^2) sum_k C_k over the 12x12 supercell basis.
        # On-site U0 acts locally on each sublattice block (4x4 each).
        rho = np.mean(Ck, axis=0)
        h_hf = np.zeros((self.dimSuper, self.dimSuper), dtype=np.complex128)

        for j in range(self.numSub):
            sl = slice(j * self.dim, (j + 1) * self.dim)
            rho_j = rho[sl, sl]
            nbar_j = float(np.trace(rho_j).real)
            h_j = self.U0 * nbar_j * np.eye(self.dim, dtype=np.complex128) - self.U0 * rho_j.T
            h_hf[sl, sl] = h_j

        assert np.allclose(h_hf, dagger(h_hf))
        e_hf = np.sum(h_hf * rho) / 2 / self.nuSuper
        return h_hf, assert_real(e_hf)

    def nearest_neighbor_density_density0(self, Ck: np.ndarray) -> np.ndarray:
        # Supercell analogue of HFnew.nearest_neighbor_density_density.
        # Keep the same six nearest-neighbor displacements on the original triangular lattice.
        if np.isclose(self.Un, 0.0):
            h_hf = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=np.complex128)
            return h_hf, 0.0

        nn_displacements_increase = [
                self.A[1],
                self.A[2],
                -(self.A[1] + self.A[2]),
            ]
        nn_displacements_decrease = [
                - self.A[1],
                - self.A[2],
                self.A[1] + self.A[2],
            ]

        h_hf_hartree = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=complex)
        for kIdx1, kIdx2 in product(*([list(self.indexToKGrid.keys())]*2)):
            for j2 in range(3):
                for displacement in nn_displacements_increase:
                    k1 = self.indexToKGrid[kIdx1][0] * self.Gs[1] / self.N + self.indexToKGrid[kIdx1][1] * self.Gs[2] / self.N
                    k2 = self.indexToKGrid[kIdx2][0] * self.Gs[1] / self.N + self.indexToKGrid[kIdx2][1] * self.Gs[2] / self.N
                    j1 = (j2 + 1) % 3
                    h_hf_hartree[kIdx2,  self.dim*j2:self.dim*(j2+1), self.dim*j2:self.dim*(j2+1)] += np.diag([np.trace(Ck[kIdx1, self.dim*j1:self.dim*(j1+1), self.dim*j1:self.dim*(j1+1)])] * self.dim) / self.N**2
                for displacement in nn_displacements_decrease:
                    k1 = self.indexToKGrid[kIdx1][0] * self.Gs[1] / self.N + self.indexToKGrid[kIdx1][1] * self.Gs[2] / self.N
                    k2 = self.indexToKGrid[kIdx2][0] * self.Gs[1] / self.N + self.indexToKGrid[kIdx2][1] * self.Gs[2] / self.N
                    j1 = (j2 - 1) % 3
                    h_hf_hartree[kIdx2,  self.dim*j2:self.dim*(j2+1), self.dim*j2:self.dim*(j2+1)] += np.diag([np.trace(Ck[kIdx1, self.dim*j1:self.dim*(j1+1), self.dim*j1:self.dim*(j1+1)])] * self.dim) / self.N**2

        h_hf_fock = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=complex)
        for kIdx1, kIdx2 in product(*([list(self.indexToKGrid.keys())]*2)):
            for j2 in range(3):
                for displacement in nn_displacements_increase:
                    k1 = self.indexToKGrid[kIdx1][0] * self.Gs[1] / self.N + self.indexToKGrid[kIdx1][1] * self.Gs[2] / self.N
                    k2 = self.indexToKGrid[kIdx2][0] * self.Gs[1] / self.N + self.indexToKGrid[kIdx2][1] * self.Gs[2] / self.N
                    phase = np.exp(-1j * IP(k1 - k2, displacement))
                    j1 = (j2 + 1) % 3
                    h_hf_fock[kIdx2,  self.dim*j2:self.dim*(j2+1), self.dim*j1:self.dim*(j1+1)] += phase * Ck[kIdx1, self.dim*j1:self.dim*(j1+1), self.dim*j2:self.dim*(j2+1)].T / self.N**2
                for displacement in nn_displacements_decrease:
                    k1 = self.indexToKGrid[kIdx1][0] * self.Gs[1] / self.N + self.indexToKGrid[kIdx1][1] * self.Gs[2] / self.N
                    k2 = self.indexToKGrid[kIdx2][0] * self.Gs[1] / self.N + self.indexToKGrid[kIdx2][1] * self.Gs[2] / self.N
                    phase = np.exp(-1j * IP(k1 - k2, displacement))
                    j1 = (j2 - 1) % 3
                    h_hf_fock[kIdx2,  self.dim*j2:self.dim*(j2+1), self.dim*j1:self.dim*(j1+1)] += phase * Ck[kIdx1, self.dim*j1:self.dim*(j1+1), self.dim*j2:self.dim*(j2+1)].T / self.N**2
                    
        h_hf = self.Un * (h_hf_hartree - h_hf_fock)

        assert np.allclose(h_hf, h_hf.swapaxes(-1, -2).conj())

        e_hf = np.sum(h_hf * Ck) / (2 * self.N**2)
        e_hf = assert_real(e_hf)
        return h_hf, e_hf / self.nuSuper

    def nearest_neighbor_density_density(self, Ck: np.ndarray) -> np.ndarray:
        # Vectorized equivalent of nearest_neighbor_density_density0.
        if np.isclose(self.Un, 0.0):
            h_hf = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=np.complex128)
            return h_hf, 0.0

        k_vectors = np.zeros((self.N**2, 2), dtype=float)
        for idx, (n1, n2) in self.indexToKGrid.items():
            k_vectors[idx] = n1 * self.Gs[1] / self.N + n2 * self.Gs[2] / self.N

        nn_displacements_increase = np.array(
            [
                self.A[1],
                self.A[2],
                -(self.A[1] + self.A[2]),
            ]
        )
        nn_displacements_decrease = np.array(
            [
                -self.A[1],
                -self.A[2],
                self.A[1] + self.A[2],
            ]
        )

        # phase_sum[k2, k1] = sum_d exp(-i (k1-k2)·d)
        def build_phase_sum(displacements: np.ndarray) -> np.ndarray:
            phase_sum = np.zeros((self.N**2, self.N**2), dtype=np.complex128)
            for d in displacements:
                u = np.exp(-1j * (k_vectors @ d))
                phase_sum += np.outer(np.conjugate(u), u)
            return phase_sum

        phase_increase = build_phase_sum(nn_displacements_increase)
        phase_decrease = build_phase_sum(nn_displacements_decrease)

        h_hf_hartree = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=np.complex128)
        h_hf_fock = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=np.complex128)

        # Precompute per-sublattice traces: tr_block[j, k1] = Tr Ck[k1] on block j.
        tr_block = np.zeros((self.numSub, self.N**2), dtype=np.complex128)
        for j in range(self.numSub):
            sl = slice(self.dim * j, self.dim * (j + 1))
            tr_block[j] = np.trace(Ck[:, sl, sl], axis1=1, axis2=2)

        # Hartree: independent of k2, replicated to all k2 blocks.
        for j2 in range(self.numSub):
            jp = (j2 + 1) % self.numSub
            jm = (j2 - 1) % self.numSub
            scalar = (3.0 / self.N**2) * (np.sum(tr_block[jp]) + np.sum(tr_block[jm]))
            sl = slice(self.dim * j2, self.dim * (j2 + 1))
            h_hf_hartree[:, sl, sl] = np.eye(self.dim, dtype=np.complex128)[None, :, :] * scalar

        # Fock: block-coupled convolution in k-space.
        for j2 in range(self.numSub):
            j_inc = (j2 + 1) % self.numSub
            j_dec = (j2 - 1) % self.numSub

            sl2 = slice(self.dim * j2, self.dim * (j2 + 1))
            sl_inc = slice(self.dim * j_inc, self.dim * (j_inc + 1))
            sl_dec = slice(self.dim * j_dec, self.dim * (j_dec + 1))

            block_inc = Ck[:, sl_inc, sl2].transpose(0, 2, 1)  # Ck[k1, j_inc, j2]^T
            block_dec = Ck[:, sl_dec, sl2].transpose(0, 2, 1)  # Ck[k1, j_dec, j2]^T

            h_hf_fock[:, sl2, sl_inc] += np.einsum("ab,bij->aij", phase_increase, block_inc) / self.N**2
            h_hf_fock[:, sl2, sl_dec] += np.einsum("ab,bij->aij", phase_decrease, block_dec) / self.N**2

        h_hf = self.Un * (h_hf_hartree - h_hf_fock)

        assert np.allclose(h_hf, h_hf.swapaxes(-1, -2).conj())

        e_hf = np.sum(h_hf * Ck) / (2 * self.N**2)
        e_hf = assert_real(e_hf)
        return h_hf, e_hf / self.nuSuper

    def on_site_hubbard_up_down(self, Ck):
        # Supercell analogue of HFnew.on_site_hubbard_up_down:
        # apply the local spin-up/spin-down interaction within each sublattice block.
        rho = np.mean(Ck, axis=0)
        h_hf = np.zeros((self.dimSuper, self.dimSuper), dtype=np.complex128)
        g = self.Vupdown
        if np.isclose(g, 0.0):
            return h_hf, 0.0

        up_local = np.array([0, 2], dtype=int)
        down_local = np.array([1, 3], dtype=int)

        for j in range(self.numSub):
            sl = slice(j * self.dim, (j + 1) * self.dim)
            rho_j = rho[sl, sl]

            n_up = float(np.trace(rho_j[np.ix_(up_local, up_local)]).real)
            n_down = float(np.trace(rho_j[np.ix_(down_local, down_local)]).real)

            h_j = np.zeros((self.dim, self.dim), dtype=np.complex128)
            h_j[np.ix_(up_local, up_local)] += 0.5 * g * n_down * np.eye(len(up_local))
            h_j[np.ix_(down_local, down_local)] += 0.5 * g * n_up * np.eye(len(down_local))

            # Fock exchange only mixes opposite-spin sectors.
            h_j[np.ix_(up_local, down_local)] -= 0.5 * g * rho_j[np.ix_(down_local, up_local)].T
            h_j[np.ix_(down_local, up_local)] -= 0.5 * g * rho_j[np.ix_(up_local, down_local)].T
            h_hf[sl, sl] = h_j

        assert np.allclose(h_hf, dagger(h_hf))
        e_hf = np.sum(h_hf * rho) / 2 / self.nuSuper
        return h_hf, assert_real(e_hf)

    def hartree_fock_terms(self, Ck: np.ndarray) -> tuple[np.ndarray, float]:
        h_u0, e_u0 = self.onsite_density_density(Ck)
        h_un, e_un = self.nearest_neighbor_density_density(Ck)
        h_vud, e_vud = self.on_site_hubbard_up_down(Ck)
        h_hf = h_u0[None, :, :] + h_vud[None, :, :] + h_un
        e_hf = assert_real(e_u0 + e_vud + e_un)
        return h_hf, e_hf

    def diagonalize_blocks(self, h):
        eigvals = np.zeros(shape=(h.shape[0], self.dimSuper))
        eigvecs = np.zeros(shape=(h.shape[0], self.dimSuper, self.dimSuper), dtype=complex)
        assert np.allclose(np.conjugate(np.transpose(h, axes=[0, 2, 1])), h)
        for i in range(h.shape[0]):
            vals, vecs = np.linalg.eigh(h[i])
            eigvals[i, :] = vals
            eigvecs[i, :, :] = vecs
        return eigvals, eigvecs

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

    def density_from_eigensystem(self, eigvecs, occ):
        Ck = np.zeros(shape=eigvecs.shape, dtype=complex)
        for i in range(eigvecs.shape[0]):
            v = eigvecs[i, :, :]
            Ck[i, :, :] = v.conj() @ np.diag(occ[i]) @ v.T
            Ck[i] = 0.5 * (Ck[i] + Ck[i].conj().T)
        return Ck

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
        verbose: bool = False):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("mix must satisfy 0 < mix <= 1.")

        if Ck0 is None:
            Ck = self.build_Ck(
                random_seed=random_seed,
                use_reference_filling=subtract_reference,
            )
        else:
            Ck = np.array(Ck0, dtype=np.complex128, copy=True)
            if Ck.shape != (self.N**2, self.dimSuper, self.dimSuper):
                raise ValueError(f"Ck0 must have shape {(self.N**2, self.dimSuper, self.dimSuper)}.")

        if subtract_reference:
            reference_Ck = self.reference_Ck()
            h_ref, e_ref = self.hartree_fock_terms(reference_Ck)
        else:
            h_ref = np.zeros((self.N**2, self.dimSuper, self.dimSuper), dtype=np.complex128)
            e_ref = 0.0

        converged = False
        for it_ in range(1, max_iter + 1):
            h_hf, e_hf = self.hartree_fock_terms(Ck)
            h_k = self.HtbSuper - h_ref + h_hf
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
            for qIdx1, qIdx2 in product(*([range(3)]*2)):
                DeltaR = self.As[1] * j1 + self.As[2] * j2 + self.q[qIdx1] - self.q[qIdx2]
                tilde_gamma = np.zeros((self.dim, self.dim), dtype=complex)
                for idx, grid in self.indexToKGrid.items():
                    n1, n2 = grid[0], grid[1]
                    k = n1 * self.Gs[1] / self.N + n2 * self.Gs[2] / self.N
                    exponent = np.exp(1j * IP(k, DeltaR))
                    tilde_gamma += exponent * h_k[idx, qIdx1*self.dim:(qIdx1+1)*self.dim, qIdx2*self.dim:(qIdx2+1)*self.dim]
                effective_hopping[j1, j2, qIdx1, qIdx2] = tilde_gamma / self.N**2
        return effective_hopping
    
    def HKtbEff(self, k, effective_hopping):
        htb = np.zeros(shape=(self.dimSuper, self.dimSuper), dtype=complex)
        for key, val in effective_hopping.items():
            qIdx1, qIdx2 = key[2], key[3]
            DeltaR = key[0] * self.As[1] + key[1] * self.As[2] + self.q[qIdx1] - self.q[qIdx2]
            htb[qIdx1*self.dim:(qIdx1+1)*self.dim, qIdx2*self.dim:(qIdx2+1)*self.dim] += np.exp(-1j * IP(k, DeltaR)) * val
        assert np.allclose(htb, dagger(htb))
        return htb


if __name__ == "__main__":
    U0_ = 0.3
    Un_ = 0.15
    Vupdown_ = 0.3

    model = HFsuper(path='TightBindingModel/Re2CoO8/withSOCwannier-dim2', nu=1, N=12, U0=U0_, Un=Un_, Vupdown=Vupdown_)
    now_int = int(np.round(datetime.datetime.now().timestamp() * 1e6))
    h_k, e_hf, Ck, converged, it_ = model.solve(max_iter=10000, alpha=0.5, verbose=True, random_seed=now_int, subtract_reference=False)
    print(f'convergence: {converged} / iteration: {it_}')
    effective_hopping = model.build_effective_hopping(h_k)

    for idx, grid in model.indexToKGrid.items():
        k = grid[0] * model.Gs[1] / model.N + grid[1] * model.Gs[2] / model.N
        eigvals1 = np.linalg.eigh(h_k[idx, :, :])[0]
        eigvals2 = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        assert np.allclose(eigvals1, eigvals2)
    print(np.round(np.trace(np.sum(Ck, axis=0)), 2))

    N_high_symmetry = 100
    band_structure = np.zeros((3*N_high_symmetry+1, model.dimSuper))
    for i in range(N_high_symmetry):
        k = interpolation(model.GammaSuper, model.Ksuper, N_high_symmetry, i)
        eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        band_structure[i, :] = eigvals - e_hf

        k = interpolation(model.Ksuper, model.Msuper, N_high_symmetry, i)
        eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        band_structure[N_high_symmetry+i, :] = eigvals - e_hf

        k = interpolation(model.Msuper, model.GammaSuper, N_high_symmetry, i)
        eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        band_structure[2*N_high_symmetry+i, :] = eigvals - e_hf
    k = model.GammaSuper
    eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
    band_structure[3*N_high_symmetry, :] = eigvals - e_hf
    for bnd in range(model.dimSuper):
        plt.plot(np.arange(3*N_high_symmetry+1), band_structure[:, bnd], color='k')
    plt.show()


