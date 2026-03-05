import argparse
from itertools import product

import numpy as np

from HFnew import HF


def IP(arr1, arr2):
    return np.sum(arr1 * arr2)


def dagger(arr):
    return np.transpose(np.conjugate(arr))


def similarity_transformation(transformed, transformation):
    return dagger(transformation) @ transformed @ transformation


class HFsuper:
    """
    Non-interacting tight-binding supercell model (Section 8).
    Class layout follows HFnew.py style:
    __init__ -> read_path -> define_convention -> build_hopping -> build_Htb/build_Htb_super
    """

    def __init__(self, path, N, a0=1.0):
        self.path = path
        self.N = N
        self.a0 = 1

        self.dim = 4
        self.numSub = 3
        self.dimSuper = self.numSub * self.dim

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

        self.A[1] = np.array([1.0, 0.0]) * self.a0
        self.A[2] = np.array([np.cos(2 * pi / 3), np.sin(2 * pi / 3)]) * self.a0

        prefac = 4 * pi / (np.sqrt(3) * self.a0)
        self.G[1] = prefac * np.array([np.cos(pi / 6), np.sin(pi / 6)])
        self.G[2] = prefac * np.array([0.0, 1.0])

        for i1, i2 in product(*([[1, 2]] * 2)):
            assert np.isclose(IP(self.A[i1], self.G[i2]), 2 * pi * int(i1 == i2))

        # Section 8: sqrt(3) x sqrt(3) supercell
        self.As[1] = 2 * self.A[1] + self.A[2]
        self.As[2] = self.A[1] + 2 * self.A[2]
        self.Gs[1] = (2 / 3) * self.G[1] - (1 / 3) * self.G[2]
        self.Gs[2] = (-1 / 3) * self.G[1] + (2 / 3) * self.G[2]

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


if __name__ == "__main__":
    

    modelSuper = HFsuper(path='TightBindingModel/Re2CoO8/withSOCwannier-dim2', N=12)
    model = HF(path='TightBindingModel/Re2CoO8/withSOCwannier-dim2', N=12, nu=1)
    Delta_k = np.random.rand() * model.G[1] + np.random.rand() * model.G[2]

    arr1 = (np.linalg.eigh(modelSuper.HKtb_super(modelSuper.GammaSuper + Delta_k))[0])

    arr2a = (np.linalg.eigh(model.HKtb(model.Gamma + Delta_k))[0])
    arr2b = (np.linalg.eigh(model.HKtb(model.K + Delta_k))[0])
    arr2c = (np.linalg.eigh(model.HKtb(-model.K + Delta_k))[0])
    arr2 = np.concat([arr2a, arr2b, arr2c])

    assert np.allclose(np.sort(arr1), np.sort(arr2))


