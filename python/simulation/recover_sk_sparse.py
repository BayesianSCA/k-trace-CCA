import numpy as np
from .kyber.reference.params import KYBER_Q
from .kyber.reference.polyvec import polyvec, polyvec_invntt, polyvec_reduce
from .kyber.reference.poly import poly
from .kyber.reference.ntt import zetas
from .kyber.reference.reduce import MONT


def invmod(a, q=KYBER_Q):
    # assuming q to be prime
    return pow(int(a), q - 2, q)
    # return pow(a, -1, q)  # requires Python 3.8


def solve_2x2(b, c, zeta, q=KYBER_Q):
    invdet = invmod((int(b[0]) ** 2 - int(zeta) * (int(b[1]) ** 2)) % q)
    s0 = ((int(b[0]) * int(c[0]) - int(zeta) * int(b[1]) * int(c[1])) * invdet) % q
    s1 = ((int(b[0]) * int(c[1]) - int(b[1]) * int(c[0])) * invdet) % q
    return [s0, s1]


class RecoverSk:
    def __init__(self, height, vec_k):
        self.sk_hat = polyvec(height, vec_k)
        self.sk_set = polyvec(height, vec_k)
        self.vec_k = self.sk_hat.vec_k

    @property
    def sk(self):
        result = polyvec_invntt(self.sk_hat.copy())
        return polyvec_reduce(result)

    @property
    def sk_hat_mont(self):
        result = np.int32(self.sk_hat) * MONT % KYBER_Q
        return polyvec_reduce(result)

    @property
    def is_complete(self):
        return (self.sk_set == True).all()

    def recover_sk_hat(self, b_hat, c_hat_poly):
        zeta_offset = b_hat.height // 4  # 64 - see poly_basemul_montgomery
        invMONT = invmod(MONT)
        for i_k, b_hat_i in enumerate(b_hat.vec):
            if (b_hat_i.coeffs == 0).all():
                continue
            else:
                for j in range(0, b_hat_i.height, 2):
                    if b_hat_i.coeffs[j] == 0 and b_hat_i.coeffs[j + 1] == 0:
                        continue
                    else:
                        if (self.sk_set.vec[i_k].coeffs[j : j + 2] == 1).any():
                            raise ValueError("Theses coeffs were recovered already")
                        zeta = (
                            zetas[zeta_offset + (j // 4)]
                            * ((-1) ** ((j & 2) >> 1))
                            * invMONT
                        )
                        self.sk_hat.vec[i_k].coeffs[j : j + 2] = solve_2x2(
                            b_hat_i.coeffs[j : j + 2],
                            c_hat_poly.coeffs[j : j + 2],
                            zeta,
                        )
                        self.sk_set.vec[i_k].coeffs[j : j + 2] = [1, 1]
