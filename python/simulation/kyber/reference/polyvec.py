import numpy as np
from .poly import (
    poly,
    poly_basemul_montgomery,
    poly_add,
    poly_reduce,
    poly_getnoise_eta1,
    poly_ntt,
    poly_invntt,
    poly_sub,
)
from .params import KYBER_N


class polyvec(np.ndarray):
    def __new__(cls, height, vec_k):
        obj = np.zeros((vec_k, height), dtype=np.int16).view(cls)
        return obj

    def len(self):
        return self.height

    @property
    def height(self):
        return self.shape[1]

    @property
    def vec_k(self):
        return self.shape[0]

    @property
    def vec(self):
        return tuple(
            self[i] for i in range(self.vec_k)
        )  # NOTE: tuple, as item assignment will cause object reference problems
        # return tuple(self)  # explicit range used instead, to avoid annoying exception for debugging

    @vec.setter
    def vec(self, val):
        self[:] = val

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key).view(poly)
        else:
            return super().__getitem__(key)


def polyvec_reduce(r: polyvec) -> polyvec:
    # r.vec = list(map(poly_reduce, r.vec))
    # return r
    return poly_reduce(r)


def polyvec_add(p0: polyvec, p1: polyvec) -> polyvec:
    return p0 + p1


def polyvec_sub(p0: polyvec, p1: polyvec) -> polyvec:
    return p0 - p1


def polyvec_getnoise_eta1(rng, height, vec_k):
    r = polyvec(height, vec_k)
    for i in range(r.vec_k):
        r[i] = poly_getnoise_eta1(rng, height)
    return r


# /*************************************************
# * Name:        polyvec_ntt
# *
# * Description: Apply forward NTT to all elements of a vector of polynomials
# *
# * Arguments:   - polyvec *r: pointer to in/output vector of polynomials
# **************************************************/
def polyvec_ntt(r: polyvec) -> polyvec:
    r_hat = r.copy()
    for i in range(r.vec_k):
        r_hat[i] = poly_ntt(r_hat[i])
    return r_hat


def polyvec_invntt(r_hat: polyvec) -> polyvec:
    r = r_hat.copy()
    for i in range(r.vec_k):
        r[i] = poly_invntt(r[i])
    return r


# /*************************************************
# * Name:        polyvec_basemul_acc_montgomery
# *
# * Description: Multiply elements of a and b in NTT domain, accumulate into r,
# *              and multiply by 2^-16.
# *
# * Arguments: - poly *r: pointer to output polynomial
# *            - const polyvec *a: pointer to first input vector of polynomials
# *            - const polyvec *b: pointer to second input vector of polynomials
# **************************************************/
def polyvec_basemul_acc_montgomery(a: polyvec, b: polyvec) -> poly:
    assert a.height == b.height
    assert a.vec_k == b.vec_k
    r = poly_basemul_montgomery(a[0], b[0])
    for i in range(1, a.vec_k):
        t = poly_basemul_montgomery(a[i], b[i])
        r = poly_add(r, t)
    r = poly_reduce(r)
    return r
