import numpy as np
import math
from .ntt import basemul, zetas, ntt, invntt
from .reduce import barrett_reduce
from .params import KYBER_ETA1, KYBER_N


class poly(np.ndarray):
    def __new__(cls, height=KYBER_N):
        obj = np.zeros(height, dtype=np.int16).view(cls)
        return obj

    @property
    def height(self):
        return self.shape[0]

    @property
    def coeffs(self):
        return self

    @coeffs.setter
    def coeffs(self, val):
        self[:] = val


# /*************************************************
# * Name:        poly_getnoise_eta1
# *
# * Description: Sample a polynomial deterministically from a seed and a nonce,
# *              with output polynomial close to centered binomial distribution
# *              with parameter KYBER_ETA1
# *
# * Arguments:   - poly *r: pointer to output polynomial
# *              - const uint8_t *seed: pointer to input seed
# *                                     (of length KYBER_SYMBYTES bytes)
# *              - uint8_t nonce: one-byte input nonce
# **************************************************/
def poly_getnoise_eta1(rng, height):
    r = poly_cbd_eta1(rng, height)
    return r


# /*************************************************
# * Name:        poly_ntt
# *
# * Description: Computes negacyclic number-theoretic transform (NTT) of
# *              a polynomial in place;
# *              inputs assumed to be in normal order, output in bitreversed order
# *
# * Arguments:   - uint16_t *r: pointer to in/output polynomial
# **************************************************/
def poly_ntt(r: poly) -> poly:
    r_hat, _ = ntt(r.copy(), r.height, int(math.log2(r.height)) - 1)
    r_hat = poly_reduce(r_hat)
    return r_hat


def poly_invntt(r_hat: poly) -> poly:
    r, _, _ = invntt(r_hat.copy(), r_hat.height, int(math.log2(r_hat.height)) - 1)
    return r


# /*************************************************
# * Name:        poly_basemul_montgomery
# *
# * Description: Multiplication of two polynomials in NTT domain
# *
# * Arguments:   - poly *r: pointer to output polynomial
# *              - const poly *a: pointer to first input polynomial
# *              - const poly *b: pointer to second input polynomial
# **************************************************/
def poly_basemul_montgomery(a: poly, b: poly) -> poly:
    assert a.height == b.height
    r = poly(height=a.height)
    zeta_offset = (
        a.height // 4
    )  # 64, NOTE: formula should be correct for smaller heights
    for i in range(a.height // 4):
        r[4 * i : 4 * i + 2] = basemul(
            a[4 * i : 4 * i + 2], b[4 * i : 4 * i + 2], zetas[zeta_offset + i]
        )
        r[4 * i + 2 : 4 * i + 4] = basemul(
            a[4 * i + 2 : 4 * i + 4], b[4 * i + 2 : 4 * i + 4], -zetas[zeta_offset + i]
        )
    return r


# /*************************************************
# * Name:        poly_reduce
# *
# * Description: Applies Barrett reduction to all coefficients of a polynomial
# *              for details of the Barrett reduction see comments in reduce.c
# *
# * Arguments:   - poly *r: pointer to input/output polynomial
# **************************************************/
def poly_reduce(r: poly) -> poly:
    return barrett_reduce(r)


# /*************************************************
# * Name:        poly_add
# *
# * Description: Add two polynomials; no modular reduction is performed
# *
# * Arguments: - poly *r: pointer to output polynomial
# *            - const poly *a: pointer to first input polynomial
# *            - const poly *b: pointer to second input polynomial
# **************************************************/
def poly_add(a: poly, b: poly) -> poly:
    return a + b


def poly_sub(a: poly, b: poly) -> poly:
    return a - b


# /*************************************************
# * Name:        cbd2
# *
# * Description: Given an array of uniformly random bytes, compute
# *              polynomial with coefficients distributed according to
# *              a centered binomial distribution with parameter eta=2
# *
# * Arguments:   - poly *r: pointer to output polynomial
# *              - const uint8_t *buf: pointer to input byte array
# **************************************************/
def poly_cbd_eta1(rng, height=KYBER_N, KYBER_ETA1=KYBER_ETA1) -> poly:
    assert KYBER_ETA1 == 2, "Not Implemented"
    # implementing cbd2():

    a = rng.integers(0, 2, height, dtype=np.int16)
    a += rng.integers(0, 2, height, dtype=np.int16)
    b = rng.integers(0, 2, height, dtype=np.int16)
    b += rng.integers(0, 2, height, dtype=np.int16)

    r = poly(height=height)
    r.coeffs = a - b
    return r
