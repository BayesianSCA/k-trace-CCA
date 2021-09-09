import numpy as np
import math
from typing import Tuple
from .reduce import (
    montgomery_reduce,
    barrett_reduce,
    KYBER_Q,
    MONT,
    montgomery_reduce_outrange,
    barret_reduce_outrange,
)

zetas = np.array(
    [
        -1044,
        -758,
        -359,
        -1517,
        1493,
        1422,
        287,
        202,
        -171,
        622,
        1577,
        182,
        962,
        -1202,
        -1474,
        1468,
        573,
        -1325,
        264,
        383,
        -829,
        1458,
        -1602,
        -130,
        -681,
        1017,
        732,
        608,
        -1542,
        411,
        -205,
        -1571,
        1223,
        652,
        -552,
        1015,
        -1293,
        1491,
        -282,
        -1544,
        516,
        -8,
        -320,
        -666,
        -1618,
        -1162,
        126,
        1469,
        -853,
        -90,
        -271,
        830,
        107,
        -1421,
        -247,
        -951,
        -398,
        961,
        -1508,
        -725,
        448,
        -1065,
        677,
        -1275,
        -1103,
        430,
        555,
        843,
        -1251,
        871,
        1550,
        105,
        422,
        587,
        177,
        -235,
        -291,
        -460,
        1574,
        1653,
        -246,
        778,
        1159,
        -147,
        -777,
        1483,
        -602,
        1119,
        -1590,
        644,
        -872,
        349,
        418,
        329,
        -156,
        -75,
        817,
        1097,
        603,
        610,
        1322,
        -1285,
        -1465,
        384,
        -1215,
        -136,
        1218,
        -1335,
        -874,
        220,
        -1187,
        -1659,
        -1185,
        -1530,
        -1278,
        794,
        -1510,
        -854,
        -870,
        478,
        -108,
        -308,
        996,
        991,
        958,
        -1460,
        1522,
        1628,
    ],
    dtype=np.int16,
)

# zetas = np.array(range(128), dtype=np.int16);

# /*************************************************
# * Name:        fqmul
# *
# * Description: Multiplication followed by Montgomery reduction
# *
# * Arguments:   - int16_t a: first factor
# *              - int16_t b: second factor
# *
# * Returns 16-bit integer congruent to a*b*R^{-1} mod q
# **************************************************/
def fqmul(a: np.int16, b: np.int16) -> np.int16:
    return montgomery_reduce(np.int32(a) * b)


# /*************************************************
# * Name:        ntt
# *
# * Description: Inplace number-theoretic transform (NTT) in Rq.
# *              input is in standard order, output is in bitreversed order
# *
# * Arguments:   - int16_t r[256]: pointer to input/output vector of elements of Zq
# **************************************************/
def ntt(
    r: np.ndarray, height: int = 256, layers: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    if not math.log2(height).is_integer():
        raise ValueError('Height not power of 2, no way to connect this.')
    if layers >= math.log2(height):
        raise ValueError('Two many layers for this height, no way to connect them.')

    intermediate_values = np.zeros((height, layers + 1), dtype=np.int16)

    k = 1
    dist = height // 2  # 128
    layer = 0
    while dist >= 2:
        intermediate_values[:, layer] = r[:]
        start = 0
        while start < height:  # 256
            zeta = zetas[k]
            k += 1
            j = start
            while j < start + dist:
                t = fqmul(zeta, r[j + dist])
                r[j + dist] = r[j] - t
                r[j] = r[j] + t
                j += 1
            start = j + dist
        dist >>= 1
        layer += 1
    intermediate_values[:, layer] = r[:]

    return r, intermediate_values


# /*************************************************
# * Name:        invntt_tomont
# *
# * Description: Inplace inverse number-theoretic transform in Rq and
# *              multiplication by Montgomery factor 2^16.
# *              Input is in bitreversed order, output is in standard order
# *
# * Arguments:   - int16_t r[256]: pointer to input/output vector of elements of Zq
# **************************************************/
def invntt(
    r: np.ndarray, height: int = 256, layers: int = 7, start_layer=0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.log2(height).is_integer():
        raise ValueError('Height not power of 2, no way to connect this.')
    if layers >= math.log2(height):
        raise ValueError('Too many layers for this height, no way to connect them.')

    intermediate_values = np.zeros((height, layers + 2), dtype=np.int16)
    intermediate_values_range = np.zeros((height, layers + 2, 2), dtype=np.int16)
    # mont^2//(height//2), extra montgomery factor and size-adjust, for height=256: 1441
    f = np.int16(
        MONT
        * (MONT * (KYBER_Q - 1) * ((KYBER_Q - 1) // (height // 2)) % KYBER_Q)
        % KYBER_Q
    )

    k = height // 2 - 1  # 127
    dist = 2
    layer = 0
    intermediate_values[:, layer] = r[:]
    for j in range(0, height):
        intermediate_values_range[
            j, layer
        ] = barret_reduce_outrange  # NOTE: Final barret reduction in polyvec accumulate mul
    layer += 1
    while dist <= height // 2:
        start = 0
        while start < height:  # 256
            zeta = zetas[k]
            k -= 1
            j = start
            while j < start + dist:
                if layer > start_layer:
                    t = r[j]
                    r[j] = barrett_reduce(t + r[j + dist])
                    r[j + dist] = r[j + dist] - t
                    r[j + dist] = fqmul(zeta, r[j + dist])
                    intermediate_values_range[j, layer] = barret_reduce_outrange
                    intermediate_values_range[
                        j + dist, layer
                    ] = montgomery_reduce_outrange
                j += 1
            start = j + dist
        intermediate_values[:, layer] = r[:]
        dist <<= 1
        layer += 1
        if layer > layers:  # early abort for smaller sub-ntt (less layers)
            break

    for j in range(0, height):
        r[j] = fqmul(r[j], f)
        intermediate_values_range[j, layer] = montgomery_reduce_outrange
    intermediate_values[:, layer] = r[:]

    return r, intermediate_values, intermediate_values_range


# /*************************************************
# * Name:        basemul
# *
# * Description: Multiplication of polynomials in Zq[X]/(X^2-zeta)
# *              used for multiplication of elements in Rq in NTT domain
# *
# * Arguments:   - int16_t r[2]: pointer to the output polynomial
# *              - const int16_t a[2]: pointer to the first factor
# *              - const int16_t b[2]: pointer to the second factor
# *              - int16_t zeta: integer defining the reduction polynomial
# **************************************************/
def basemul(a: np.ndarray, b: np.ndarray, zeta: np.int16) -> np.ndarray:
    r = np.zeros(2, dtype=np.int16)
    r[0] = fqmul(a[1], b[1])
    r[0] = fqmul(r[0], zeta)
    r[0] += fqmul(a[0], b[0])

    r[1] = fqmul(a[0], b[1])
    r[1] += fqmul(a[1], b[0])

    return r
