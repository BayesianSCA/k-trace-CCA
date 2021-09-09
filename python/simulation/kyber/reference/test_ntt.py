import numpy as np
import math
import pytest
from .ntt import ntt, invntt
from .reduce import montgomery_reduce, barrett_reduce, KYBER_Q, MONT


def test_barret_reduce():
    for iteration_nbr in range(100):
        rng = np.random.default_rng(iteration_nbr)
        a = rng.integers(-(2 ** 15), 2 ** 15, dtype=np.int16)
        b = barrett_reduce(a)

        assert b % KYBER_Q == a % KYBER_Q
        assert b > -KYBER_Q
        assert b < KYBER_Q
        assert b >= -(KYBER_Q - 1) // 2
        assert b <= (KYBER_Q - 1) // 2
        assert isinstance(b, np.int16)


def test_barret_reduce_full():
    for a_val in range(-(2 ** 15), 2 ** 15):
        a = np.int16(a_val)
        b = barrett_reduce(a)

        assert b % KYBER_Q == a % KYBER_Q
        assert b > -KYBER_Q
        assert b < KYBER_Q
        assert b >= -(KYBER_Q - 1) // 2
        assert b <= (KYBER_Q - 1) // 2
        assert isinstance(b, np.int16)


def test_montgomery_reduce():
    for iteration_nbr in range(100):
        rng = np.random.default_rng(iteration_nbr)
        a = rng.integers(-KYBER_Q * 2 ** 15, KYBER_Q * 2 ** 15, dtype=np.int32)
        b = montgomery_reduce(a)

        # NOTE: extra montgomery factor (normally included in zetas)
        assert (b * MONT) % KYBER_Q == a % KYBER_Q
        assert b > -KYBER_Q
        assert b < KYBER_Q
        assert isinstance(b, np.int16)


def test_montgomery_reduce_full():
    for a_val in range(-KYBER_Q * 2 ** 15, KYBER_Q * 2 ** 15):
        a = np.int32(a_val)
        b = montgomery_reduce(a)

        # NOTE: extra montgomery factor (normally included in zetas)
        assert (b * MONT) % KYBER_Q == a % KYBER_Q
        assert b > -KYBER_Q
        assert b < KYBER_Q
        assert isinstance(b, np.int16)


@pytest.mark.parametrize("height", [256, 128, 16, 8])
def test_ntt_consistency(height):
    layers = int(math.log2(height)) - 1
    for iteration_nbr in range(100):
        rng = np.random.default_rng(iteration_nbr)
        poly = rng.integers(-KYBER_Q + 1, KYBER_Q, height, dtype=np.int16)
        poly_hat, _ = ntt(poly.copy(), height=height, layers=layers)
        poly_new, _, _ = invntt(poly_hat.copy(), height=height, layers=layers)

        # NOTE: invntt multiplies extra montgomery-factor onto polynomial
        poly_mont = poly.astype(np.int32) * MONT % KYBER_Q
        poly_new_red = poly_new % KYBER_Q

        np.testing.assert_equal(poly_mont, poly_new_red)


if __name__ == '__main__':
    pytest.main([__file__])
