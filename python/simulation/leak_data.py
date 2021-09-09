import numpy as np
from scipy.stats import norm

from .gen_input import INTTInputMasked, INTTInputUnmasked
from .kyber.reference.ntt import invntt
from helpers.misc import find_zero_pairs

KYBER_Q = 3329
int_size = 16


class LeakDataMasked:
    def __init__(
        self,
        maskshare,
        skm,
        intermediates_mask,
        intermediates_skm,
        hw_leakage_mask,
        hw_leakage_skm,
        sigma,
        height,
        layers,
        intermediate_values_range,
        zero_indices,
    ):

        self.intermediates_mask = intermediates_mask
        self.intermediates_skm = intermediates_skm
        self.hw_leakage_mask = hw_leakage_mask
        self.hw_leakage_skm = hw_leakage_skm
        self.sigma = sigma
        self.height = height
        self.layers = layers
        self.intermediate_values_range = intermediate_values_range
        self.zero_indices = zero_indices
        self.maskshare = maskshare
        self.skm = skm

    def get_leak_dict_mask(self):
        return {
            (h, l): hw_prior_to_value_prior(
                h,
                l,
                self.hw_leakage_mask,
                self.intermediate_values_range,
                self.zero_indices,
            )
            for h in range(self.height)
            for l in range(self.layers + 1)
        }

    def get_leak_dict_skm(self):
        return {
            (h, l): hw_prior_to_value_prior(
                h,
                l,
                self.hw_leakage_skm,
                self.intermediate_values_range,
                self.zero_indices,
            )
            for h in range(self.height)
            for l in range(self.layers + 1)
        }

    @staticmethod
    def generate(rng, inttinput, layers, sigma):
        maskshare, skm = inttinput.get_intt_inputs()
        _, intermediates_mask, intermediate_values_range = invntt(
            maskshare.coeffs.copy(), inttinput.height(), layers
        )
        _, intermediates_skm, _ = invntt(skm.coeffs.copy(), inttinput.height(), layers)

        hw_mask = hw_leakage(intermediates_mask, sigma, rng)
        hw_skm = hw_leakage(intermediates_skm, sigma, rng)

        zero_indices = find_zero_pairs(inttinput.bhat.vec)

        assert all(
            [skm.coeffs[i] == 0 and maskshare.coeffs[i] == 0 for i in zero_indices]
        )

        return LeakDataMasked(
            maskshare,
            skm,
            intermediates_mask,
            intermediates_skm,
            hw_mask,
            hw_skm,
            sigma,
            inttinput.height(),
            layers,
            intermediate_values_range,
            zero_indices,
        )


class LeakDataUnmasked:
    def __init__(
        self,
        sk,
        intermediates,
        hw_leakage,
        sigma,
        height,
        layers,
        intermediate_values_range,
        zero_indices,
    ):

        self.sk = sk
        self.intermediates = intermediates
        self.hw_leakage = hw_leakage
        self.sigma = sigma
        self.height = height
        self.layers = layers
        self.intermediate_values_range = intermediate_values_range
        self.zero_indices = zero_indices

    def get_leak_dict(self):
        return {
            (h, l): hw_prior_to_value_prior(
                h, l, self.hw_leakage, self.intermediate_values_range, self.zero_indices
            )
            for h in range(self.height)
            for l in range(self.layers + 1)
        }

    @staticmethod
    def generate(rng, inttinput, layers, sigma):
        skin = inttinput.get_intt_input()
        _, intermediates, intermediate_values_range = invntt(
            skin.coeffs.copy(), inttinput.height(), layers
        )

        hw = hw_leakage(intermediates, sigma, rng)

        zero_indices = find_zero_pairs(inttinput.bhat.vec)
        assert all([skin.coeffs[i] == 0 for i in zero_indices])

        return LeakDataUnmasked(
            skin,
            intermediates,
            hw,
            sigma,
            inttinput.height(),
            layers,
            intermediate_values_range,
            zero_indices,
        )


def hw_leakage(interm, sigma, rng):
    # rng = np.random.default_rng(seed=seed)
    leak = np.zeros((interm.shape[0], interm.shape[1], int_size + 1))
    for i in range(0, leak.shape[0]):
        for j in range(0, leak.shape[1]):

            # Compute Hamming Weight of an intermediate
            hw = hw_int(interm[i, j])
            # Add leakage according to gaussian noise
            hw_leak = hw + rng.normal(0, sigma, 1)
            range_hw = range(0, int_size + 1)

            # Template matching
            p = norm.pdf(hw_leak, range_hw, sigma)
            leak[i, j] = p

    return leak


def hw_int(value):
    """return HW of 16-bit signed integer in two's complement"""
    if value < 0:
        result = int_size - hw_uint(-value - 1)
    else:
        result = hw_uint(value)
    return result


def hw_uint(value):
    """return HW of 16-bit unsigned integer in two's complement"""
    bitcount = bin(value).count("1")
    return bitcount


def hw_prior_to_value_prior(h, l, priors_hw, intermediate_values_range, zero_indices):
    v_range = range(
        intermediate_values_range[h, l, 0], intermediate_values_range[h, l, 1] + 1
    )
    if check_zero_block(h, l, zero_indices):
        return {0: 1}
    return {v: priors_hw[h, l, hw_int(v)] for v in v_range}


def get_block_indices(height, layer):
    block = [height]
    for i_layer in range(layer):
        dist = 2 << i_layer
        new_block = []
        for h_block in block:
            op1 = h_block & ~dist  # first operator of butterfly
            op2 = op1 + dist  # second operator of butterfly
            new_block += [op1, op2]
        block = new_block
    return block


def check_zero_block(height, layer, zero_indices):
    block = get_block_indices(height, layer)
    return set(block).issubset(zero_indices)
