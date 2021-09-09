import numpy as np
from enum import Enum
from .kyber.reference.polyvec import (
    polyvec,
    polyvec_basemul_acc_montgomery,
    polyvec_sub,
    polyvec_reduce,
)
from .kyber.reference.reduce import barret_reduce_outrange
from .kyber.reference.indcpa import indcpa_keypair
from .kyber.reference.params import KYBER_N
from .kyber.reference.poly import poly, poly_reduce
from helpers.misc import find_zero_pairs
from .generate_sparse import gen_sparse


class INTTInputMasked:
    def __init__(self, skpv, mask, bhat):
        self.sk = skpv
        self.mask = mask
        self.bhat = bhat

    def height(self):
        return self.sk.len()

    def get_sk_masked(self):
        sub = polyvec_sub(self.sk, self.mask)
        return polyvec_reduce(sub)

    def get_intt_inputs(self):
        input_share0 = polyvec_basemul_acc_montgomery(self.bhat, self.mask)
        input_share1 = polyvec_basemul_acc_montgomery(self.bhat, self.get_sk_masked())
        return input_share0, input_share1

    @staticmethod
    def generate(rng, height, vec_k, skpv, b_hat):
        mask = gen_mask(rng, height, vec_k)
        return INTTInputMasked(skpv, mask, b_hat)


class INTTInputUnmasked:
    def __init__(self, skpv, bhat):
        self.sk = skpv
        self.bhat = bhat

    def height(self):
        return self.sk.len()

    def get_sk(self):
        return polyvec_reduce(self.sk)

    def get_intt_input(self):
        input0 = polyvec_basemul_acc_montgomery(self.bhat, self.get_sk())
        return input0

    @staticmethod
    def generate(rng, height, vec_k, skpv, b_hat):
        return INTTInputUnmasked(skpv, b_hat)


class BHatInfo:
    class BHatInfoType(Enum):
        ZEROS = 0
        RANDOM_ZEROS = 1
        GENERATED_BHAT = 2
        GENERATED_BHAT_REARRANGE = 3
        GENERATED_BHAT_32 = 4

    def __init__(self, info_type, info):
        self.info_type = info_type
        self.info = info

    @classmethod
    def from_random_zeros(cls, number_of_random_zeros):
        return cls(cls.BHatInfoType.RANDOM_ZEROS, number_of_random_zeros)

    @classmethod
    def from_zeros(cls, zero_indices):
        return cls(cls.BHatInfoType.ZEROS, zero_indices)

    @classmethod
    def from_generated_bhat(cls, blocks):
        return cls(cls.BHatInfoType.GENERATED_BHAT, blocks)

    @classmethod
    def from_generated_bhat_32(cls, blocks):
        return cls(cls.BHatInfoType.GENERATED_BHAT_32, blocks)

    @classmethod
    def from_generated_bhat_rearrange(cls, nrblocks):
        return cls(cls.BHatInfoType.GENERATED_BHAT_REARRANGE, nrblocks)

    def generate_bhat(self, rng, height, vec_k):
        if self.info_type == BHatInfo.BHatInfoType.ZEROS:
            zero_indices = self.info
            bhat = gen_sparse_b_hat(zero_indices, rng, height, vec_k)
        elif self.info_type == BHatInfo.BHatInfoType.RANDOM_ZEROS:
            zero_indices = rng.integers(0, height // 2, self.info // 2)
            zero_indices = (2 * zero_indices).tolist()
            zero_indices += [i + 1 for i in zero_indices]
            bhat = gen_sparse_b_hat(zero_indices, rng, height, vec_k)
        elif self.info_type == BHatInfo.BHatInfoType.GENERATED_BHAT:
            assert height == KYBER_N
            bhat = polyvec(height, vec_k)
            for dimension, blocks in enumerate(self.info):
                rhat = poly(height)
                rhat.coeffs = gen_sparse(blocks, rng)
                bhat[dimension] = rhat
            zero_indices = find_zero_pairs(bhat.vec)
            print(zero_indices)
        elif self.info_type == BHatInfo.BHatInfoType.GENERATED_BHAT_32:
            assert height == KYBER_N
            assert vec_k == 4
            bhat = polyvec(height, vec_k)
            for dimension, blocks in enumerate(self.info):
                rhat = poly(height)
                rhat.coeffs = gen_sparse(blocks, rng, block_size=32, kyber_k=4)
                bhat[dimension] = rhat
            zero_indices = find_zero_pairs(bhat.vec)
            print(zero_indices)
        elif self.info_type == BHatInfo.BHatInfoType.GENERATED_BHAT_REARRANGE:
            assert height == KYBER_N
            bhat = polyvec(height, vec_k)

            offsets = [0, 2, 1, 3]
            if self.info == 0.5:  # this means 32 set coefficients
                nrblocks = 1
            else:
                nrblocks = self.info
            for nbl in range(nrblocks):
                rhat = poly(height)
                rhat.coeffs = gen_sparse([nbl], rng)
                rhat_rearrange = poly(height)
                offset = offsets[nbl]
                for j, i in enumerate(range(0, height, 8)):
                    rhat_rearrange.coeffs[i + (offset * 2)] = rhat.coeffs[
                        j * 2 + 64 * nbl
                    ]
                    rhat_rearrange.coeffs[i + 1 + (offset * 2)] = rhat.coeffs[
                        j * 2 + 1 + 64 * nbl
                    ]
                assert list(
                    filter(lambda num: num != 0, rhat_rearrange.coeffs)
                ) == list(filter(lambda num: num != 0, rhat.coeffs))
                bhat[nbl % 3] += rhat_rearrange
            if self.info == 0.5:  # this means 32 set coefficients
                # remove extra coefficients (simple solution)
                bhat[:, 8::16] = 0
                bhat[:, 9::16] = 0
            zero_indices = find_zero_pairs(bhat.vec)
            print(zero_indices)

        else:
            print("ValueError: BHatInfoType is {}.".format(self.info))
            raise ValueError("Unknown bhat info type.")
        return bhat, len(zero_indices)

    def get_nbr_nonzeros(self, height, vec_k):
        if self.info_type == BHatInfo.BHatInfoType.ZEROS:
            zero_indices = self.info
            nbr_nonzeros = height - len(zero_indices)
        elif self.info_type == BHatInfo.BHatInfoType.RANDOM_ZEROS:
            raise NotImplementedError
        elif self.info_type == BHatInfo.BHatInfoType.GENERATED_BHAT:
            assert height == KYBER_N
            nonzero_blocks = set()
            for dimension, blocks in enumerate(self.info):
                nonzero_blocks.update(blocks)
            nbr_nonzeros = len(nonzero_blocks) * KYBER_N // 4
        elif self.info_type == BHatInfo.BHatInfoType.GENERATED_BHAT_32:
            assert height == KYBER_N
            assert vec_k == 4
            assert height == KYBER_N
            nonzero_blocks = set()
            for dimension, blocks in enumerate(self.info):
                nonzero_blocks.update(blocks)
            nbr_nonzeros = len(nonzero_blocks) * KYBER_N // 8
        elif self.info_type == BHatInfo.BHatInfoType.GENERATED_BHAT_REARRANGE:
            raise NotImplementedError
        else:
            print("ValueError: BHatInfoType is {}.".format(self.info))
            raise ValueError("Unknown bhat info type.")
        return nbr_nonzeros

    @property
    def __key(self):
        return (self.info_type, self.info)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__key == other.__key
        return NotImplemented

    def __repr__(self):
        return "{}_{}".format(self.info_type.value, self.info)


def gen_skpv(rng, height, vec_k):
    skpv, skpv_hat = indcpa_keypair(rng, height=height, vec_k=vec_k)
    return skpv, skpv_hat


def gen_mask(rng, height, vec_k):
    r = polyvec(height, vec_k)
    for i in range(r.vec_k):
        r.vec[i].coeffs = rng.integers(
            barret_reduce_outrange[0],
            barret_reduce_outrange[1] + 1,
            r.height,
            dtype=np.int16,
        )
    return r


def gen_sparse_b_hat(zero_indices, rng, height, vec_k):
    """
    In reference indcpa_dec:
     - b is decompressed and ntt-transformed
     - the output is barrett reduced,
     - i.e. integer in {-(q-1)/2,...,(q-1)/2} congruent to a modulo q
     barret_reduce_outrange = (-(KYBER_Q - 1)//2, (KYBER_Q - 1)//2)
    """
    r = polyvec(height, vec_k)
    for i in range(r.vec_k):
        r.vec[i].coeffs = rng.integers(
            barret_reduce_outrange[0],
            barret_reduce_outrange[1] + 1,
            r.height,
            dtype=np.int16,
        )
        r.vec[i].coeffs[zero_indices] = 0

    return r
