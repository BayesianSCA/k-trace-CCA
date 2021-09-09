import numpy as np
from .generate_sparse import gen_sparse


def test_gen_sparse_64():
    for iteration_nbr in range(100):
        rng = np.random.default_rng(iteration_nbr)
        gen_sparse([0], rng)
        gen_sparse([1], rng)
        gen_sparse([2], rng)
        gen_sparse([3], rng)


def test_gen_sparse_128():
    for iteration_nbr in range(100):
        rng = np.random.default_rng(iteration_nbr)
        gen_sparse([0, 1], rng)
        # gen_sparse([0,2], rng)
        # gen_sparse([0,3], rng)
        # gen_sparse([1,2], rng)
        # gen_sparse([1,3], rng)
        gen_sparse([2, 3], rng)


# def test_gen_sparse_192():
#     for iteration_nbr in range(100):
#         rng = np.random.default_rng(iteration_nbr)
#         gen_sparse([0,1,2], rng)
#         gen_sparse([0,1,3], rng)
#         gen_sparse([0,2,3], rng)
#         gen_sparse([1,2,3], rng)
