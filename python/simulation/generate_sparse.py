import numpy as np
from .kyber.reference.ntt import invntt, ntt
from .kyber.reference.reduce import barrett_reduce
from .kyber.reference.params import KYBER_N, KYBER_Q

# intermediate values in layer 5 of invntt which are true under compression
# split by blocks of 64 coeffs (KYBER_N//4)
compressed_sparse_interm = np.array(
    [
        [0, 1653, 1704],  # block 0
        [0, 1340, 2920],  # block 1
        [0, 1698, 73],  # block 2
        [0, 1289, 3038],
    ],  # block 3
    dtype=np.int16,
)
interm_layer = 5
interm_p = None  # [0.33,0.33,0.33]  # for random choice

# better analysis: all blocks have the same values!
# 64 value blocks
compressed_sparse_interm_5 = np.array(
    [
        51,
        73,
        124,
        285,
        291,
        336,
        409,
        460,
        1289,
        1340,
        1413,
        1574,
        1580,
        1625,
        1631,
        1653,
        1676,
        1698,
        1704,
        1749,
        1755,
        1916,
        1989,
        2040,
        2869,
        2920,
        2993,
        3038,
        3044,
        3205,
        3256,
        3278,
    ],
    dtype=np.int16,
)

# 128 value blocks
compressed_sparse_interm_6 = np.array(
    [
        6,
        12,
        16,
        45,
        51,
        61,
        67,
        73,
        96,
        106,
        112,
        118,
        124,
        157,
        163,
        173,
        179,
        185,
        218,
        224,
        230,
        236,
        269,
        275,
        281,
        285,
        291,
        297,
        330,
        336,
        342,
        348,
        381,
        387,
        393,
        403,
        409,
        438,
        442,
        448,
        454,
        460,
        466,
        493,
        499,
        505,
        515,
        521,
        550,
        554,
        560,
        566,
        572,
        578,
        605,
        611,
        617,
        627,
        633,
        662,
        672,
        678,
        684,
        690,
        717,
        723,
        729,
        735,
        739,
        745,
        774,
        784,
        790,
        796,
        802,
        829,
        835,
        841,
        847,
        857,
        863,
        886,
        896,
        902,
        908,
        914,
        947,
        953,
        959,
        969,
        975,
        1004,
        1008,
        1014,
        1020,
        1026,
        1032,
        1059,
        1065,
        1071,
        1081,
        1087,
        1116,
        1126,
        1132,
        1138,
        1144,
        1171,
        1177,
        1183,
        1193,
        1199,
        1228,
        1238,
        1244,
        1250,
        1256,
        1283,
        1289,
        1295,
        1301,
        1305,
        1311,
        1340,
        1350,
        1356,
        1362,
        1368,
        1395,
        1401,
        1407,
        1413,
        1423,
        1452,
        1462,
        1468,
        1474,
        1480,
        1507,
        1513,
        1519,
        1525,
        1535,
        1564,
        1570,
        1574,
        1580,
        1586,
        1592,
        1619,
        1625,
        1631,
        1637,
        1647,
        1653,
        1676,
        1682,
        1692,
        1698,
        1704,
        1710,
        1737,
        1743,
        1749,
        1755,
        1759,
        1765,
        1794,
        1804,
        1810,
        1816,
        1822,
        1849,
        1855,
        1861,
        1867,
        1877,
        1906,
        1916,
        1922,
        1928,
        1934,
        1961,
        1967,
        1973,
        1979,
        1989,
        2018,
        2024,
        2028,
        2034,
        2040,
        2046,
        2073,
        2079,
        2085,
        2091,
        2101,
        2130,
        2136,
        2146,
        2152,
        2158,
        2185,
        2191,
        2197,
        2203,
        2213,
        2242,
        2248,
        2258,
        2264,
        2270,
        2297,
        2303,
        2309,
        2315,
        2321,
        2325,
        2354,
        2360,
        2370,
        2376,
        2382,
        2415,
        2421,
        2427,
        2433,
        2443,
        2466,
        2472,
        2482,
        2488,
        2494,
        2527,
        2533,
        2539,
        2545,
        2555,
        2584,
        2590,
        2594,
        2600,
        2606,
        2612,
        2639,
        2645,
        2651,
        2657,
        2667,
        2696,
        2702,
        2712,
        2718,
        2724,
        2751,
        2757,
        2763,
        2769,
        2775,
        2779,
        2808,
        2814,
        2824,
        2830,
        2836,
        2863,
        2869,
        2875,
        2881,
        2887,
        2891,
        2920,
        2926,
        2936,
        2942,
        2948,
        2981,
        2987,
        2993,
        2999,
        3032,
        3038,
        3044,
        3048,
        3054,
        3060,
        3093,
        3099,
        3105,
        3111,
        3144,
        3150,
        3156,
        3166,
        3172,
        3205,
        3211,
        3217,
        3223,
        3233,
        3256,
        3262,
        3268,
        3278,
        3284,
        3313,
        3317,
        3323,
    ],
    dtype=np.int16,
)
# Note: 1871 only works in first half, 1458 only in second half - not used

# 32 value blocks (only KYBER-1024)
compressed_sparse_interm_4 = np.array(
    [
        44,
        61,
        73,
        103,
        171,
        182,
        216,
        243,
        267,
        285,
        334,
        409,
        491,
        580,
        616,
        622,
        693,
        790,
        791,
        798,
        852,
        889,
        917,
        962,
        1020,
        1059,
        1088,
        1202,
        1337,
        1347,
        1370,
        1413,
        1468,
        1474,
        1518,
        1536,
        1569,
        1577,
        1639,
        1650,
        1679,
        1690,
        1752,
        1760,
        1793,
        1811,
        1855,
        1861,
        1916,
        1959,
        1982,
        1992,
        2127,
        2241,
        2270,
        2309,
        2367,
        2412,
        2440,
        2477,
        2531,
        2538,
        2539,
        2636,
        2707,
        2713,
        2749,
        2838,
        2920,
        2995,
        3044,
        3062,
        3086,
        3113,
        3147,
        3158,
        3226,
        3256,
        3268,
        3285,
    ],
    dtype=np.int16,
)


def is_compressed(x: np.int16, kyber_k) -> bool:
    x = barrett_reduce(x)
    x += (x >> 15) & KYBER_Q  # make sure its positive
    if kyber_k == 4:
        compressed = np.uint16(
            (((np.uint32(x) << 11) + KYBER_Q // 2) // KYBER_Q) & 0x7FF
        )
        decompressed = np.int16((np.uint32(compressed & 0x7FF) * KYBER_Q + 1024) >> 11)
    elif kyber_k == 2 or kyber_k == 3:
        compressed = np.uint16(
            (((np.uint32(x) << 10) + KYBER_Q // 2) // KYBER_Q) & 0x3FF
        )
        decompressed = np.int16((np.uint32(compressed & 0x3FF) * KYBER_Q + 512) >> 10)
    else:
        raise NotImplementedError("Not implemented for Kyber K = {}".format(kyber_k))
    return decompressed % KYBER_Q == x % KYBER_Q


def is_sparse(blocks: list, r_hat: np.array, block_size) -> bool:
    # get nonzeros
    nonzeros = r_hat != 0
    cnt_nonzeros = np.count_nonzero(nonzeros)
    cnt_zeros = len(r_hat) - cnt_nonzeros
    # print("Values: Zeros: {:3}, Nonzeros {:3}".format(cnt_zeros, cnt_nonzeros))
    # get nonzero pairs (for pairwise-pointwise)
    nonzeros.shape = (-1, 2)
    nonzero_pairs = np.sum(nonzeros, axis=1)
    cnt_nonzero_pairs = np.count_nonzero(nonzero_pairs)
    cnt_zero_pairs = len(nonzero_pairs) - cnt_nonzero_pairs
    # print("Pairs:  Zeros: {:3}, Nonzeros {:3}".format(cnt_zero_pairs, cnt_nonzero_pairs))
    # get nonzero pairs within the blocks
    nbr_blocks = KYBER_N // block_size
    nonzero_pairs.shape = (nbr_blocks, -1)
    nonzero_blocks = np.sum(nonzero_pairs, axis=1)
    cnt_nonzero_blocks = np.count_nonzero(nonzero_blocks)
    cnt_zero_blocks = len(nonzero_blocks) - cnt_nonzero_blocks
    # print("Blocks: Zeros: {:3}, Nonzeros {:3}".format(cnt_zero_blocks, cnt_nonzero_blocks))
    # print(nonzero_blocks)
    # Note: single zeros would actually actually be nice, but not necessary (for each pair)
    # Check for linear independence of neighboring coeffs, i.e. they cannot be equal (this is not actually necessary)
    r_diff = np.diff(np.reshape(r_hat, (-1, 2)), axis=1)
    diff_nonzero = r_diff != 0
    diff_nonzero.shape = (nbr_blocks, -1)
    diff_nonzero_blocks = np.sum(diff_nonzero, axis=1)
    for block in range(nbr_blocks):
        if block in blocks:
            # if nonzero_blocks[block] != 2*nonzero_pairs.shape[1]:
            #     return False  # "Wanted block is not fully set."
            #     # Note: this doesn't matter and is actually nice for solving the small 2x2 Gauss
            if diff_nonzero_blocks[block] != diff_nonzero.shape[1]:
                return False  # Not all pairs are linear independent in the set block
        else:
            if nonzero_blocks[block] != 0:
                raise ValueError("Sparse block is nonzero. Please report!")
    return True


def gen_sparse(
    blocks: list, rng: np.random.Generator, nbr_retries=3, block_size=64, kyber_k=3
):
    """blocks: list of nonzero blocks, possibilities: [0],[1],[2],[3],[0,1],[2,3]
    It will retry generating a fully set nonzero block for nbr_retries
    NOTE: only works for full KYBER_N NTT
    """
    if len(blocks) == 1 and block_size == 64:
        assert blocks[0] in range(4), "Block value out of range"
        interm_layer = 5
        interm_values = compressed_sparse_interm_5
    elif len(blocks) == 2 and block_size == 64:
        assert min(blocks) in [
            0,
            2,
        ], "Only even starting blocks allowed for 2-block sparse value"
        assert max(blocks) == min(blocks) + 1, "Only neighboring blocks allowed"
        interm_layer = 6
        interm_values = compressed_sparse_interm_6
    elif len(blocks) == 1 and block_size == 32:
        assert blocks[0] in range(8), "Block value out of range"
        assert kyber_k == 4, "Block size of 32 only works for KYBER-1024"
        interm_layer = 4
        interm_values = compressed_sparse_interm_4
    else:
        raise ValueError("Number of blocks for sparse vector not supported.")
    for i_try in range(nbr_retries):
        r_int = np.zeros(KYBER_N, dtype=np.int16)
        # randomly set values:
        for block in blocks:
            r_int[block * block_size : (block + 1) * block_size] = rng.choice(
                interm_values, block_size, p=interm_p
            )
        # r_int[0] = compressed_sparse_interm[0,1]  # debug, set single inter only
        r, _, _ = invntt(r_int.copy(), start_layer=interm_layer)
        np.testing.assert_equal(
            is_compressed(r, kyber_k), True, "Generated uncompressed polynomial"
        )
        r_hat, _ = ntt(r)
        r_hat = barrett_reduce(r_hat)  # reference reduces after ntt in poly_ntt
        if is_sparse(blocks, r_hat, block_size):
            break
        print(
            "Not all pairs in set block linear independent, retrying gen_sparse, i_try: ",
            i_try,
        )
    else:  # if no proper sparse polynomial was found
        raise ValueError("No proper sparse polynomial found")
    return r_hat
