def find_zero_pairs(pvec):
    all_zeros = lambda l: all(map(lambda x: x == 0, l))
    coeffs_bhs = list(zip(*map(lambda x: x.coeffs, pvec)))
    zero_indices = []
    for i, bhs in list(enumerate(coeffs_bhs))[1::2]:
        if all_zeros(bhs) and all_zeros(coeffs_bhs[i - 1]):
            zero_indices.append(i - 1)
            zero_indices.append(i)
    return zero_indices
