import numpy as np

KYBER_Q = np.int16(3329)
QINV = np.int32(62209)
BARRET_V = np.int16(((1 << 26) + KYBER_Q // 2) // KYBER_Q)
MONT = np.int32(-1044)  # 2^16 mod q

# /*************************************************
# * Name:        montgomery_reduce
# *
# * Description: Montgomery reduction; given a 32-bit integer a, computes
# *              16-bit integer congruent to a * R^-1 mod q, where R=2^16
# *
# * Arguments:   - int32_t a: input integer to be reduced;
# *                           has to be in {-q2^15,...,q2^15-1}
# *
# * Returns:     integer in {-q+1,...,q-1} congruent to a * R^-1 modulo q.
# **************************************************/
def montgomery_reduce(a: np.int32) -> np.int16:
    u = np.int16(np.int64(a) * np.int64(QINV))  # dropping upper bits on purpose
    t = np.int32(u) * np.int32(KYBER_Q)
    t = a - t
    t >>= 16
    return np.int16(t)


montgomery_reduce_outrange = (-KYBER_Q + 1, KYBER_Q - 1)

# /*************************************************
# * Name:        barrett_reduce
# *
# * Description: Barrett reduction; given a 16-bit integer a, computes
# *              centered representative congruent to a mod q in {-(q-1)/2,...,(q-1)/2}
# *
# * Arguments:   - int16_t a: input integer to be reduced
# *
# * Returns:     integer in {-(q-1)/2,...,(q-1)/2} congruent to a modulo q.
# **************************************************/
def barrett_reduce(a: np.int16) -> np.int16:
    t = np.int16((np.int32(BARRET_V) * np.int32(a) + (1 << 25)) >> 26)
    t = np.int16(np.int32(t) * np.int32(KYBER_Q))
    return np.int16(np.int32(a) - np.int32(t))


barret_reduce_outrange = (-(KYBER_Q - 1) // 2, (KYBER_Q - 1) // 2)


# /*************************************************
# * Name:        csubq
# *
# * Description: Conditionallly subtract q
# *
# * Arguments:   - int16_t x: input integer
# *
# * Returns:     a - q if a >= q, else a
# **************************************************/
def csubq(a: np.int16) -> np.int16:
    a -= KYBER_Q
    a += (a >> 15) & KYBER_Q
    return a
