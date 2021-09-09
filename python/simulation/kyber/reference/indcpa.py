from .polyvec import (
    polyvec_getnoise_eta1,
    polyvec_ntt,
    polyvec,
    polyvec_basemul_acc_montgomery,
)
from .poly import poly

# /*************************************************
# * Name:        indcpa_keypair
# *
# * Description: Generates public and private key for the CPA-secure
# *              public-key encryption scheme underlying Kyber
# *
# * Arguments:   - uint8_t *pk: pointer to output public key
# *                             (of length KYBER_INDCPA_PUBLICKEYBYTES bytes)
# *              - uint8_t *sk: pointer to output private key
#                               (of length KYBER_INDCPA_SECRETKEYBYTES bytes)
# **************************************************/
def indcpa_keypair(rng, height, vec_k):

    # unsigned int i
    # uint8_t buf[2*KYBER_SYMBYTES]
    # const uint8_t *publicseed = buf
    # const uint8_t *noiseseed = buf+KYBER_SYMBYTES
    # uint8_t nonce = 0
    # polyvec a[KYBER_K], e, pkpv, skpv

    # randombytes(buf, KYBER_SYMBYTES)
    # hash_g(buf, buf, KYBER_SYMBYTES)

    # gen_a(a, publicseed)

    # for(i=0;i<KYBER_K;i++)
    #     poly_getnoise_eta1(&skpv.vec[i], noiseseed, nonce++)
    skpv = polyvec_getnoise_eta1(rng, height, vec_k)

    # for(i=0;i<KYBER_K;i++)
    #     poly_getnoise_eta1(&e.vec[i], noiseseed, nonce++)

    # polyvec_ntt(&skpv)
    skpv_hat = polyvec_ntt(skpv.copy())
    # polyvec_ntt(&e)

    # // matrix-vector multiplication
    # for(i=0;i<KYBER_K;i++) {
    #     polyvec_basemul_acc_montgomery(&pkpv.vec[i], &a[i], &skpv)
    #     poly_tomont(&pkpv.vec[i])
    # }

    # polyvec_add(&pkpv, &pkpv, &e)
    # polyvec_reduce(&pkpv)

    # pack_sk(sk, &skpv)
    # pack_pk(pk, &pkpv, publicseed)
    return skpv, skpv_hat


# /*************************************************
# * Name:        indcpa_dec
# *
# * Description: Decryption function of the CPA-secure
# *              public-key encryption scheme underlying Kyber.
# *
# * Arguments:   - uint8_t *m: pointer to output decrypted message
# *                            (of length KYBER_INDCPA_MSGBYTES)
# *              - const uint8_t *c: pointer to input ciphertext
# *                                  (of length KYBER_INDCPA_BYTES)
# *              - const uint8_t *sk: pointer to input secret key
# *                                   (of length KYBER_INDCPA_SECRETKEYBYTES)
# **************************************************/
def indcpa_dec(b_hat: polyvec, skpv: polyvec) -> poly:

    # polyvec b, skpv
    # poly v, mp

    # unpack_ciphertext(&b, &v, c)
    # unpack_sk(&skpv, sk)

    # polyvec_ntt(&b)
    mp = polyvec_basemul_acc_montgomery(skpv, b_hat)
    return mp
    # poly_invntt_tomont(&mp)  # NOTE: attack here in invntt on coeffs

    # poly_sub(&mp, &v, &mp)
    # poly_reduce(&mp)

    # poly_tomsg(m, &mp)
