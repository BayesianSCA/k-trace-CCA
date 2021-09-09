use crate::constants::{KYBER_Qi32, KYBER_Q};

const QINV_OLD: i32 = 62209;
const MONT: i16 = -1044; // 2^16 mod q
const QINV: i32 = -3327; // q^-1 mod 2^16
const BARRET_V: i16 = ((((1u32 << 26) as i32) + KYBER_Qi32 / 2) / KYBER_Qi32) as i16;

//Untestested!

pub fn montgomery_reduce_old(a: i32) -> i16 {
    let u = (a.wrapping_mul(QINV)) as i16;
    let mut t = (u as i32) * KYBER_Qi32;
    t = a - t;
    t >>= 16;
    t as i16
}

pub fn montgomery_reduce(a: i32) -> i16 {
    let mut t: i16 = a.wrapping_mul(QINV) as i16;
    t = ((a - (t as i32) * (KYBER_Q as i32)) >> 16) as i16;
    return t;
}

pub fn barrett_reduce(a: i16) -> i16 {
    let mut t: i16 = (((BARRET_V as i32) * (a as i32) + (1 << 25) as i32) >> 26) as i16;
    t = ((t as i32) * (KYBER_Q as i32)) as i16;
    (a as i32 - t as i32) as i16
}

pub fn csubq(mut a: i16) -> i16 {
    a -= KYBER_Q;
    a += (a >> 15) & KYBER_Q;
    a
}

pub fn fqmul(a: i16, b: i16) -> i16 {
    montgomery_reduce(a as i32 * b as i32)
}

//TODO: Tests
