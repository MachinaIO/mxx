use crate::{
    bgg::{
        encoding::BggEncoding,
        sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    },
    matrix::base::BaseMatrix,
    poly::{
        Poly,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyUniformSampler, hash::DCRTPolyHashSampler, uniform::DCRTPolyUniformSampler,
    },
};
use keccak_asm::Keccak256;
use memory_stats::memory_stats;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::Rng;
use std::{
    env,
    future::Future,
    time::{Duration, Instant},
};
use tracing::{debug, info};

/// ideal thread chunk size for parallel
pub fn chunk_size_for(original: usize) -> usize {
    match original {
        0..=2048 => 128,
        2049..=4096 => 256,
        4097..=8192 => 512,
        _ => 1024,
    }
}

pub fn block_size() -> usize {
    env::var("BLOCK_SIZE").map(|str| str.parse::<usize>().unwrap()).unwrap_or(100)
}

#[macro_export]
macro_rules! parallel_iter {
    ($i: expr) => {{ rayon::iter::IntoParallelIterator::into_par_iter($i) }};
}

/// Implements $tr for all combinations of T and &T by delegating to the &T/&T implementation.
#[macro_export]
macro_rules! impl_binop_with_refs {
    ($T:ty => $tr:ident::$f:ident $($t:tt)*) => {
        impl $tr<$T> for $T {
            type Output = $T;

            #[inline]
            fn $f(self, rhs: $T) -> Self::Output {
                <&$T as $tr<&$T>>::$f(&self, &rhs)
            }
        }

        impl $tr<&$T> for $T {
            type Output = $T;

            #[inline]
            fn $f(self, rhs: &$T) -> Self::Output {
                <&$T as $tr<&$T>>::$f(&self, rhs)
            }
        }

        impl $tr<$T> for &$T {
            type Output = $T;

            #[inline]
            fn $f(self, rhs: $T) -> Self::Output {
                <&$T as $tr<&$T>>::$f(self, &rhs)
            }
        }

        impl $tr<&$T> for &$T {
            type Output = $T;

            #[inline]
            fn $f $($t)*
        }
    };
}

pub fn debug_mem<T: Into<String>>(tag: T) {
    if let Some(usage) = memory_stats() {
        debug!(
            "{} || Current physical/virtual memory usage: {} | {}",
            tag.into(),
            usage.physical_mem,
            usage.virtual_mem
        );
    } else {
        debug!("Couldn't get the current memory usage :(");
    }
}

pub fn log_mem<T: Into<String>>(tag: T) {
    if let Some(usage) = memory_stats() {
        info!(
            "{} || Current physical/virtual memory usage: {} | {}",
            tag.into(),
            usage.physical_mem,
            usage.virtual_mem,
        );
    } else {
        info!("Couldn't get the current memory usage :(");
    }
}

// Helper function to create a random polynomial using UniformSampler
pub fn create_random_poly(params: &DCRTPolyParams) -> DCRTPoly {
    let sampler = DCRTPolyUniformSampler::new();
    sampler.sample_poly(params, &DistType::FinRingDist)
}

pub fn create_bit_random_poly(params: &DCRTPolyParams) -> DCRTPoly {
    let sampler = DCRTPolyUniformSampler::new();
    sampler.sample_poly(params, &DistType::BitDist)
}

pub fn create_ternary_random_poly(params: &DCRTPolyParams) -> DCRTPoly {
    let sampler = DCRTPolyUniformSampler::new();
    sampler.sample_poly(params, &DistType::TernaryDist)
}

// Helper function to create a bit polynomial (0 or 1)
pub fn create_bit_poly(params: &DCRTPolyParams, bit: bool) -> DCRTPoly {
    if bit { DCRTPoly::const_one(params) } else { DCRTPoly::const_zero(params) }
}

pub fn random_bgg_encodings(
    input_size: usize,
    secret_size: usize,
    params: &DCRTPolyParams,
) -> Vec<BggEncoding<BaseMatrix<DCRTPoly>>> {
    // Create samplers
    let key: [u8; 32] = rand::random();
    let bgg_pubkey_sampler =
        BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, secret_size);

    // Generate random tag for sampling
    let tag: u64 = rand::random();
    let tag_bytes = tag.to_le_bytes();
    // Create secret and plaintexts
    let secrets = vec![create_bit_random_poly(params); secret_size];
    let plaintexts = vec![create_random_poly(params); input_size];

    // Create random public keys
    let reveal_plaintexts = vec![true; input_size + 1];
    let bgg_encoding_sampler =
        BGGEncodingSampler::<DCRTPolyUniformSampler>::new(params, &secrets, None);
    let pubkeys = bgg_pubkey_sampler.sample(params, &tag_bytes, &reveal_plaintexts);
    bgg_encoding_sampler.sample(params, &pubkeys, &plaintexts)
}

pub fn timed_read<T, F: FnOnce() -> T>(label: &str, f: F, total: &mut Duration) -> T {
    let start = Instant::now();
    let res = f();
    let elapsed = start.elapsed();
    *total += elapsed;
    crate::utils::log_mem(format!("{label} loaded in {elapsed:?}"));
    res
}

/// Async variant of `timed_read` that awaits the provided future-producing closure.
pub async fn timed_read_async<T, F, Fut>(
    label: &str,
    f: F,
    total: &mut Duration,
) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    let start = Instant::now();
    let res = f().await;
    let elapsed = start.elapsed();
    *total += elapsed;
    crate::utils::log_mem(format!("{label} loaded in {elapsed:?}"));
    res
}

/// Calculate modular inverse using the extended Euclidean algorithm
/// Exposed for reuse by other CRT components.
pub fn mod_inverse(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    if m == &BigUint::one() {
        return Some(BigUint::ZERO);
    }

    let (mut old_r, mut r) = (a.clone(), m.clone());
    let (mut old_s, mut s) = (BigUint::one(), BigUint::ZERO);

    while !r.is_zero() {
        let quotient = &old_r / &r;
        let temp_r = &old_r - &quotient * &r;
        old_r = std::mem::replace(&mut r, temp_r);
        let temp_s = if &quotient * &s <= old_s {
            &old_s - &quotient * &s
        } else {
            // Handle underflow by adding m
            m - ((&quotient * &s - &old_s) % m)
        };
        old_s = std::mem::replace(&mut s, temp_s);
    }

    if old_r == BigUint::one() {
        Some(old_s % m)
    } else {
        None // No modular inverse exists
    }
}

pub fn gen_biguint_for_modulus<R: Rng>(
    rng: &mut R,
    limb_bit_size: usize,
    modulus: &BigUint,
) -> BigUint {
    if limb_bit_size == 0 || modulus.is_zero() {
        return BigUint::ZERO;
    }
    let max_bits = modulus.bits() as usize;
    let max_bytes = max_bits.div_ceil(8);
    if max_bytes == 0 {
        return BigUint::ZERO;
    }
    let mut bytes = vec![0u8; max_bytes];
    rng.fill_bytes(&mut bytes);
    BigUint::from_bytes_be(&bytes) % modulus
}
