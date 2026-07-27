pub use mxx_primitives::utils::*;

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
use std::{
    fs,
    future::Future,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};
use tracing::debug;

/// Persist bytes through a same-directory temporary file and atomically publish them.
///
/// A process interrupted before `rename` leaves no partially written checkpoint at
/// `path`, so existence remains a valid completion test for resumable jobs.
pub fn write_bytes_atomic(path: &Path, bytes: &[u8]) {
    let file_name = path.file_name().and_then(|name| name.to_str()).unwrap_or_else(|| {
        panic!("atomic output path must have a UTF-8 file name: {}", path.display())
    });
    let temporary_path = path.with_file_name(format!(".{file_name}.tmp"));
    let mut file = fs::File::create(&temporary_path).unwrap_or_else(|err| {
        panic!("failed to create atomic temporary file {}: {err}", temporary_path.display())
    });
    file.write_all(bytes).unwrap_or_else(|err| {
        panic!("failed to write atomic temporary file {}: {err}", temporary_path.display())
    });
    file.sync_all().unwrap_or_else(|err| {
        panic!("failed to sync atomic temporary file {}: {err}", temporary_path.display())
    });
    drop(file);
    fs::rename(&temporary_path, path).unwrap_or_else(|err| {
        panic!(
            "failed to publish atomic file {} as {}: {err}",
            temporary_path.display(),
            path.display()
        )
    });
    let parent = path.parent().unwrap_or_else(|| {
        panic!("atomic output path must have a parent directory: {}", path.display())
    });
    fs::File::open(parent)
        .unwrap_or_else(|err| {
            panic!("failed to open atomic output directory {}: {err}", parent.display())
        })
        .sync_all()
        .unwrap_or_else(|err| {
            panic!("failed to sync atomic output directory {}: {err}", parent.display())
        });
}

#[cfg(any(test, feature = "test-support"))]
thread_local! {
    static PRE_DELETE_FILE_LENGTHS:
        std::cell::RefCell<Option<Vec<(std::path::PathBuf, u64)>>> =
        const { std::cell::RefCell::new(None) };
}

#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn start_pre_delete_file_length_observer() {
    PRE_DELETE_FILE_LENGTHS.with(|observations| {
        *observations.borrow_mut() = Some(Vec::new());
    });
}

#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn observe_file_before_delete(path: &Path) {
    PRE_DELETE_FILE_LENGTHS.with(|observations| {
        let mut observations = observations.borrow_mut();
        if let Some(observations) = observations.as_mut() {
            let bytes = fs::metadata(path)
                .unwrap_or_else(|err| {
                    panic!("failed to stat observed deletion {}: {err}", path.display())
                })
                .len();
            observations.push((path.to_path_buf(), bytes));
        }
    });
}

#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn take_pre_delete_file_lengths() -> Vec<(std::path::PathBuf, u64)> {
    PRE_DELETE_FILE_LENGTHS
        .with(|observations| observations.borrow_mut().take().unwrap_or_default())
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
    let reveal_plaintexts = vec![true; input_size];
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
    debug!("{}", format!("{label} loaded in {elapsed:?}"));
    res
}

/// Async variant of `timed_read` that awaits the provided future-producing closure.
pub async fn timed_read_async<T, F, Fut>(label: &str, f: F, total: &mut Duration) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    let start = Instant::now();
    let res = f().await;
    let elapsed = start.elapsed();
    *total += elapsed;
    debug!("{}", format!("{label} loaded in {elapsed:?}"));
    res
}
