use crate::{
    element::{PolyElem, finite_ring::FinRingElem},
    impl_binop_with_refs,
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
#[cfg(test)]
use sequential_test::sequential;
use std::{
    collections::HashMap,
    ffi::CStr,
    fmt::Debug,
    hash::Hash,
    mem,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    os::raw::{c_char, c_int},
    ptr::{self, NonNull},
    slice,
    sync::{Arc, Mutex, OnceLock, Weak},
};
use tracing::info;

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuContextOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuMatrixOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuEventSetOpaque {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn gpu_context_create(
        log_n: u32,
        l: u32,
        dnum: u32,
        moduli: *const u64,
        moduli_len: usize,
        gpu_ids: *const c_int,
        gpu_ids_len: usize,
        batch: u32,
        out_ctx: *mut *mut GpuContextOpaque,
    ) -> c_int;
    fn gpu_context_destroy(ctx: *mut GpuContextOpaque);
    fn gpu_context_get_N(ctx: *const GpuContextOpaque, out_n: *mut c_int) -> c_int;

    pub(crate) fn gpu_event_set_wait(events: *mut GpuEventSetOpaque) -> c_int;
    pub(crate) fn gpu_event_set_destroy(events: *mut GpuEventSetOpaque);

    pub(crate) fn gpu_matrix_create(
        ctx: *mut GpuContextOpaque,
        level: c_int,
        rows: usize,
        cols: usize,
        format: c_int,
        out_mat: *mut *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_destroy(mat: *mut GpuMatrixOpaque);
    pub(crate) fn gpu_matrix_copy(dst: *mut GpuMatrixOpaque, src: *const GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_load_rns_batch(
        mat: *mut GpuMatrixOpaque,
        bytes: *const u8,
        bytes_per_poly: usize,
        format: c_int,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_store_rns_batch(
        mat: *const GpuMatrixOpaque,
        bytes_out: *mut u8,
        bytes_per_poly: usize,
        format: c_int,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_store_compact_bytes(
        mat: *mut GpuMatrixOpaque,
        payload_out: *mut u8,
        payload_capacity: usize,
        out_max_coeff_bits: *mut u16,
        out_bytes_per_coeff: *mut u16,
        out_payload_len: *mut usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_load_compact_bytes(
        mat: *mut GpuMatrixOpaque,
        payload: *const u8,
        payload_len: usize,
        max_coeff_bits: u16,
    ) -> c_int;
    pub(crate) fn gpu_matrix_add(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sub(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_equal(
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
        out_equal: *mut c_int,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_scalar(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        scalar: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_copy_block(
        out: *mut GpuMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        src_row: usize,
        src_col: usize,
        rows: usize,
        cols: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_fill_gadget(out: *mut GpuMatrixOpaque, base_bits: u32) -> c_int;
    pub(crate) fn gpu_matrix_fill_small_gadget(out: *mut GpuMatrixOpaque, base_bits: u32) -> c_int;
    pub(crate) fn gpu_matrix_fill_small_decomposed_identity_chunk(
        out: *mut GpuMatrixOpaque,
        scalar_by_digit: *const GpuMatrixOpaque,
        chunk_idx: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_decompose_base(
        src: *const GpuMatrixOpaque,
        base_bits: u32,
        out: *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_decompose_base_small(
        src: *const GpuMatrixOpaque,
        base_bits: u32,
        out: *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_gauss_samp_gq_arb_base(
        src: *mut GpuMatrixOpaque,
        base_bits: u32,
        c: f64,
        dgg_stddev: f64,
        seed: u64,
        out: *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_p1_full(
        a_mat: *const GpuMatrixOpaque,
        b_mat: *const GpuMatrixOpaque,
        d_mat: *const GpuMatrixOpaque,
        tp2: *const GpuMatrixOpaque,
        sigma: f64,
        s: f64,
        dgg_stddev: f64,
        seed: u64,
        out: *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_distribution(
        out: *mut GpuMatrixOpaque,
        dist_type: c_int,
        sigma: f64,
        seed: u64,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_distribution_decompose_base(
        out: *mut GpuMatrixOpaque,
        dist_type: c_int,
        sigma: f64,
        seed: u64,
        base_bits: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_distribution_decompose_base_small(
        out: *mut GpuMatrixOpaque,
        dist_type: c_int,
        sigma: f64,
        seed: u64,
        base_bits: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_ntt_all(mat: *mut GpuMatrixOpaque, batch: c_int) -> c_int;
    pub(crate) fn gpu_matrix_intt_all(mat: *mut GpuMatrixOpaque, batch: c_int) -> c_int;
    fn gpu_device_synchronize() -> c_int;
    fn gpu_device_count(out_count: *mut c_int) -> c_int;
    // fn gpu_device_mem_info(device: c_int, out_free: *mut usize, out_total: *mut usize) -> c_int;

    fn gpu_last_error() -> *const c_char;

    fn gpu_pinned_alloc(bytes: usize) -> *mut u8;
    fn gpu_pinned_free(ptr: *mut u8);
}

pub(crate) const GPU_POLY_FORMAT_COEFF: c_int = 0;
pub(crate) const GPU_POLY_FORMAT_EVAL: c_int = 1;
pub(crate) const GPU_MATRIX_DIST_UNIFORM: c_int = 0;
pub(crate) const GPU_MATRIX_DIST_GAUSS: c_int = 1;
pub(crate) const GPU_MATRIX_DIST_BIT: c_int = 2;
pub(crate) const GPU_MATRIX_DIST_TERNARY: c_int = 3;

pub(crate) fn last_error_string() -> String {
    unsafe {
        let ptr = gpu_last_error();
        if ptr.is_null() {
            return "unknown GPU error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

pub(crate) fn check_status(code: c_int, context: &str) {
    if code != 0 {
        panic!("{context} failed: {}", last_error_string());
    }
}

#[doc(hidden)]
pub fn gpu_device_sync() {
    let status = unsafe { gpu_device_synchronize() };
    check_status(status, "gpu_device_synchronize");
}

// #[derive(Clone, Copy, Debug)]
// pub(crate) struct GpuMemoryInfo {
//     pub device: c_int,
//     pub free: usize,
//     pub total: usize,
// }

// pub(crate) fn gpu_memory_infos() -> Result<Vec<GpuMemoryInfo>, String> {
//     let mut count: c_int = 0;
//     let status = unsafe { gpu_device_count(&mut count) };
//     if status != 0 {
//         return Err(last_error_string());
//     }
//     if count < 0 {
//         return Err("invalid GPU device count".to_string());
//     }
//     if count == 0 {
//         return Ok(Vec::new());
//     }

//     let mut infos = Vec::with_capacity(count as usize);
//     for device in 0..count {
//         let mut free: usize = 0;
//         let mut total: usize = 0;
//         let status = unsafe { gpu_device_mem_info(device, &mut free, &mut total) };
//         if status != 0 {
//             return Err(last_error_string());
//         }
//         infos.push(GpuMemoryInfo { device, free, total });
//     }
//     Ok(infos)
// }

fn available_gpu_ids() -> Vec<i32> {
    let mut count: c_int = 0;
    let status = unsafe { gpu_device_count(&mut count) };
    if status != 0 || count <= 0 {
        return Vec::new();
    }
    (0..count).map(|idx| idx as i32).collect()
}

#[cfg(feature = "gpu")]
pub fn detected_gpu_device_count() -> usize {
    available_gpu_ids().len()
}

#[cfg(feature = "gpu")]
pub fn detected_gpu_device_ids() -> Vec<i32> {
    available_gpu_ids()
}

fn pinned_alloc<T>(len: usize) -> NonNull<T> {
    if len == 0 {
        return NonNull::dangling();
    }
    let bytes = len.checked_mul(mem::size_of::<T>()).expect("pinned buffer size overflow");
    let ptr = unsafe { gpu_pinned_alloc(bytes) } as *mut T;
    if ptr.is_null() {
        panic!("gpu_pinned_alloc failed: {}", last_error_string());
    }
    NonNull::new(ptr).expect("gpu_pinned_alloc returned null")
}

pub struct PinnedHostBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

unsafe impl<T: Send> Send for PinnedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedHostBuffer<T> {}

impl<T> PinnedHostBuffer<T> {
    pub(crate) fn new() -> Self {
        Self { ptr: NonNull::dangling(), len: 0, cap: 0 }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    // pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
    //     if self.len == 0 {
    //         &mut []
    //     } else {
    //         unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    //     }
    // }

    // pub(crate) fn len(&self) -> usize {
    //     self.len
    // }
}

impl<T: Copy> PinnedHostBuffer<T> {
    pub(crate) fn from_slice(slice: &[T]) -> Self {
        if slice.is_empty() {
            return Self::new();
        }
        let ptr = pinned_alloc::<T>(slice.len());
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), ptr.as_ptr(), slice.len());
        }
        Self { ptr, len: slice.len(), cap: slice.len() }
    }

    // pub(crate) fn to_vec(&self) -> Vec<T> {
    //     self.as_slice().to_vec()
    // }
}

impl<T: Copy> Clone for PinnedHostBuffer<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl<T: PartialEq> PartialEq for PinnedHostBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for PinnedHostBuffer<T> {}

impl<T> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        unsafe {
            gpu_pinned_free(self.ptr.as_ptr() as *mut u8);
        }
    }
}

fn bits_in_u64(value: u64) -> usize {
    (u64::BITS - value.leading_zeros()) as usize
}

#[inline(always)]
fn log2_u32(value: u32) -> u32 {
    assert!(value.is_power_of_two(), "ring_dimension must be a power of 2");
    value.trailing_zeros()
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DeviceContextCacheKey {
    ring_dimension: u32,
    moduli: Vec<u64>,
    base_bits: u32,
    device_id: i32,
    batch: u32,
}

fn single_device_context_cache() -> &'static Mutex<HashMap<DeviceContextCacheKey, Weak<GpuContext>>>
{
    static CACHE: OnceLock<Mutex<HashMap<DeviceContextCacheKey, Weak<GpuContext>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
pub struct GpuDCRTPolyParams {
    ring_dimension: u32,
    moduli: Vec<u64>,
    crt_bits: usize,
    crt_depth: usize,
    modulus: Arc<BigUint>,
    base_bits: u32,
    gpu_ids: Vec<i32>,
    dnum: u32,
    batch: u32,
    ctx: Arc<GpuContext>,
}

impl Debug for GpuDCRTPolyParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDCRTPolyParams")
            .field("ring_dimension", &self.ring_dimension)
            .field("crt_depth", &self.crt_depth)
            .field("crt_bits", &self.crt_bits)
            .field("base_bits", &self.base_bits)
            .field("gpu_ids", &self.gpu_ids)
            .field("dnum", &self.dnum)
            .field("batch", &self.batch)
            .finish()
    }
}

impl PartialEq for GpuDCRTPolyParams {
    fn eq(&self, other: &Self) -> bool {
        self.ring_dimension == other.ring_dimension &&
            self.moduli == other.moduli &&
            self.base_bits == other.base_bits &&
            self.gpu_ids == other.gpu_ids &&
            self.dnum == other.dnum &&
            self.batch == other.batch
    }
}

impl Eq for GpuDCRTPolyParams {}

impl PolyParams for GpuDCRTPolyParams {
    type Modulus = Arc<BigUint>;

    fn ring_dimension(&self) -> u32 {
        self.ring_dimension
    }

    fn modulus(&self) -> Self::Modulus {
        self.modulus.clone()
    }

    fn base_bits(&self) -> u32 {
        self.base_bits
    }

    fn modulus_bits(&self) -> usize {
        self.modulus.bits() as usize
    }

    fn modulus_digits(&self) -> usize {
        self.crt_bits.div_ceil(self.base_bits as usize) * self.crt_depth
    }

    fn to_crt(&self) -> (Vec<u64>, usize, usize) {
        (self.moduli.clone(), self.crt_bits, self.crt_depth)
    }

    fn device_ids(&self) -> Vec<i32> {
        self.gpu_ids.clone()
    }

    fn params_for_device(&self, device_id: i32) -> Self {
        let ctx = self.single_device_context(device_id);
        Self {
            ring_dimension: self.ring_dimension,
            moduli: self.moduli.clone(),
            crt_bits: self.crt_bits,
            crt_depth: self.crt_depth,
            modulus: self.modulus.clone(),
            base_bits: self.base_bits,
            gpu_ids: vec![device_id],
            dnum: 1,
            batch: self.batch,
            ctx,
        }
    }
}

impl GpuDCRTPolyParams {
    fn single_device_context(&self, device_id: i32) -> Arc<GpuContext> {
        let key = DeviceContextCacheKey {
            ring_dimension: self.ring_dimension,
            moduli: self.moduli.clone(),
            base_bits: self.base_bits,
            device_id,
            batch: self.batch,
        };

        if let Some(existing) = {
            let cache = single_device_context_cache();
            let guard = cache.lock().expect("single_device_context_cache mutex poisoned");
            guard.get(&key).and_then(Weak::upgrade)
        } {
            return existing;
        }

        let log_n = log2_u32(self.ring_dimension);
        let created =
            Arc::new(GpuContext::create(log_n, &self.moduli, &[device_id], 1, self.batch));

        let cache = single_device_context_cache();
        let mut guard = cache.lock().expect("single_device_context_cache mutex poisoned");
        if let Some(existing) = guard.get(&key).and_then(Weak::upgrade) {
            return existing;
        }
        guard.insert(key, Arc::downgrade(&created));
        created
    }

    pub fn new(ring_dimension: u32, moduli: Vec<u64>, base_bits: u32) -> Self {
        let gpu_ids = available_gpu_ids();
        // Default params stay single-device so low-level matrix/poly ops keep the
        // invariant that all limbs of a matrix live on one device.
        let default_gpu_ids = gpu_ids.into_iter().take(1).collect::<Vec<_>>();
        Self::new_with_gpu(ring_dimension, moduli, base_bits, default_gpu_ids, None, 1)
    }

    pub fn new_with_gpu(
        ring_dimension: u32,
        moduli: Vec<u64>,
        base_bits: u32,
        gpu_ids: Vec<i32>,
        dnum: Option<u32>,
        batch: u32,
    ) -> Self {
        assert!(!moduli.is_empty(), "moduli must not be empty");
        let crt_depth = moduli.len();
        let crt_bits = moduli.iter().map(|m| bits_in_u64(*m)).max().unwrap_or(0);
        let modulus = moduli.iter().fold(BigUint::one(), |acc, m| acc * m);
        let dnum =
            dnum.unwrap_or_else(|| if gpu_ids.is_empty() { 1 } else { gpu_ids.len() as u32 });
        let log_n = log2_u32(ring_dimension);
        let ctx = Arc::new(GpuContext::create(log_n, &moduli, &gpu_ids, dnum, batch));

        Self {
            ring_dimension,
            moduli,
            crt_bits,
            crt_depth,
            modulus: Arc::new(modulus),
            base_bits,
            gpu_ids,
            dnum,
            batch,
            ctx,
        }
    }

    pub fn crt_depth(&self) -> usize {
        self.crt_depth
    }

    pub fn crt_bits(&self) -> usize {
        self.crt_bits
    }

    pub fn moduli(&self) -> &[u64] {
        &self.moduli
    }

    pub fn gpu_ids(&self) -> &[i32] {
        &self.gpu_ids
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub(crate) fn ctx_raw(&self) -> *mut GpuContextOpaque {
        self.ctx.raw_ptr()
    }

    pub(crate) fn modulus_for_level(&self, level: usize) -> BigUint {
        self.moduli.iter().take(level + 1).fold(BigUint::one(), |acc, m| acc * m)
    }

    pub(crate) fn reconstruct_coeffs_for_level(&self, level: usize) -> Vec<BigUint> {
        let modulus = self.modulus_for_level(level);
        (0..=level)
            .map(|idx| {
                let qi = BigUint::from(self.moduli[idx]);
                let q_over_qi = &modulus / &qi;
                let q_over_qi_mod = &q_over_qi % &qi;
                let inv = mod_inverse(
                    q_over_qi_mod.to_u64().expect("CRT residue must fit in u64"),
                    self.moduli[idx],
                )
                .expect("CRT moduli must be coprime");
                (q_over_qi * BigUint::from(inv)) % &modulus
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct GpuContext {
    raw: *mut GpuContextOpaque,
    pub n: usize,
    pub moduli: Vec<u64>,
    pub gpu_ids: Vec<i32>,
    pub dnum: u32,
    pub batch: u32,
}

/// # Safety
/// GpuContext is an opaque handle to a GPU context managed on the C++ side.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    fn create(log_n: u32, moduli: &[u64], gpu_ids: &[i32], dnum: u32, batch: u32) -> Self {
        info!(
            "{}",
            format!(
                "Creating GPU context with log_n={}, moduli={:?}, gpu_ids={:?}, dnum={}, batch={}",
                log_n, moduli, gpu_ids, dnum, batch
            )
        );
        let l = moduli.len().saturating_sub(1) as u32;
        let mut ctx_ptr: *mut GpuContextOpaque = ptr::null_mut();
        let (gpu_ids_ptr, gpu_ids_len) = if gpu_ids.is_empty() {
            (ptr::null(), 0usize)
        } else {
            (gpu_ids.as_ptr(), gpu_ids.len())
        };
        let status = unsafe {
            gpu_context_create(
                log_n,
                l,
                dnum,
                moduli.as_ptr(),
                moduli.len(),
                gpu_ids_ptr,
                gpu_ids_len,
                batch,
                &mut ctx_ptr as *mut *mut GpuContextOpaque,
            )
        };
        check_status(status, "gpu_context_create");

        let mut n_out = 0i32;
        let status = unsafe { gpu_context_get_N(ctx_ptr, &mut n_out as *mut c_int) };
        check_status(status, "gpu_context_get_N");
        let n = if n_out > 0 { n_out as usize } else { 1usize << log_n };

        Self { raw: ctx_ptr, n, moduli: moduli.to_vec(), gpu_ids: gpu_ids.to_vec(), dnum, batch }
    }

    pub(crate) fn raw_ptr(&self) -> *mut GpuContextOpaque {
        self.raw
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            #[cfg(test)]
            gpu_device_sync();
            unsafe { gpu_context_destroy(self.raw) };
            self.raw = ptr::null_mut();
            info!("GPU context destroyed");
        }
    }
}

pub struct GpuDCRTPoly {
    inner: GpuDCRTPolyMatrix,
}

impl Debug for GpuDCRTPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDCRTPoly")
            .field("level", &self.level())
            .field("is_ntt", &self.is_ntt())
            .field("coeffs", &self.coeffs())
            .finish()
    }
}

/// # Safety
/// GpuDCRTPoly is an opaque handle to GPU memory managed on the C++ side.
unsafe impl Send for GpuDCRTPoly {}
unsafe impl Sync for GpuDCRTPoly {}

impl GpuDCRTPoly {
    pub(crate) fn from_inner(inner: GpuDCRTPolyMatrix) -> Self {
        inner.assert_singleton();
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &GpuDCRTPolyMatrix {
        &self.inner
    }

    pub(crate) fn params_ref(&self) -> &GpuDCRTPolyParams {
        &self.inner.params
    }

    pub(crate) fn level(&self) -> usize {
        self.inner.level()
    }

    fn from_flat(
        params: Arc<GpuDCRTPolyParams>,
        level: usize,
        flat: Vec<u64>,
        is_ntt: bool,
    ) -> Self {
        let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        let bytes_len = flat.len().saturating_mul(mem::size_of::<u64>());
        let bytes = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, bytes_len) };
        let mut mat = GpuDCRTPolyMatrix::new_empty_with_state(params.as_ref(), 1, 1, level, is_ntt);
        mat.load_rns_bytes(bytes, bytes_len, format);
        Self::from_inner(mat)
    }

    fn from_u64_vecs(params: &GpuDCRTPolyParams, coeffs: &[Vec<u64>]) -> Self {
        let n = params.ring_dimension as usize;
        assert!(
            coeffs.len() <= n,
            "coeffs length must be <= ring dimension (got {}, expected <= {})",
            coeffs.len(),
            n
        );
        let mut coeffs_buf;
        let coeffs = if coeffs.len() == n {
            coeffs
        } else {
            coeffs_buf = Vec::with_capacity(n);
            coeffs_buf.extend(coeffs.iter().cloned());
            coeffs_buf.resize_with(n, Vec::new);
            &coeffs_buf
        };
        let num_limbs = coeffs.iter().map(|v| v.len()).max().unwrap_or(0).max(1);
        assert!(num_limbs <= params.crt_depth, "coeff limb count exceeds CRT depth");
        let level = num_limbs.saturating_sub(1);

        let mut flat = vec![0u64; num_limbs * n];
        for (i, coeff) in coeffs.iter().enumerate() {
            for limb in 0..num_limbs {
                let value = coeff.get(limb).copied().unwrap_or(0);
                flat[limb * n + i] = value;
            }
        }

        Self::from_flat(Arc::new(params.clone()), level, flat, false)
    }

    pub(crate) fn store_rns_bytes(&mut self, bytes_out: &mut [u8], format: c_int) {
        if bytes_out.is_empty() {
            return;
        }
        self.inner.store_rns_bytes(bytes_out, bytes_out.len(), format);
    }

    pub(crate) fn ensure_coeff_domain(&self) -> Self {
        if !self.is_ntt() {
            return self.clone();
        }
        let mut tmp = self.clone();
        tmp.inner.singleton_intt_in_place(self.params_ref().batch() as c_int);
        tmp
    }

    pub(crate) fn ensure_eval_domain(&self) -> Self {
        if self.is_ntt() {
            return self.clone();
        }
        let mut tmp = self.clone();
        tmp.inner.singleton_ntt_in_place(self.params_ref().batch() as c_int);
        tmp
    }

    pub(crate) fn is_ntt(&self) -> bool {
        self.inner.is_ntt()
    }

    pub(crate) fn ntt_in_place(&mut self) {
        if self.is_ntt() {
            return;
        }
        self.inner.singleton_ntt_in_place(self.params_ref().batch() as c_int);
    }

    fn assert_compatible(&self, other: &Self) {
        assert_eq!(self.level(), other.level(), "GPU polynomials must have the same level");
        assert_eq!(self.params_ref(), other.params_ref(), "GPU params must match");
    }

    fn constant_with_value(params: &Arc<GpuDCRTPolyParams>, value: &BigUint) -> Self {
        let n = params.ring_dimension as usize;
        let q = params.modulus();
        let mut coeffs = vec![FinRingElem::zero(&q); n];
        if n > 0 {
            coeffs[0] = FinRingElem::new(value.clone(), q.clone());
        }
        Self::from_coeffs(params.as_ref(), &coeffs)
    }

    fn residues_from_biguints(params: &GpuDCRTPolyParams, coeffs: &[BigUint]) -> Vec<Vec<u64>> {
        let moduli = params.moduli();
        coeffs
            .par_iter()
            .map(|coeff| {
                moduli
                    .iter()
                    .map(|m| {
                        let modulus = BigUint::from(*m);
                        (coeff % modulus).to_u64().unwrap_or(0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl Clone for GpuDCRTPoly {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl PartialEq for GpuDCRTPoly {
    fn eq(&self, other: &Self) -> bool {
        if self.params_ref() != other.params_ref() || self.level() != other.level() {
            return false;
        }
        if std::ptr::eq(self, other) {
            return true;
        }
        if self.is_ntt() == other.is_ntt() {
            return self.inner == other.inner;
        }
        let lhs = self.ensure_coeff_domain();
        let rhs = other.ensure_coeff_domain();
        lhs.inner == rhs.inner
    }
}

impl Eq for GpuDCRTPoly {}

impl Hash for GpuDCRTPoly {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for coeff in self.coeffs() {
            coeff.value().hash(state);
        }
    }
}

impl Poly for GpuDCRTPoly {
    type Elem = FinRingElem;
    type Params = GpuDCRTPolyParams;

    fn from_bool_vec(params: &Self::Params, coeffs: &[bool]) -> Self {
        let coeffs = coeffs.iter().map(|&b| if b { 1u64 } else { 0u64 }).collect::<Vec<_>>();
        Self::from_u64_vecs(
            params,
            &coeffs.iter().map(|v| vec![*v; params.crt_depth()]).collect::<Vec<_>>(),
        )
    }

    fn from_coeffs(params: &Self::Params, coeffs: &[Self::Elem]) -> Self {
        let modulus = params.modulus();
        let residues = coeffs
            .par_iter()
            .map(|coeff| {
                debug_assert_eq!(coeff.modulus(), &modulus);
                params
                    .moduli()
                    .iter()
                    .map(|m| {
                        let modulus = BigUint::from(*m);
                        (coeff.value() % modulus).to_u64().unwrap_or(0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::from_u64_vecs(params, &residues)
    }

    fn from_u32s(params: &Self::Params, coeffs: &[u32]) -> Self {
        let residues = coeffs
            .iter()
            .map(|v| params.moduli().iter().map(|m| (*v as u64) % *m).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Self::from_u64_vecs(params, &residues)
    }

    fn from_biguints(params: &Self::Params, coeffs: &[BigUint]) -> Self {
        let residues = Self::residues_from_biguints(params, coeffs);
        Self::from_u64_vecs(params, &residues)
    }

    fn from_biguints_eval(params: &Self::Params, slots: &[BigUint]) -> Self {
        let mut poly = Self::from_biguints(params, slots);
        poly.ntt_in_place();
        poly
    }

    fn from_decomposed(params: &Self::Params, decomposed: &[Self]) -> Self {
        let mut reconstructed = Self::const_zero(params);
        for (i, bit_poly) in decomposed.iter().enumerate() {
            let power_of_two = BigUint::from(2u32).pow(i as u32);
            let const_poly_power_of_two = Self::from_biguint_to_constant(params, power_of_two);
            reconstructed += bit_poly * &const_poly_power_of_two;
        }
        reconstructed
    }

    fn from_compact_bytes(params: &Self::Params, bytes: &[u8]) -> Self {
        let mat = GpuDCRTPolyMatrix::from_compact_bytes(params, bytes);
        let (rows, cols) = mat.size();
        assert_eq!(rows, 1, "GpuDCRTPoly compact bytes must decode to 1x1 matrix");
        assert_eq!(cols, 1, "GpuDCRTPoly compact bytes must decode to 1x1 matrix");
        mat.entry(0, 0)
    }

    fn coeffs(&self) -> Vec<Self::Elem> {
        let mut poly = self.ensure_coeff_domain();
        let n = poly.params_ref().ring_dimension() as usize;
        let level = poly.level();
        let modulus = poly.params_ref().modulus();
        let modulus_level = Arc::new(poly.params_ref().modulus_for_level(level));
        let reconstruct_coeffs = Arc::new(poly.params_ref().reconstruct_coeffs_for_level(level));
        let expected_len = (level + 1).saturating_mul(n);
        let expected_bytes = expected_len.saturating_mul(mem::size_of::<u64>());
        let bytes_per_poly = (level + 1).saturating_mul(n).saturating_mul(mem::size_of::<u64>());
        let mut bytes = vec![0u8; bytes_per_poly];
        poly.store_rns_bytes(&mut bytes, GPU_POLY_FORMAT_COEFF);
        assert!(bytes.len() >= expected_bytes, "RNS byte length underflow in coeff extraction");
        let mut flat = Vec::with_capacity(expected_len);
        for limb_bytes in bytes[..expected_bytes].chunks_exact(mem::size_of::<u64>()) {
            let le: [u8; 8] = limb_bytes.try_into().expect("u64 chunk size mismatch");
            flat.push(u64::from_le_bytes(le));
        }
        debug_assert_eq!(flat.len(), expected_len, "RNS flat length mismatch");

        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let mut acc = BigUint::ZERO;
            for limb in 0..=level {
                let residue = flat[limb * n + i];
                acc += &reconstruct_coeffs[limb] * BigUint::from(residue);
            }
            let value = acc % modulus_level.as_ref();
            assert!(
                &value < modulus_level.as_ref(),
                "GPU reconstructed coefficient out of range at index {i}"
            );
            coeffs.push(FinRingElem::new(value, modulus.clone()));
        }
        coeffs
    }

    fn const_zero(params: &Self::Params) -> Self {
        Self::constant_with_value(&Arc::new(params.clone()), &BigUint::ZERO)
    }

    fn const_one(params: &Self::Params) -> Self {
        Self::constant_with_value(&Arc::new(params.clone()), &BigUint::one())
    }

    fn const_minus_one(params: &Self::Params) -> Self {
        let modulus = params.modulus();
        let value = modulus.as_ref() - BigUint::from(1u32);
        Self::constant_with_value(&Arc::new(params.clone()), &value)
    }

    fn const_max(params: &Self::Params) -> Self {
        let coeffs = vec![FinRingElem::max_q(&params.modulus()); params.ring_dimension as usize];
        Self::from_coeffs(params, &coeffs)
    }

    fn from_power_of_base_to_constant(params: &Self::Params, k: usize) -> Self {
        let base = 1u32 << params.base_bits();
        let value = BigUint::from(base).pow(k as u32);
        Self::from_biguint_to_constant(params, value)
    }

    fn from_elem_to_constant(params: &Self::Params, elem: &Self::Elem) -> Self {
        Self::from_biguint_to_constant(params, elem.value().clone())
    }

    fn from_biguint_to_constant(params: &Self::Params, int: BigUint) -> Self {
        Self::constant_with_value(&Arc::new(params.clone()), &int)
    }

    fn from_usize_to_constant(params: &Self::Params, int: usize) -> Self {
        Self::constant_with_value(&Arc::new(params.clone()), &BigUint::from(int as u64))
    }

    fn from_usize_to_lsb(params: &Self::Params, int: usize) -> Self {
        let n = params.ring_dimension as usize;
        if n <= usize::BITS as usize {
            debug_assert!(
                int < (1usize << n),
                "Input exceeds representable range for ring dimension"
            );
        }
        let q = params.modulus();
        let one = FinRingElem::one(&q);
        let zero = FinRingElem::zero(&q);

        let coeffs: Vec<FinRingElem> = (0..n)
            .map(|i| {
                if i < usize::BITS as usize && (int >> i) & 1 == 1 {
                    one.clone()
                } else {
                    zero.clone()
                }
            })
            .collect();

        Self::from_coeffs(params, &coeffs)
    }

    fn decompose_base(&self, params: &Self::Params) -> Vec<Self> {
        let num_digits = params.modulus_digits();
        if num_digits == 0 {
            return Vec::new();
        }
        let decomposed = self.inner().decompose();
        let (rows, cols) = decomposed.size();
        assert_eq!(cols, 1, "1x1 poly decomposition must keep single column");
        assert_eq!(rows, num_digits, "decomposition row count mismatch");
        (0..rows).map(|row| decomposed.entry(row, 0)).collect::<Vec<_>>()
    }

    fn extract_bits_with_threshold(&self, params: &Self::Params) -> Vec<bool> {
        let modulus = params.modulus();
        let half_q = FinRingElem::half_q(&modulus);
        let quarter_q = half_q.value() >> 1;
        let three_quarter_q = &quarter_q * 3u32;

        self.coeffs()
            .iter()
            .map(|coeff| coeff.value())
            .map(|coeff| coeff >= &quarter_q && coeff < &three_quarter_q)
            .collect()
    }

    fn to_bool_vec(&self) -> Vec<bool> {
        self.coeffs()
            .into_iter()
            .map(|c| {
                let v = c.value();
                if v == &BigUint::from(0u32) {
                    false
                } else if v == &BigUint::from(1u32) {
                    true
                } else {
                    panic!("Coefficient is not 0 or 1: {v}");
                }
            })
            .collect()
    }

    fn to_compact_bytes(&self) -> Vec<u8> {
        self.inner().to_compact_bytes()
    }

    fn to_const_int(&self) -> usize {
        let mut sum = 0usize;
        for (i, coeff) in self.coeffs().into_iter().enumerate() {
            if i >= usize::BITS as usize {
                break;
            }
            // Convert BigUint to usize safely, saturating if too large
            let coeff_val = coeff.value().try_into().expect("coeff_val is an invalid usize");
            sum = sum.saturating_add((1usize << i).saturating_mul(coeff_val));
        }
        sum
    }
}

impl_binop_with_refs!(GpuDCRTPoly => Add::add(self, rhs: &GpuDCRTPoly) -> GpuDCRTPoly {
    self.assert_compatible(rhs);
    let lhs = self.ensure_eval_domain();
    let rhs = rhs.ensure_eval_domain();
    GpuDCRTPoly::from_inner(&lhs.inner + &rhs.inner)
});

impl_binop_with_refs!(GpuDCRTPoly => Sub::sub(self, rhs: &GpuDCRTPoly) -> GpuDCRTPoly {
    self.assert_compatible(rhs);
    let lhs = self.ensure_eval_domain();
    let rhs = rhs.ensure_eval_domain();
    GpuDCRTPoly::from_inner(&lhs.inner - &rhs.inner)
});

impl_binop_with_refs!(GpuDCRTPoly => Mul::mul(self, rhs: &GpuDCRTPoly) -> GpuDCRTPoly {
    self.assert_compatible(rhs);
    let lhs = self.ensure_eval_domain();
    let rhs = rhs.ensure_eval_domain();
    GpuDCRTPoly::from_inner(&lhs.inner * &rhs.inner)
});

impl Neg for GpuDCRTPoly {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Neg for &GpuDCRTPoly {
    type Output = GpuDCRTPoly;

    fn neg(self) -> Self::Output {
        let zero = GpuDCRTPoly::const_zero(self.params_ref());
        &zero - self
    }
}

impl AddAssign for GpuDCRTPoly {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl AddAssign<&GpuDCRTPoly> for GpuDCRTPoly {
    fn add_assign(&mut self, rhs: &Self) {
        *self = &*self + rhs;
    }
}

impl SubAssign for GpuDCRTPoly {
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl SubAssign<&GpuDCRTPoly> for GpuDCRTPoly {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = &*self - rhs;
    }
}

impl MulAssign for GpuDCRTPoly {
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl MulAssign<&GpuDCRTPoly> for GpuDCRTPoly {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = &*self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };
    use rand::prelude::*;

    fn gpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 17, 1)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _crt_bits, _crt_depth) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    fn gpu_poly_from_cpu(poly: &DCRTPoly, gpu_params: &GpuDCRTPolyParams) -> GpuDCRTPoly {
        GpuDCRTPoly::from_coeffs(gpu_params, &poly.coeffs())
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_const_int_roundtrip() {
        gpu_device_sync();
        let mut rng = rand::rng();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let max_value = 1usize << 33;

        for _ in 0..10 {
            let value = rng.random_range(0..max_value);
            let lsb_poly = GpuDCRTPoly::from_usize_to_lsb(&gpu_params, value);
            let poly = GpuDCRTPoly::from_usize_to_constant(&gpu_params, value);
            let back = poly.to_const_int();
            let back_from_lsb = lsb_poly.to_const_int();
            assert_eq!(value, back);
            assert_eq!(value, back_from_lsb);
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_coeffs() {
        gpu_device_sync();
        let mut rng = rand::rng();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let q = gpu_params.modulus();
        let n = gpu_params.ring_dimension() as usize;
        let mut coeffs: Vec<FinRingElem> = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let value = rng.random_range(0..10000);
            coeffs.push(FinRingElem::new(value, q.clone()));
        }
        let poly = GpuDCRTPoly::from_coeffs(&gpu_params, &coeffs);
        let extracted_coeffs = poly.coeffs();
        assert_eq!(coeffs, extracted_coeffs);
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_arithmetic() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let q = gpu_params.modulus();
        let n = gpu_params.ring_dimension() as usize;

        let mut coeffs1 = vec![FinRingElem::zero(&q); n];
        let mut coeffs2 = vec![FinRingElem::zero(&q); n];
        coeffs1[0] = FinRingElem::new(100u32, q.clone());
        coeffs1[1] = FinRingElem::new(200u32, q.clone());
        coeffs1[2] = FinRingElem::new(300u32, q.clone());
        coeffs1[3] = FinRingElem::new(400u32, q.clone());
        coeffs2[0] = FinRingElem::new(500u32, q.clone());
        coeffs2[1] = FinRingElem::new(600u32, q.clone());
        coeffs2[2] = FinRingElem::new(700u32, q.clone());
        coeffs2[3] = FinRingElem::new(800u32, q.clone());

        let poly1 = GpuDCRTPoly::from_coeffs(&gpu_params, &coeffs1);
        let poly2 = GpuDCRTPoly::from_coeffs(&gpu_params, &coeffs2);

        let sum = poly1.clone() + poly2.clone();
        let product = &poly1 * &poly2;

        let neg_poly2 = poly2.clone().neg();
        let difference = poly1.clone() - poly2.clone();

        let mut poly_add_assign = poly1.clone();
        poly_add_assign += poly2.clone();

        let mut poly_mul_assign = poly1.clone();
        poly_mul_assign *= poly2.clone();

        assert!(sum != poly1, "Sum should differ from original poly1");
        assert!(neg_poly2 != poly2, "Negated polynomial should differ from original");
        assert_eq!(difference + poly2, poly1, "p1 - p2 + p2 should be p1");

        assert_eq!(poly_add_assign, sum, "+= result should match separate +");
        assert_eq!(poly_mul_assign, product, "*= result should match separate *");

        let const_poly = GpuDCRTPoly::from_usize_to_constant(&gpu_params, 123);
        let mut const_coeffs = vec![FinRingElem::zero(&q); n];
        const_coeffs[0] = FinRingElem::new(123, q.clone());
        assert_eq!(
            const_poly,
            GpuDCRTPoly::from_coeffs(&gpu_params, &const_coeffs),
            "from_const should produce a polynomial with constant term = 123"
        );
        let zero_poly = GpuDCRTPoly::const_zero(&gpu_params);
        assert_eq!(
            zero_poly,
            GpuDCRTPoly::from_coeffs(&gpu_params, &vec![FinRingElem::new(0, q.clone()); n]),
            "const_zero should produce a polynomial with all coeffs = 0"
        );

        let one_poly = GpuDCRTPoly::const_one(&gpu_params);
        let mut one_coeffs = vec![FinRingElem::zero(&q); n];
        one_coeffs[0] = FinRingElem::new(1, q);
        assert_eq!(
            one_poly,
            GpuDCRTPoly::from_coeffs(&gpu_params, &one_coeffs),
            "one_poly should produce a polynomial with constant term = 1"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_partial_eq_across_domains() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();
        let cpu_poly = sampler.sample_poly(&params, &DistType::FinRingDist);
        let coeff_poly = gpu_poly_from_cpu(&cpu_poly, &gpu_params);
        let mut eval_poly = coeff_poly.clone();
        eval_poly.ntt_in_place();
        assert_eq!(coeff_poly, eval_poly, "PartialEq should match across coeff/eval domains");
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_decompose() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();
        let cpu_poly = sampler.sample_poly(&params, &DistType::FinRingDist);
        let poly = gpu_poly_from_cpu(&cpu_poly, &gpu_params);
        let decomposed = poly.decompose_base(&gpu_params);
        assert_eq!(decomposed.len(), { gpu_params.modulus_digits() });
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_to_compact_bytes_bit_dist() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();
        let cpu_poly = sampler.sample_poly(&params, &DistType::BitDist);
        let poly = gpu_poly_from_cpu(&cpu_poly, &gpu_params);
        let bytes = poly.to_compact_bytes();
        assert!(!bytes.is_empty(), "compact serialization should not be empty");
        let reconstructed = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            reconstructed, poly,
            "compact roundtrip should preserve BitDist polynomial values"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_to_compact_bytes_uniform_dist() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();
        let cpu_poly = sampler.sample_poly(&params, &DistType::FinRingDist);
        let poly = gpu_poly_from_cpu(&cpu_poly, &gpu_params);
        let bytes = poly.to_compact_bytes();
        assert!(!bytes.is_empty(), "compact serialization should not be empty");
        let reconstructed = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            reconstructed, poly,
            "compact roundtrip should preserve uniform polynomial values"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_from_compact_bytes() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();

        let original_cpu_poly = sampler.sample_poly(&params, &DistType::BitDist);
        let original_poly = gpu_poly_from_cpu(&original_cpu_poly, &gpu_params);
        let bytes = original_poly.to_compact_bytes();
        let reconstructed_poly = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            original_poly, reconstructed_poly,
            "Reconstructed polynomial does not match original (BitDist)"
        );

        let original_cpu_poly = sampler.sample_poly(&params, &DistType::FinRingDist);
        let original_poly = gpu_poly_from_cpu(&original_cpu_poly, &gpu_params);
        let bytes = original_poly.to_compact_bytes();
        let reconstructed_poly = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            original_poly, reconstructed_poly,
            "Reconstructed polynomial does not match original (FinRingDist)"
        );

        let original_cpu_poly = sampler.sample_poly(&params, &DistType::GaussDist { sigma: 3.2 });
        let original_poly = gpu_poly_from_cpu(&original_cpu_poly, &gpu_params);
        let bytes = original_poly.to_compact_bytes();
        let reconstructed_poly = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            original_poly, reconstructed_poly,
            "Reconstructed polynomial does not match original (GaussDist)"
        );
    }
}
