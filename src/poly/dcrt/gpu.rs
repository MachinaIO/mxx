use crate::{
    element::{PolyElem, finite_ring::FinRingElem},
    impl_binop_with_refs,
    poly::{Poly, PolyParams},
    utils::{chunk_size_for, log_mem, mod_inverse},
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use rayon::prelude::*;
#[cfg(test)]
use sequential_test::sequential;
use std::{
    ffi::CStr,
    fmt::Debug,
    hash::Hash,
    mem,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    os::raw::{c_char, c_int},
    ptr::{self, NonNull},
    slice,
    sync::Arc,
    time::Instant,
};

#[allow(non_camel_case_types)]
#[repr(C)]
struct GpuContextOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
struct GpuPolyOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
struct GpuEventSetOpaque {
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

    fn gpu_poly_create(
        ctx: *mut GpuContextOpaque,
        level: c_int,
        out_poly: *mut *mut GpuPolyOpaque,
    ) -> c_int;
    fn gpu_poly_destroy(poly: *mut GpuPolyOpaque);
    fn gpu_poly_clone(src: *const GpuPolyOpaque, out_poly: *mut *mut GpuPolyOpaque) -> c_int;

    fn gpu_poly_load_rns(
        poly: *mut GpuPolyOpaque,
        rns_flat: *const u64,
        rns_len: usize,
        format: c_int,
    ) -> c_int;
    fn gpu_poly_store_rns(
        poly: *mut GpuPolyOpaque,
        rns_flat_out: *mut u64,
        rns_len: usize,
        format: c_int,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    fn gpu_poly_load_rns_batch(
        polys: *const *mut GpuPolyOpaque,
        poly_count: usize,
        bytes: *const u8,
        bytes_per_poly: usize,
        format: c_int,
    ) -> c_int;
    fn gpu_poly_store_rns_batch(
        polys: *const *mut GpuPolyOpaque,
        poly_count: usize,
        bytes_out: *mut u8,
        bytes_per_poly: usize,
        format: c_int,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    fn gpu_event_set_wait(events: *mut GpuEventSetOpaque) -> c_int;
    fn gpu_event_set_destroy(events: *mut GpuEventSetOpaque);

    fn gpu_poly_add(
        out: *mut GpuPolyOpaque,
        a: *const GpuPolyOpaque,
        b: *const GpuPolyOpaque,
    ) -> c_int;
    fn gpu_poly_sub(
        out: *mut GpuPolyOpaque,
        a: *const GpuPolyOpaque,
        b: *const GpuPolyOpaque,
    ) -> c_int;
    fn gpu_poly_mul(
        out: *mut GpuPolyOpaque,
        a: *const GpuPolyOpaque,
        b: *const GpuPolyOpaque,
    ) -> c_int;
    fn gpu_block_add(
        out: *const *mut GpuPolyOpaque,
        lhs: *const *const GpuPolyOpaque,
        rhs: *const *const GpuPolyOpaque,
        count: usize,
    ) -> c_int;
    fn gpu_block_sub(
        out: *const *mut GpuPolyOpaque,
        lhs: *const *const GpuPolyOpaque,
        rhs: *const *const GpuPolyOpaque,
        count: usize,
    ) -> c_int;
    fn gpu_block_entrywise_mul(
        out: *const *mut GpuPolyOpaque,
        lhs: *const *const GpuPolyOpaque,
        rhs: *const *const GpuPolyOpaque,
        count: usize,
    ) -> c_int;

    fn gpu_poly_ntt(poly: *mut GpuPolyOpaque, batch: c_int) -> c_int;
    fn gpu_poly_intt(poly: *mut GpuPolyOpaque, batch: c_int) -> c_int;
    fn gpu_device_synchronize() -> c_int;

    fn gpu_last_error() -> *const c_char;

    fn gpu_pinned_alloc(bytes: usize) -> *mut u8;
    fn gpu_pinned_free(ptr: *mut u8);
}

pub(crate) const GPU_POLY_FORMAT_COEFF: c_int = 0;
pub(crate) const GPU_POLY_FORMAT_EVAL: c_int = 1;

fn last_error_string() -> String {
    unsafe {
        let ptr = gpu_last_error();
        if ptr.is_null() {
            return "unknown GPU error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn check_status(code: c_int, context: &str) {
    if code != 0 {
        panic!("{context} failed: {}", last_error_string());
    }
}

pub(crate) fn gpu_device_sync() {
    let status = unsafe { gpu_device_synchronize() };
    check_status(status, "gpu_device_synchronize");
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

    pub(crate) fn set_slice(&mut self, slice: &[T]) {
        if slice.len() > self.cap {
            self.reserve(slice.len());
        }
        if !slice.is_empty() {
            unsafe {
                ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.as_ptr(), slice.len());
            }
        }
        self.len = slice.len();
    }

    pub(crate) fn reserve(&mut self, capacity: usize) {
        if capacity <= self.cap {
            return;
        }
        let new_ptr = pinned_alloc::<T>(capacity);
        if self.len > 0 {
            unsafe {
                ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }
        }
        if self.cap > 0 {
            unsafe {
                gpu_pinned_free(self.ptr.as_ptr() as *mut u8);
            }
        }
        self.ptr = new_ptr;
        self.cap = capacity;
    }
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

fn log2_u32(value: u32) -> u32 {
    assert!(value.is_power_of_two(), "ring_dimension must be a power of 2");
    value.trailing_zeros()
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
}

impl GpuDCRTPolyParams {
    pub fn new(ring_dimension: u32, moduli: Vec<u64>, base_bits: u32) -> Self {
        Self::new_with_gpu(ring_dimension, moduli, base_bits, Vec::new(), None, 1)
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

    pub fn batch(&self) -> u32 {
        self.batch
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
        log_mem(&format!(
            "Creating GPU context with log_n={}, moduli={:?}, gpu_ids={:?}, dnum={}, batch={}",
            log_n, moduli, gpu_ids, dnum, batch
        ));
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
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            #[cfg(test)]
            gpu_device_sync();
            unsafe { gpu_context_destroy(self.raw) };
            self.raw = ptr::null_mut();
            log_mem("GPU context destroyed");
        }
    }
}

pub struct GpuDCRTPoly {
    params: Arc<GpuDCRTPolyParams>,
    raw: *mut GpuPolyOpaque,
    level: usize,
    is_ntt: bool,
}

impl Debug for GpuDCRTPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDCRTPoly")
            .field("level", &self.level)
            .field("is_ntt", &self.is_ntt)
            .field("coeffs", &self.coeffs())
            .finish()
    }
}

/// # Safety
/// GpuDCRTPoly is an opaque handle to GPU memory managed on the C++ side.
unsafe impl Send for GpuDCRTPoly {}
unsafe impl Sync for GpuDCRTPoly {}

impl GpuDCRTPoly {
    pub(crate) fn new_empty(params: Arc<GpuDCRTPolyParams>, level: usize, is_ntt: bool) -> Self {
        let mut poly_ptr: *mut GpuPolyOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_poly_create(
                params.ctx.raw,
                level as c_int,
                &mut poly_ptr as *mut *mut GpuPolyOpaque,
            )
        };
        check_status(status, "gpu_poly_create");
        Self { params, raw: poly_ptr, level, is_ntt }
    }

    fn from_flat(
        params: Arc<GpuDCRTPolyParams>,
        level: usize,
        flat: Vec<u64>,
        is_ntt: bool,
    ) -> Self {
        let poly = Self::new_empty(params, level, is_ntt);
        let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        let status = unsafe { gpu_poly_load_rns(poly.raw, flat.as_ptr(), flat.len(), format) };
        check_status(status, "gpu_poly_load_rns");
        poly
    }

    fn store_rns_flat(&self) -> Vec<u64> {
        let n = self.params.ring_dimension as usize;
        let len = (self.level + 1) * n;
        let mut flat = vec![0u64; len];
        let mut events: *mut GpuEventSetOpaque = ptr::null_mut();
        let format = if self.is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        let status = unsafe {
            gpu_poly_store_rns(self.raw, flat.as_mut_ptr(), flat.len(), format, &mut events)
        };
        check_status(status, "gpu_poly_store_rns");
        let wait_status = unsafe { gpu_event_set_wait(events) };
        unsafe { gpu_event_set_destroy(events) };
        check_status(wait_status, "gpu_event_set_wait");
        flat
    }

    pub(crate) fn load_rns_bytes(&mut self, bytes: &[u8], format: c_int) {
        if bytes.is_empty() {
            return;
        }
        let ptrs = [self.raw];
        let status = unsafe {
            gpu_poly_load_rns_batch(ptrs.as_ptr(), 1, bytes.as_ptr(), bytes.len(), format)
        };
        check_status(status, "gpu_poly_load_rns_batch");
        self.is_ntt = format == GPU_POLY_FORMAT_EVAL;
    }

    pub(crate) fn store_rns_bytes(&mut self, bytes_out: &mut [u8], format: c_int) {
        if bytes_out.is_empty() {
            return;
        }
        let ptrs = [self.raw];
        let mut events: *mut GpuEventSetOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_poly_store_rns_batch(
                ptrs.as_ptr(),
                1,
                bytes_out.as_mut_ptr(),
                bytes_out.len(),
                format,
                &mut events,
            )
        };
        check_status(status, "gpu_poly_store_rns_batch");
        let wait_status = unsafe { gpu_event_set_wait(events) };
        unsafe { gpu_event_set_destroy(events) };
        check_status(wait_status, "gpu_event_set_wait");
        self.is_ntt = format == GPU_POLY_FORMAT_EVAL;
    }

    pub(crate) fn load_rns_bytes_batch(
        polys: &mut [GpuDCRTPoly],
        bytes: &[u8],
        bytes_per_poly: usize,
        format: c_int,
    ) {
        if polys.is_empty() || bytes.is_empty() {
            return;
        }
        let ptrs = polys.iter_mut().map(|poly| poly.raw).collect::<Vec<_>>();
        let status = unsafe {
            gpu_poly_load_rns_batch(
                ptrs.as_ptr(),
                polys.len(),
                bytes.as_ptr(),
                bytes_per_poly,
                format,
            )
        };
        check_status(status, "gpu_poly_load_rns_batch");
        let is_ntt = format == GPU_POLY_FORMAT_EVAL;
        for poly in polys.iter_mut() {
            poly.is_ntt = is_ntt;
        }
    }

    pub(crate) fn store_rns_bytes_batch(
        polys: &mut [GpuDCRTPoly],
        bytes_out: &mut [u8],
        bytes_per_poly: usize,
        format: c_int,
    ) {
        if polys.is_empty() || bytes_out.is_empty() {
            return;
        }
        let ptrs = polys.iter_mut().map(|poly| poly.raw).collect::<Vec<_>>();
        let mut events: *mut GpuEventSetOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_poly_store_rns_batch(
                ptrs.as_ptr(),
                polys.len(),
                bytes_out.as_mut_ptr(),
                bytes_per_poly,
                format,
                &mut events,
            )
        };
        check_status(status, "gpu_poly_store_rns_batch");
        let wait_status = unsafe { gpu_event_set_wait(events) };
        unsafe { gpu_event_set_destroy(events) };
        check_status(wait_status, "gpu_event_set_wait");
        let is_ntt = format == GPU_POLY_FORMAT_EVAL;
        for poly in polys.iter_mut() {
            poly.is_ntt = is_ntt;
        }
    }

    pub(crate) fn ensure_coeff_domain(&self) -> Self {
        if !self.is_ntt {
            return self.clone();
        }
        let mut tmp = self.clone();
        let status = unsafe { gpu_poly_intt(tmp.raw, self.params.batch() as c_int) };
        check_status(status, "gpu_poly_intt");
        tmp.is_ntt = false;
        tmp
    }

    pub(crate) fn is_ntt(&self) -> bool {
        self.is_ntt
    }

    pub(crate) fn ntt_in_place(&mut self) {
        if self.is_ntt {
            return;
        }
        let status = unsafe { gpu_poly_ntt(self.raw, self.params.batch() as c_int) };
        check_status(status, "gpu_poly_ntt");
        self.is_ntt = true;
    }

    fn assert_compatible(&self, other: &Self) {
        assert_eq!(self.level, other.level, "GPU polynomials must have the same level");
        assert_eq!(self.params.as_ref(), other.params.as_ref(), "GPU params must match");
        assert_eq!(self.is_ntt, other.is_ntt, "GPU polynomials must share the same domain");
    }

    pub(crate) fn load_from_compact_bytes(&mut self, bytes: &[u8]) {
        let ring_dimension = self.params.ring_dimension() as usize;
        if ring_dimension == 0 {
            return;
        }
        let max_byte_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let bit_vector_byte_size = ring_dimension.div_ceil(8);
        let bit_vector = &bytes[4..4 + bit_vector_byte_size];
        let coeffs_base_offset = 4 + bit_vector_byte_size;
        let modulus = self.params.modulus();
        let coeffs: Vec<FinRingElem> = reconstruct_coeffs_chunked(
            bytes,
            ring_dimension,
            max_byte_size,
            bit_vector,
            coeffs_base_offset,
            &modulus,
            chunk_size_for(ring_dimension),
        );

        let level = self.level;
        let moduli = self.params.moduli();
        let modulus_bigints =
            moduli[..=level].iter().map(|m| BigUint::from(*m)).collect::<Vec<_>>();
        let residues_by_limb = modulus_bigints
            .iter()
            .map(|modulus| {
                coeffs
                    .par_iter()
                    .map(|coeff| (coeff.value() % modulus).to_u64().unwrap_or(0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let flat = residues_by_limb.into_iter().flatten().collect::<Vec<_>>();

        let status = unsafe {
            gpu_poly_load_rns(self.raw, flat.as_ptr(), flat.len(), GPU_POLY_FORMAT_COEFF)
        };
        check_status(status, "gpu_poly_load_rns");
        self.is_ntt = false;
    }

    pub(crate) fn add_into(out: &mut GpuDCRTPoly, a: &GpuDCRTPoly, b: &GpuDCRTPoly) {
        a.assert_compatible(b);
        assert_eq!(out.level, a.level, "GPU polynomials must have the same level");
        assert_eq!(out.params.as_ref(), a.params.as_ref(), "GPU params must match");
        let status = unsafe { gpu_poly_add(out.raw, a.raw, b.raw) };
        check_status(status, "gpu_poly_add");
        out.is_ntt = a.is_ntt;
    }

    pub(crate) fn sub_into(out: &mut GpuDCRTPoly, a: &GpuDCRTPoly, b: &GpuDCRTPoly) {
        a.assert_compatible(b);
        assert_eq!(out.level, a.level, "GPU polynomials must have the same level");
        assert_eq!(out.params.as_ref(), a.params.as_ref(), "GPU params must match");
        let status = unsafe { gpu_poly_sub(out.raw, a.raw, b.raw) };
        check_status(status, "gpu_poly_sub");
        out.is_ntt = a.is_ntt;
    }

    pub(crate) fn mul_into(out: &mut GpuDCRTPoly, a: &GpuDCRTPoly, b: &GpuDCRTPoly) {
        a.assert_compatible(b);
        assert_eq!(out.level, a.level, "GPU polynomials must have the same level");
        assert_eq!(out.params.as_ref(), a.params.as_ref(), "GPU params must match");
        let status = unsafe { gpu_poly_mul(out.raw, a.raw, b.raw) };
        check_status(status, "gpu_poly_mul");
        out.is_ntt = true;
    }

    fn block_op_into(
        out: &mut [GpuDCRTPoly],
        a: &[GpuDCRTPoly],
        b: &[GpuDCRTPoly],
        op: unsafe extern "C" fn(
            *const *mut GpuPolyOpaque,
            *const *const GpuPolyOpaque,
            *const *const GpuPolyOpaque,
            usize,
        ) -> c_int,
        context: &str,
    ) {
        assert_eq!(out.len(), a.len(), "GPU block op requires equal lengths");
        assert_eq!(out.len(), b.len(), "GPU block op requires equal lengths");
        if out.is_empty() {
            return;
        }

        debug_assert!(
            a.iter().zip(b.iter()).all(|(lhs, rhs)| {
                lhs.level == rhs.level &&
                    lhs.params.as_ref() == rhs.params.as_ref() &&
                    lhs.is_ntt == rhs.is_ntt
            }),
            "GPU block op requires compatible inputs"
        );

        let out_ptrs = out.iter_mut().map(|poly| poly.raw).collect::<Vec<_>>();
        let lhs_ptrs = a.iter().map(|poly| poly.raw as *const GpuPolyOpaque).collect::<Vec<_>>();
        let rhs_ptrs = b.iter().map(|poly| poly.raw as *const GpuPolyOpaque).collect::<Vec<_>>();

        let status =
            unsafe { op(out_ptrs.as_ptr(), lhs_ptrs.as_ptr(), rhs_ptrs.as_ptr(), out.len()) };
        check_status(status, context);

        let is_ntt = a[0].is_ntt;
        let level = a[0].level;
        for poly in out.iter_mut() {
            poly.is_ntt = is_ntt;
            poly.level = level;
        }
    }

    pub(crate) fn block_add_into(out: &mut [GpuDCRTPoly], a: &[GpuDCRTPoly], b: &[GpuDCRTPoly]) {
        Self::block_op_into(out, a, b, gpu_block_add, "gpu_block_add");
    }

    pub(crate) fn block_sub_into(out: &mut [GpuDCRTPoly], a: &[GpuDCRTPoly], b: &[GpuDCRTPoly]) {
        Self::block_op_into(out, a, b, gpu_block_sub, "gpu_block_sub");
    }

    pub(crate) fn block_mul_into(out: &mut [GpuDCRTPoly], a: &[GpuDCRTPoly], b: &[GpuDCRTPoly]) {
        Self::block_op_into(out, a, b, gpu_block_entrywise_mul, "gpu_block_entrywise_mul");
    }

    pub(crate) fn block_mul_into_refs(
        out: &mut [GpuDCRTPoly],
        a: &[&GpuDCRTPoly],
        b: &[&GpuDCRTPoly],
    ) {
        assert_eq!(out.len(), a.len(), "GPU block op requires equal lengths");
        assert_eq!(out.len(), b.len(), "GPU block op requires equal lengths");
        if out.is_empty() {
            return;
        }

        debug_assert!(
            a.iter().zip(b.iter()).all(|(lhs, rhs)| {
                lhs.level == rhs.level &&
                    lhs.params.as_ref() == rhs.params.as_ref() &&
                    lhs.is_ntt == rhs.is_ntt
            }),
            "GPU block op requires compatible inputs"
        );

        let out_ptrs = out.iter_mut().map(|poly| poly.raw).collect::<Vec<_>>();
        let lhs_ptrs = a.iter().map(|poly| poly.raw as *const GpuPolyOpaque).collect::<Vec<_>>();
        let rhs_ptrs = b.iter().map(|poly| poly.raw as *const GpuPolyOpaque).collect::<Vec<_>>();

        let status = unsafe {
            gpu_block_entrywise_mul(
                out_ptrs.as_ptr(),
                lhs_ptrs.as_ptr(),
                rhs_ptrs.as_ptr(),
                out.len(),
            )
        };
        check_status(status, "gpu_block_entrywise_mul");

        let is_ntt = a[0].is_ntt;
        let level = a[0].level;
        for poly in out.iter_mut() {
            poly.is_ntt = is_ntt;
            poly.level = level;
        }
    }

    pub(crate) fn block_add_assign(out: &mut [GpuDCRTPoly], rhs: &[GpuDCRTPoly]) {
        assert_eq!(out.len(), rhs.len(), "GPU block op requires equal lengths");
        if out.is_empty() {
            return;
        }

        debug_assert!(
            out.iter().zip(rhs.iter()).all(|(lhs, rhs)| {
                lhs.level == rhs.level &&
                    lhs.params.as_ref() == rhs.params.as_ref() &&
                    lhs.is_ntt == rhs.is_ntt
            }),
            "GPU block op requires compatible inputs"
        );

        let out_ptrs = out.iter_mut().map(|poly| poly.raw).collect::<Vec<_>>();
        let lhs_ptrs = out.iter().map(|poly| poly.raw as *const GpuPolyOpaque).collect::<Vec<_>>();
        let rhs_ptrs = rhs.iter().map(|poly| poly.raw as *const GpuPolyOpaque).collect::<Vec<_>>();

        let status = unsafe {
            gpu_block_add(out_ptrs.as_ptr(), lhs_ptrs.as_ptr(), rhs_ptrs.as_ptr(), out.len())
        };
        check_status(status, "gpu_block_add");
    }

    pub(crate) fn add_in_place(&mut self, rhs: &GpuDCRTPoly) {
        self.assert_compatible(rhs);
        let status = unsafe { gpu_poly_add(self.raw, self.raw, rhs.raw) };
        check_status(status, "gpu_poly_add");
    }

    // pub(crate) fn sub_in_place(&mut self, rhs: &GpuDCRTPoly) {
    //     self.assert_compatible(rhs);
    //     let status = unsafe { gpu_poly_sub(self.raw, self.raw, rhs.raw) };
    //     check_status(status, "gpu_poly_sub");
    // }

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
        let mut poly_ptr: *mut GpuPolyOpaque = ptr::null_mut();
        let status = unsafe { gpu_poly_clone(self.raw, &mut poly_ptr as *mut *mut GpuPolyOpaque) };
        check_status(status, "gpu_poly_clone");
        gpu_device_sync();
        Self { params: self.params.clone(), raw: poly_ptr, level: self.level, is_ntt: self.is_ntt }
    }
}

impl Drop for GpuDCRTPoly {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // #[cfg(test)]
            // gpu_device_sync();
            unsafe { gpu_poly_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

impl PartialEq for GpuDCRTPoly {
    fn eq(&self, other: &Self) -> bool {
        if self.params.as_ref() != other.params.as_ref() {
            return false;
        }
        self.coeffs() == other.coeffs()
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

    fn from_u64_vecs(params: &Self::Params, coeffs: &[Vec<u64>]) -> Self {
        let n = params.ring_dimension as usize;
        assert_eq!(coeffs.len(), n, "coeffs length must match ring dimension");
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
        let ring_dimension = params.ring_dimension as usize;
        let modulus = params.modulus();

        let max_byte_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let bit_vector_byte_size = ring_dimension.div_ceil(8);
        let bit_vector = &bytes[4..4 + bit_vector_byte_size];
        let coeffs_base_offset = 4 + bit_vector_byte_size;

        let coeffs: Vec<FinRingElem> = reconstruct_coeffs_chunked(
            bytes,
            ring_dimension,
            max_byte_size,
            bit_vector,
            coeffs_base_offset,
            &modulus,
            chunk_size_for(ring_dimension),
        );

        Self::from_coeffs(params, &coeffs)
    }

    fn coeffs(&self) -> Vec<Self::Elem> {
        let poly = self.ensure_coeff_domain();
        let n = poly.params.ring_dimension as usize;
        let level = poly.level;
        let flat_time = Instant::now();
        let flat = poly.store_rns_flat();
        let modulus = poly.params.modulus();
        let reconstruct_coeffs = poly.params.reconstruct_coeffs_for_level(level);
        let modulus_level = Arc::new(poly.params.modulus_for_level(level));

        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let mut acc = BigUint::zero();
            for limb in 0..=level {
                let residue = flat[limb * n + i];
                acc += &reconstruct_coeffs[limb] * BigUint::from(residue);
            }
            acc %= &*modulus_level;
            coeffs.push(FinRingElem::new(acc, modulus.clone()));
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
        let n = params.ring_dimension as usize;
        let base_bits = params.base_bits() as usize;
        let digits_per_tower = params.crt_bits().div_ceil(base_bits);
        let num_digits = digits_per_tower * params.crt_depth();
        let base_mask = (1u64 << base_bits) - 1u64;

        let poly = self.ensure_coeff_domain();
        let flat = poly.store_rns_flat();

        (0..num_digits)
            .into_par_iter()
            .map(|digit_idx| {
                let tower = digit_idx / digits_per_tower;
                let shift = (digit_idx % digits_per_tower) * base_bits;

                let mut values = vec![vec![0u64; params.crt_depth()]; n];
                for i in 0..n {
                    let residue = flat[tower * n + i];
                    let digit = (residue >> shift) & base_mask;
                    values[i][tower] = digit;
                }

                GpuDCRTPoly::from_u64_vecs(params, &values)
            })
            .collect()
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
        let coeffs = self.coeffs();
        let ring_dimension = coeffs.len();
        let processed_coeffs: Vec<(bool, Vec<u8>)> =
            process_coeffs_chunked(&coeffs, chunk_size_for(ring_dimension));

        build_compact_bytes(processed_coeffs, ring_dimension)
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
    let out = GpuDCRTPoly::new_empty(self.params.clone(), self.level, self.is_ntt);
    let status = unsafe { gpu_poly_add(out.raw, self.raw, rhs.raw) };
    check_status(status, "gpu_poly_add");
    out
});

impl_binop_with_refs!(GpuDCRTPoly => Sub::sub(self, rhs: &GpuDCRTPoly) -> GpuDCRTPoly {
    self.assert_compatible(rhs);
    let out = GpuDCRTPoly::new_empty(self.params.clone(), self.level, self.is_ntt);
    let status = unsafe { gpu_poly_sub(out.raw, self.raw, rhs.raw) };
    check_status(status, "gpu_poly_sub");
    out
});

impl_binop_with_refs!(GpuDCRTPoly => Mul::mul(self, rhs: &GpuDCRTPoly) -> GpuDCRTPoly {
    self.assert_compatible(rhs);
    let mut out = GpuDCRTPoly::new_empty(self.params.clone(), self.level, false);
    let status = unsafe { gpu_poly_mul(out.raw, self.raw, rhs.raw) };
    check_status(status, "gpu_poly_mul");
    out.is_ntt = true;
    out
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
        let zero = GpuDCRTPoly::const_zero(&self.params);
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

// ==== Compact bytes helpers ====

fn build_compact_bytes(processed_coeffs: Vec<(bool, Vec<u8>)>, ring_dimension: usize) -> Vec<u8> {
    let bit_vector_byte_size = ring_dimension.div_ceil(8);
    let mut bit_vector = vec![0u8; bit_vector_byte_size];
    let mut max_byte_size = 1;

    for (i, (is_negative, value_bytes)) in processed_coeffs.iter().enumerate() {
        if *is_negative {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bit_vector[byte_idx] |= 1 << bit_idx;
        }
        max_byte_size = std::cmp::max(max_byte_size, value_bytes.len());
    }

    let total_byte_size = 4 + bit_vector_byte_size + (ring_dimension * max_byte_size);
    let mut result = vec![0u8; total_byte_size];

    result[0..4].copy_from_slice(&(max_byte_size as u32).to_le_bytes());
    result[4..4 + bit_vector_byte_size].copy_from_slice(&bit_vector);

    let coeffs_base_offset = 4 + bit_vector_byte_size;
    for (i, (_, value_bytes)) in processed_coeffs.iter().enumerate() {
        if !value_bytes.is_empty() {
            let start_pos = coeffs_base_offset + (i * max_byte_size);
            result[start_pos..start_pos + value_bytes.len()].copy_from_slice(value_bytes);
        }
    }

    result
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

        let ring_dimension = gpu_params.ring_dimension() as usize;

        let max_byte_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert_eq!(max_byte_size, 1, "Max byte size size should be 1 for BitDist");

        let bit_vector_byte_size = ring_dimension.div_ceil(8);

        let expected_total_size = 4 + bit_vector_byte_size + (ring_dimension * max_byte_size);
        assert_eq!(bytes.len(), expected_total_size, "Incorrect total byte size");

        let bit_vector = &bytes[4..4 + bit_vector_byte_size];
        assert_eq!(bit_vector.len(), bit_vector_byte_size, "Bit vector size is incorrect");

        let coeffs_section = &bytes[4 + bit_vector_byte_size..];
        assert_eq!(
            coeffs_section.len(),
            ring_dimension * max_byte_size,
            "Coefficient section size is incorrect"
        );

        for (i, &coeff_byte) in coeffs_section.iter().enumerate() {
            assert!(
                coeff_byte == 0 || coeff_byte == 1,
                "Coefficient at position {} should be 0 or 1, got {}",
                i,
                coeff_byte
            );
        }
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
        let modulus = gpu_params.modulus();
        let modulus_byte_size = modulus.to_bytes_le().len();

        let ring_dimension = gpu_params.ring_dimension() as usize;

        let max_byte_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert!(
            max_byte_size <= modulus_byte_size,
            "Max byte size should be less than or equal to modulus byte size"
        );

        let bit_vector_byte_size = ring_dimension.div_ceil(8);

        let expected_total_size = 4 + bit_vector_byte_size + (ring_dimension * max_byte_size);
        assert_eq!(bytes.len(), expected_total_size, "Incorrect total byte size");

        let bit_vector = &bytes[4..4 + bit_vector_byte_size];
        assert_eq!(bit_vector.len(), bit_vector_byte_size, "Bit vector size is incorrect");

        let coeffs_section = &bytes[4 + bit_vector_byte_size..];
        assert_eq!(
            coeffs_section.len(),
            ring_dimension * max_byte_size,
            "Coefficient section size is incorrect"
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

fn reconstruct_coeffs_chunked(
    bytes: &[u8],
    ring_dimension: usize,
    max_byte_size: usize,
    bit_vector: &[u8],
    coeffs_base_offset: usize,
    modulus: &Arc<BigUint>,
    chunk_size: usize,
) -> Vec<FinRingElem> {
    (0..ring_dimension)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            chunk
                .into_iter()
                .map(|i| {
                    reconstruct_single_coeff(
                        i,
                        bytes,
                        max_byte_size,
                        bit_vector,
                        coeffs_base_offset,
                        modulus,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[inline(always)]
fn reconstruct_single_coeff(
    i: usize,
    bytes: &[u8],
    max_byte_size: usize,
    bit_vector: &[u8],
    coeffs_base_offset: usize,
    modulus: &Arc<BigUint>,
) -> FinRingElem {
    let start = coeffs_base_offset + (i * max_byte_size);

    let mut value_len = max_byte_size;
    let value_bytes = &bytes[start..start + max_byte_size];
    while value_len > 0 && value_bytes[value_len - 1] == 0 {
        value_len -= 1;
    }

    let value = if value_len == 0 {
        BigUint::ZERO
    } else {
        BigUint::from_bytes_le(&value_bytes[..value_len])
    };

    let byte_idx = i / 8;
    let bit_idx = i % 8;
    let is_negative = (bit_vector[byte_idx] & (1 << bit_idx)) != 0;

    let final_value =
        if is_negative && !value.is_zero() { modulus.as_ref() - &value } else { value };

    FinRingElem::new(final_value, modulus.clone())
}

fn process_coeffs_chunked(coeffs: &[FinRingElem], chunk_size: usize) -> Vec<(bool, Vec<u8>)> {
    if coeffs.is_empty() {
        return Vec::new();
    }

    let modulus_arc = coeffs[0].modulus().clone();
    let modulus_ref = modulus_arc.as_ref();
    let q_half = modulus_ref >> 1;

    coeffs
        .par_chunks(chunk_size)
        .flat_map(|chunk| {
            chunk
                .iter()
                .map(|coeff| process_single_coeff_with(coeff, modulus_ref, &q_half))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[inline(always)]
fn process_single_coeff_with(
    coeff: &FinRingElem,
    modulus: &BigUint,
    q_half: &BigUint,
) -> (bool, Vec<u8>) {
    let coeff_val = coeff.value();

    if coeff_val > q_half {
        let centered_value = modulus - coeff_val;
        let value_bytes =
            if centered_value.is_zero() { Vec::new() } else { centered_value.to_bytes_le() };
        (true, value_bytes)
    } else if coeff_val.is_zero() {
        (false, Vec::new())
    } else {
        (false, coeff_val.to_bytes_le())
    }
}
