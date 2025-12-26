use crate::{
    element::{PolyElem, finite_ring::FinRingElem},
    impl_binop_with_refs, parallel_iter,
    poly::{Poly, PolyParams},
    utils::{chunk_size_for, mod_inverse},
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use rayon::prelude::*;
use std::{
    ffi::CStr,
    fmt::Debug,
    hash::Hash,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    os::raw::{c_char, c_int},
    ptr,
    sync::Arc,
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

    fn gpu_poly_create(ctx: *mut GpuContextOpaque, level: c_int, out_poly: *mut *mut GpuPolyOpaque)
        -> c_int;
    fn gpu_poly_destroy(poly: *mut GpuPolyOpaque);
    fn gpu_poly_clone(src: *const GpuPolyOpaque, out_poly: *mut *mut GpuPolyOpaque) -> c_int;

    fn gpu_poly_load_rns(poly: *mut GpuPolyOpaque, coeffs_flat: *const u64, coeffs_len: usize)
        -> c_int;
    fn gpu_poly_store_rns(
        poly: *mut GpuPolyOpaque,
        coeffs_flat_out: *mut u64,
        coeffs_len: usize,
    ) -> c_int;

    fn gpu_poly_add(out: *mut GpuPolyOpaque, a: *const GpuPolyOpaque, b: *const GpuPolyOpaque)
        -> c_int;
    fn gpu_poly_sub(out: *mut GpuPolyOpaque, a: *const GpuPolyOpaque, b: *const GpuPolyOpaque)
        -> c_int;
    fn gpu_poly_mul(out: *mut GpuPolyOpaque, a: *const GpuPolyOpaque, b: *const GpuPolyOpaque)
        -> c_int;

    fn gpu_poly_ntt(poly: *mut GpuPolyOpaque, batch: c_int) -> c_int;
    fn gpu_poly_intt(poly: *mut GpuPolyOpaque, batch: c_int) -> c_int;

    fn gpu_last_error() -> *const c_char;
}

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
        let log_n = log2_u32(ring_dimension);
        let dnum = dnum.unwrap_or_else(|| if gpu_ids.is_empty() { 1 } else { gpu_ids.len() as u32 });
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

    fn modulus_for_level(&self, level: usize) -> BigUint {
        self.moduli
            .iter()
            .take(level + 1)
            .fold(BigUint::one(), |acc, m| acc * m)
    }

    fn reconstruct_coeffs_for_level(&self, level: usize) -> Vec<BigUint> {
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

        Self {
            raw: ctx_ptr,
            n,
            moduli: moduli.to_vec(),
            gpu_ids: gpu_ids.to_vec(),
            dnum,
            batch,
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { gpu_context_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

#[derive(Debug)]
pub struct GpuDCRTPoly {
    params: Arc<GpuDCRTPolyParams>,
    raw: *mut GpuPolyOpaque,
    level: usize,
    is_ntt: bool,
}

/// # Safety
/// GpuDCRTPoly is an opaque handle to GPU memory managed on the C++ side.
unsafe impl Send for GpuDCRTPoly {}
unsafe impl Sync for GpuDCRTPoly {}

impl GpuDCRTPoly {
    fn new_empty(params: Arc<GpuDCRTPolyParams>, level: usize, is_ntt: bool) -> Self {
        let mut poly_ptr: *mut GpuPolyOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_poly_create(params.ctx.raw, level as c_int, &mut poly_ptr as *mut *mut GpuPolyOpaque)
        };
        check_status(status, "gpu_poly_create");
        Self { params, raw: poly_ptr, level, is_ntt }
    }

    fn from_flat(params: Arc<GpuDCRTPolyParams>, level: usize, flat: Vec<u64>, is_ntt: bool) -> Self {
        let poly = Self::new_empty(params, level, is_ntt);
        let status = unsafe { gpu_poly_load_rns(poly.raw, flat.as_ptr(), flat.len()) };
        check_status(status, "gpu_poly_load_rns");
        poly
    }

    fn store_rns_flat(&self) -> Vec<u64> {
        let n = self.params.ring_dimension as usize;
        let len = (self.level + 1) * n;
        let mut flat = vec![0u64; len];
        let status = unsafe { gpu_poly_store_rns(self.raw, flat.as_mut_ptr(), flat.len()) };
        check_status(status, "gpu_poly_store_rns");
        flat
    }

    fn ensure_coeff_domain(&self) -> Self {
        if !self.is_ntt {
            return self.clone();
        }
        let mut tmp = self.clone();
        let status = unsafe { gpu_poly_intt(tmp.raw, self.params.batch() as c_int) };
        check_status(status, "gpu_poly_intt");
        tmp.is_ntt = false;
        tmp
    }

    fn ntt_in_place(&mut self) {
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

    fn constant_with_value(params: &Arc<GpuDCRTPolyParams>, value: &BigUint) -> Self {
        let n = params.ring_dimension as usize;
        let level = params.crt_depth.saturating_sub(1);
        let mut flat = vec![0u64; (level + 1) * n];
        for limb in 0..=level {
            let modulus = BigUint::from(params.moduli[limb]);
            let residue = (value % &modulus).to_u64().unwrap_or(0);
            let base = limb * n;
            for i in 0..n {
                flat[base + i] = residue;
            }
        }
        Self::from_flat(params.clone(), level, flat, false)
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
        Self {
            params: self.params.clone(),
            raw: poly_ptr,
            level: self.level,
            is_ntt: self.is_ntt,
        }
    }
}

impl Drop for GpuDCRTPoly {
    fn drop(&mut self) {
        if !self.raw.is_null() {
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
            &coeffs
                .iter()
                .map(|v| vec![*v; params.crt_depth()])
                .collect::<Vec<_>>(),
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
        assert!(
            num_limbs <= params.crt_depth,
            "coeff limb count exceeds CRT depth"
        );
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
        let flat = poly.store_rns_flat();
        let modulus = poly.params.modulus();
        let reconstruct_coeffs = poly.params.reconstruct_coeffs_for_level(level);
        let modulus_level = Arc::new(poly.params.modulus_for_level(level));

        parallel_iter!(0..n)
            .map(|i| {
                let mut acc = BigUint::zero();
                for limb in 0..=level {
                    let residue = flat[limb * n + i];
                    acc += &reconstruct_coeffs[limb] * BigUint::from(residue);
                }
                acc %= &*modulus_level;
                FinRingElem::new(acc, modulus.clone())
            })
            .collect()
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
        debug_assert!(int < (1 << n), "Input exceeds representable range for ring dimension");
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
            let coeff_val = coeff.value().try_into().unwrap_or(usize::MAX);
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
    if self.is_ntt {
        panic!("gpu_poly_mul expects coefficient-domain inputs");
    }
    let out = GpuDCRTPoly::new_empty(self.params.clone(), self.level, false);
    let status = unsafe { gpu_poly_mul(out.raw, self.raw, rhs.raw) };
    check_status(status, "gpu_poly_mul");
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
        let value_bytes = if centered_value.is_zero() { Vec::new() } else { centered_value.to_bytes_le() };
        (true, value_bytes)
    } else if coeff_val.is_zero() {
        (false, Vec::new())
    } else {
        (false, coeff_val.to_bytes_le())
    }
}
