use crate::{
    element::PolyElem,
    matrix::PolyMatrix,
    parallel_iter,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GPU_POLY_FORMAT_EVAL, GpuDCRTPoly, GpuDCRTPolyParams, PinnedHostBuffer},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    utils::{block_size, log_mem},
};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;
#[cfg(test)]
use sequential_test::sequential;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Neg, Range, Sub},
    path::Path,
    sync::Arc,
    time::Instant,
};

#[derive(Clone)]
pub struct GpuDCRTPolyMatrix {
    pub params: GpuDCRTPolyParams,
    pub nrow: usize,
    pub ncol: usize,
    entries: Vec<PinnedHostBuffer<u8>>,
}

impl Debug for GpuDCRTPolyMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coeffs = parallel_iter!(0..self.nrow)
            .map(|row| {
                parallel_iter!(0..self.ncol)
                    .map(|col| {
                        let poly = self.entry(row, col);
                        poly.coeffs()
                            .into_iter()
                            .map(|coeff| coeff.value().clone())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        f.debug_struct("GpuDCRTPolyMatrix")
            .field("params", &self.params)
            .field("nrow", &self.nrow)
            .field("ncol", &self.ncol)
            .field("coeffs", &coeffs)
            .finish()
    }
}

impl PartialEq for GpuDCRTPolyMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.params != other.params || self.nrow != other.nrow || self.ncol != other.ncol {
            return false;
        }
        if self.entries == other.entries {
            return true;
        }
        let lhs = self.to_cpu_matrix();
        let rhs = other.to_cpu_matrix();
        lhs == rhs
    }
}

impl Eq for GpuDCRTPolyMatrix {}

impl GpuDCRTPolyMatrix {
    fn new_empty(params: &GpuDCRTPolyParams, nrow: usize, ncol: usize) -> Self {
        let zero_bytes = zero_rns_bytes(params);
        let entries = parallel_iter!(0..nrow.saturating_mul(ncol))
            .map(|_| PinnedHostBuffer::from_slice(&zero_bytes))
            .collect::<Vec<_>>();
        Self { params: params.clone(), nrow, ncol, entries }
    }

    fn entry_index(&self, row: usize, col: usize) -> usize {
        row * self.ncol + col
    }

    fn poly_from_rns_bytes(&self, bytes: &[u8]) -> GpuDCRTPoly {
        let params = Arc::new(self.params.clone());
        let level = params.crt_depth().saturating_sub(1);
        let mut poly = GpuDCRTPoly::new_empty(params, level, true);
        poly.load_rns_bytes(bytes, GPU_POLY_FORMAT_EVAL);
        poly
    }

    pub(crate) fn entry_bytes(&self, row: usize, col: usize) -> &[u8] {
        self.entries[self.entry_index(row, col)].as_slice()
    }

    fn set_entry_bytes(&mut self, row: usize, col: usize, bytes: &[u8]) {
        let idx = self.entry_index(row, col);
        self.entries[idx].set_slice(bytes);
    }

    fn cpu_params(&self) -> DCRTPolyParams {
        DCRTPolyParams::new(
            self.params.ring_dimension(),
            self.params.crt_depth(),
            self.params.crt_bits(),
            self.params.base_bits(),
        )
    }

    pub(crate) fn to_cpu_matrix(&self) -> super::dcrt_poly::DCRTPolyMatrix {
        let cpu_params = self.cpu_params();
        if self.nrow == 0 || self.ncol == 0 {
            return super::dcrt_poly::DCRTPolyMatrix::new_empty(&cpu_params, self.nrow, self.ncol);
        }
        let bytes_per_poly = rns_bytes_len(&self.params);
        if bytes_per_poly == 0 {
            return super::dcrt_poly::DCRTPolyMatrix::new_empty(&cpu_params, self.nrow, self.ncol);
        }
        let total = self.nrow.saturating_mul(self.ncol);
        let level = cpu_params.crt_depth().saturating_sub(1);
        let n = cpu_params.ring_dimension() as usize;
        let expected_len = (level + 1).saturating_mul(n);
        let reconstruct_coeffs = Arc::new(self.params.reconstruct_coeffs_for_level(level));
        let modulus_level = Arc::new(self.params.modulus_for_level(level));

        let polys_cpu = parallel_iter!(0..total)
            .map(|idx| {
                let entry = &self.entries[idx];
                let entry_bytes = entry.as_slice();
                debug_assert_eq!(
                    entry_bytes.len(),
                    bytes_per_poly,
                    "entry bytes must match rns byte size"
                );
                let mut flat = Vec::with_capacity(expected_len);
                for limb_bytes in entry_bytes.chunks_exact(std::mem::size_of::<u64>()) {
                    let bytes: [u8; 8] = limb_bytes.try_into().expect("u64 chunk size mismatch");
                    flat.push(u64::from_le_bytes(bytes));
                }
                debug_assert_eq!(flat.len(), expected_len, "RNS flat length mismatch");

                let mut eval_slots = Vec::with_capacity(n);
                for i in 0..n {
                    let mut acc = BigUint::zero();
                    for limb in 0..=level {
                        let residue = flat[limb * n + i];
                        acc += &reconstruct_coeffs[limb] * BigUint::from(residue);
                    }
                    acc %= &*modulus_level;
                    eval_slots.push(acc);
                }
                DCRTPoly::from_biguints_eval(&cpu_params, &eval_slots)
            })
            .collect::<Vec<_>>();

        let rows = polys_cpu.chunks(self.ncol).map(|row| row.to_vec()).collect::<Vec<_>>();
        super::dcrt_poly::DCRTPolyMatrix::from_poly_vec(&cpu_params, rows)
    }

    pub fn from_cpu_matrix(
        params: &GpuDCRTPolyParams,
        matrix: &super::dcrt_poly::DCRTPolyMatrix,
    ) -> Self {
        let (nrow, ncol) = matrix.size();
        if nrow == 0 || ncol == 0 {
            return Self { params: params.clone(), nrow, ncol, entries: Vec::new() };
        }
        let bytes_per_poly = rns_bytes_len(params);
        if bytes_per_poly == 0 {
            return Self::new_empty(params, nrow, ncol);
        }
        let n = params.ring_dimension() as usize;
        let moduli = params.moduli();
        let moduli_big = moduli.iter().map(|m| BigUint::from(*m)).collect::<Vec<_>>();
        let expected_len = moduli.len().saturating_mul(n);
        let expected_bytes = expected_len * std::mem::size_of::<u64>();
        debug_assert_eq!(bytes_per_poly, expected_bytes, "rns_bytes_len must match moduli*n*u64");

        let mut out = Self::new_empty(params, nrow, ncol);
        out.entries.par_iter_mut().enumerate().for_each(|(idx, entry)| {
            let row = idx / ncol;
            let col = idx % ncol;
            let poly = matrix.entry(row, col);
            let eval_slots = poly.eval_slots();

            let mut flat = vec![0u64; expected_len];
            for (limb, modulus) in moduli_big.iter().enumerate() {
                let base = limb * n;
                for coeff_idx in 0..n {
                    let value = eval_slots.get(coeff_idx).cloned().unwrap_or_default();
                    let residue = (value % modulus).to_u64().unwrap_or(0);
                    flat[base + coeff_idx] = residue;
                }
            }

            let bytes = unsafe {
                std::slice::from_raw_parts(
                    flat.as_ptr() as *const u8,
                    flat.len() * std::mem::size_of::<u64>(),
                )
            };
            entry.set_slice(bytes);
        });

        out
    }

    pub fn concat_rows_owned(self, mut others: Vec<Self>) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.ncol != other.ncol {
                panic!(
                    "Concat error: while the shape of the first matrix is ({}, {}), that of the {}-th matrix is ({},{})",
                    self.nrow, self.ncol, idx, other.nrow, other.ncol
                );
            }
            if self.params != other.params {
                panic!(
                    "Concat error: mismatched params at index {} (lhs={:?}, rhs={:?})",
                    idx, self.params, other.params
                );
            }
        }
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let ncol = self.ncol;
        let params = self.params;
        let mut entries = self.entries;
        entries.reserve(others.iter().map(|x| x.entries.len()).sum::<usize>());
        for other in others.iter_mut() {
            entries.append(&mut other.entries);
        }
        Self { params, nrow, ncol, entries }
    }

    pub fn concat_columns_owned(self, others: Vec<Self>) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.nrow != other.nrow {
                panic!(
                    "Concat error: while the shape of the first matrix is ({}, {}), that of the {}-th matrix is ({},{})",
                    self.nrow, self.ncol, idx, other.nrow, other.ncol
                );
            }
            if self.params != other.params {
                panic!(
                    "Concat error: mismatched params at index {} (lhs={:?}, rhs={:?})",
                    idx, self.params, other.params
                );
            }
        }

        let nrow = self.nrow;
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let params = self.params;
        let mut entries = Vec::with_capacity(nrow.saturating_mul(ncol));
        let mut sources = Vec::with_capacity(1 + others.len());
        sources.push((self.ncol, self.entries.into_iter()));
        for other in others.into_iter() {
            sources.push((other.ncol, other.entries.into_iter()));
        }
        for _ in 0..nrow {
            for (row_ncol, iter) in sources.iter_mut() {
                entries.extend(iter.by_ref().take(*row_ncol));
            }
        }
        Self { params, nrow, ncol, entries }
    }

    fn write_block(
        entries: &mut [PinnedHostBuffer<u8>],
        ncol: usize,
        rows: &Range<usize>,
        cols: &Range<usize>,
        block: &mut GpuBlock,
    ) {
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        let total = rows_len * cols_len;
        // let bytes_time = Instant::now();
        let bytes_per_poly = rns_bytes_len(&block.params);
        let mut bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
        GpuDCRTPoly::store_rns_bytes_batch(
            &mut block.polys[..total],
            &mut bytes,
            bytes_per_poly,
            GPU_POLY_FORMAT_EVAL,
        );
        // log_mem(format!(
        //     "write_block to rns bytes rows={:?} cols={:?} in {:?}",
        //     rows,
        //     cols,
        //     bytes_time.elapsed()
        // ));
        entries.par_chunks_mut(ncol).enumerate().for_each(|(row_idx, row_entries)| {
            if row_idx < rows.start || row_idx >= rows.end {
                return;
            }
            let local_row = row_idx - rows.start;
            let base = local_row * cols_len;
            row_entries[cols.start..cols.start + cols_len].par_iter_mut().enumerate().for_each(
                |(j, entry)| {
                    let offset = (base + j) * bytes_per_poly;
                    entry.set_slice(&bytes[offset..offset + bytes_per_poly]);
                },
            );
        });
    }

    fn elementwise_block_binary(&self, rhs: &Self, op: BlockOp) -> Self {
        debug_assert!(
            self.nrow == rhs.nrow && self.ncol == rhs.ncol,
            "Elementwise operation requires same dimensions"
        );
        debug_assert_eq!(self.params, rhs.params, "Elementwise operation requires same params");
        let mut out = Self::new_empty(&self.params, self.nrow, self.ncol);
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }

        let block_nrow = block_size().min(self.nrow.max(1));
        let block_ncol = block_size().min(self.ncol.max(1));
        let row_offsets = block_offsets(0..self.nrow, block_nrow);
        let col_offsets = block_offsets(0..self.ncol, block_ncol);

        let mut block_lhs = GpuBlock::new(&self.params, block_nrow * block_ncol);
        let mut block_rhs = GpuBlock::new(&self.params, block_nrow * block_ncol);
        let mut block_out = GpuBlock::new(&self.params, block_nrow * block_ncol);

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            // log_mem(format!("rows {:?}", rows));
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                // log_mem(format!("cols {:?}", cols));
                block_lhs.load_from_entries(&self.entries, self.ncol, rows.clone(), cols.clone());
                block_rhs.load_from_entries(&rhs.entries, rhs.ncol, rows.clone(), cols.clone());
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                block_out.prepare(rows_len, cols_len);

                let total = rows_len * cols_len;
                let out_slice = &mut block_out.polys[..total];
                let lhs_slice = &block_lhs.polys[..total];
                let rhs_slice = &block_rhs.polys[..total];
                match op {
                    BlockOp::Add => GpuDCRTPoly::block_add_into(out_slice, lhs_slice, rhs_slice),
                    BlockOp::Sub => GpuDCRTPoly::block_sub_into(out_slice, lhs_slice, rhs_slice),
                }

                Self::write_block(&mut out.entries, out.ncol, &rows, &cols, &mut block_out);
            }
        }

        out
    }

    fn elementwise_unary_with_zero<F>(&self, op: F) -> Self
    where
        F: Fn(&mut GpuDCRTPoly, &GpuDCRTPoly, &GpuDCRTPoly) + Sync + Send,
    {
        let mut out = Self::new_empty(&self.params, self.nrow, self.ncol);
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }

        let block_nrow = block_size().min(self.nrow.max(1));
        let block_ncol = block_size().min(self.ncol.max(1));
        let row_offsets = block_offsets(0..self.nrow, block_nrow);
        let col_offsets = block_offsets(0..self.ncol, block_ncol);

        let mut block_lhs = GpuBlock::new(&self.params, block_nrow * block_ncol);
        let mut block_out = GpuBlock::new(&self.params, block_nrow * block_ncol);
        let mut zero = GpuDCRTPoly::const_zero(&self.params);
        zero.ntt_in_place();

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                block_lhs.load_from_entries(&self.entries, self.ncol, rows.clone(), cols.clone());
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                block_out.prepare(rows_len, cols_len);

                let total = rows_len * cols_len;
                block_out.polys[..total].par_iter_mut().enumerate().for_each(|(idx, poly)| {
                    let i = idx / cols_len;
                    let j = idx % cols_len;
                    op(poly, &zero, block_lhs.poly(i, j));
                });

                Self::write_block(&mut out.entries, out.ncol, &rows, &cols, &mut block_out);
            }
        }

        out
    }

    fn mul_scalar(&self, scalar: &GpuDCRTPoly) -> Self {
        let mut out = Self::new_empty(&self.params, self.nrow, self.ncol);
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }
        let scalar_coeff = if scalar.is_ntt() { Some(scalar.ensure_coeff_domain()) } else { None };
        let scalar_ref = scalar_coeff.as_ref().unwrap_or(scalar);

        let block_nrow = block_size().min(self.nrow.max(1));
        let block_ncol = block_size().min(self.ncol.max(1));
        let row_offsets = block_offsets(0..self.nrow, block_nrow);
        let col_offsets = block_offsets(0..self.ncol, block_ncol);

        let mut block_lhs = GpuBlock::new(&self.params, block_nrow * block_ncol);
        let mut block_out = GpuBlock::new(&self.params, block_nrow * block_ncol);

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                block_lhs.load_from_entries(&self.entries, self.ncol, rows.clone(), cols.clone());
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                block_out.prepare(rows_len, cols_len);

                let total = rows_len * cols_len;
                block_out.polys[..total].par_iter_mut().enumerate().for_each(|(idx, poly)| {
                    let i = idx / cols_len;
                    let j = idx % cols_len;
                    GpuDCRTPoly::mul_into(poly, block_lhs.poly(i, j), scalar_ref);
                });

                Self::write_block(&mut out.entries, out.ncol, &rows, &cols, &mut block_out);
            }
        }

        out
    }
}

#[derive(Copy, Clone)]
enum BlockOp {
    Add,
    Sub,
}

struct GpuBlock {
    polys: Vec<GpuDCRTPoly>,
    rows: usize,
    cols: usize,
    params: Arc<GpuDCRTPolyParams>,
    level: usize,
}

impl GpuBlock {
    fn new(params: &GpuDCRTPolyParams, capacity: usize) -> Self {
        let params = Arc::new(params.clone());
        let level = params.crt_depth().saturating_sub(1);
        let mut polys = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            polys.push(GpuDCRTPoly::new_empty(params.clone(), level, false));
        }
        Self { polys, rows: 0, cols: 0, params, level }
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if self.polys.len() >= capacity {
            return;
        }
        let current = self.polys.len();
        for _ in current..capacity {
            self.polys.push(GpuDCRTPoly::new_empty(self.params.clone(), self.level, false));
        }
    }

    fn prepare(&mut self, rows: usize, cols: usize) {
        let needed = rows * cols;
        self.ensure_capacity(needed);
        self.rows = rows;
        self.cols = cols;
    }

    fn load_from_entries(
        &mut self,
        entries: &[PinnedHostBuffer<u8>],
        ncol: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        self.prepare(rows_len, cols_len);
        let total = rows_len * cols_len;
        let bytes_per_poly = rns_bytes_len(&self.params);
        if bytes_per_poly == 0 {
            return;
        }
        let mut bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
        bytes.par_chunks_mut(bytes_per_poly).enumerate().for_each(|(idx, chunk)| {
            let i = idx / cols_len;
            let j = idx % cols_len;
            let entry_idx = (rows.start + i) * ncol + (cols.start + j);
            let entry_bytes = entries[entry_idx].as_slice();
            debug_assert_eq!(
                entry_bytes.len(),
                bytes_per_poly,
                "entry bytes must match rns byte size"
            );
            chunk.copy_from_slice(entry_bytes);
        });
        GpuDCRTPoly::load_rns_bytes_batch(
            &mut self.polys[..total],
            &bytes,
            bytes_per_poly,
            GPU_POLY_FORMAT_EVAL,
        );
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    fn poly(&self, row: usize, col: usize) -> &GpuDCRTPoly {
        &self.polys[self.idx(row, col)]
    }
}

impl PolyMatrix for GpuDCRTPolyMatrix {
    type P = GpuDCRTPoly;

    fn to_compact_bytes(&self) -> Vec<u8> {
        let entries = parallel_iter!(0..self.nrow)
            .map(|i| {
                let mut row = Vec::with_capacity(self.ncol);
                let cols = parallel_iter!(0..self.ncol)
                    .map(|j| self.entry_bytes(i, j).to_vec())
                    .collect::<Vec<_>>();
                row.extend(cols);
                row
            })
            .collect::<Vec<_>>();
        bincode::encode_to_vec(&entries, bincode::config::standard())
            .expect("Failed to serialize matrix to compact bytes")
    }

    fn from_compact_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self {
        let entries_bytes: Vec<Vec<Vec<u8>>> =
            bincode::decode_from_slice(bytes, bincode::config::standard())
                .expect("Failed to deserialize matrix from compact bytes")
                .0;
        let nrow = entries_bytes.len();
        let ncol = if nrow > 0 { entries_bytes[0].len() } else { 0 };
        let rows = parallel_iter!(entries_bytes)
            .map(|row| {
                row.into_iter()
                    .map(|bytes| PinnedHostBuffer::from_slice(&bytes))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let entries = rows.into_iter().flatten().collect::<Vec<_>>();
        Self { params: params.clone(), nrow, ncol, entries }
    }

    fn from_poly_vec(params: &<Self::P as Poly>::Params, vec: Vec<Vec<Self::P>>) -> Self {
        if vec.is_empty() {
            return Self { params: params.clone(), nrow: 0, ncol: 0, entries: Vec::new() };
        }
        let nrow = vec.len();
        let ncol = vec[0].len();
        if ncol == 0 {
            return Self { params: params.clone(), nrow, ncol, entries: Vec::new() };
        }
        let bytes_per_poly = rns_bytes_len(params);
        let total = nrow.saturating_mul(ncol);
        let mut polys = vec.into_iter().flatten().collect::<Vec<_>>();
        polys.par_iter_mut().for_each(|poly| {
            if !poly.is_ntt() {
                poly.ntt_in_place();
            }
        });
        let entries = if bytes_per_poly == 0 {
            (0..total).map(|_| PinnedHostBuffer::from_slice(&[])).collect::<Vec<_>>()
        } else {
            let mut bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
            GpuDCRTPoly::store_rns_bytes_batch(
                &mut polys,
                &mut bytes,
                bytes_per_poly,
                GPU_POLY_FORMAT_EVAL,
            );
            bytes.chunks(bytes_per_poly).map(PinnedHostBuffer::from_slice).collect::<Vec<_>>()
        };
        Self { params: params.clone(), nrow, ncol, entries }
    }

    fn entry(&self, i: usize, j: usize) -> Self::P {
        self.poly_from_rns_bytes(self.entry_bytes(i, j))
    }

    fn set_entry(&mut self, i: usize, j: usize, elem: Self::P) {
        let mut elem = elem;
        if !elem.is_ntt() {
            elem.ntt_in_place();
        }
        let bytes_per_poly = rns_bytes_len(&self.params);
        let mut bytes = vec![0u8; bytes_per_poly];
        elem.store_rns_bytes(&mut bytes, GPU_POLY_FORMAT_EVAL);
        self.set_entry_bytes(i, j, &bytes);
    }

    fn get_row(&self, i: usize) -> Vec<Self::P> {
        parallel_iter!(0..self.ncol)
            .map(|j| self.poly_from_rns_bytes(self.entry_bytes(i, j)))
            .collect::<Vec<_>>()
    }

    fn get_column(&self, j: usize) -> Vec<Self::P> {
        parallel_iter!(0..self.nrow)
            .map(|i| self.poly_from_rns_bytes(self.entry_bytes(i, j)))
            .collect::<Vec<_>>()
    }

    fn size(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    fn slice(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        let nrow = row_end - row_start;
        let ncol = col_end - col_start;
        let rows = parallel_iter!(row_start..row_end)
            .map(|i| {
                parallel_iter!(col_start..col_end)
                    .map(|j| self.entries[self.entry_index(i, j)].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let entries = rows.into_iter().flatten().collect::<Vec<_>>();
        Self { params: self.params.clone(), nrow, ncol, entries }
    }

    fn zero(params: &<Self::P as Poly>::Params, nrow: usize, ncol: usize) -> Self {
        Self::new_empty(params, nrow, ncol)
    }

    fn identity(params: &<Self::P as Poly>::Params, size: usize, scalar: Option<Self::P>) -> Self {
        let zero_bytes = zero_rns_bytes(params);
        let scalar_bytes = match scalar {
            Some(mut poly) => {
                if !poly.is_ntt() {
                    poly.ntt_in_place();
                }
                let bytes_per_poly = rns_bytes_len(params);
                let mut bytes = vec![0u8; bytes_per_poly];
                poly.store_rns_bytes(&mut bytes, GPU_POLY_FORMAT_EVAL);
                bytes
            }
            None => one_rns_bytes(params),
        };
        let entries = parallel_iter!(0..size.saturating_mul(size))
            .map(|idx| {
                let row = idx / size;
                let col = idx % size;
                if row == col {
                    PinnedHostBuffer::from_slice(&scalar_bytes)
                } else {
                    PinnedHostBuffer::from_slice(&zero_bytes)
                }
            })
            .collect::<Vec<_>>();
        Self { params: params.clone(), nrow: size, ncol: size, entries }
    }

    fn transpose(&self) -> Self {
        let rows = parallel_iter!(0..self.ncol)
            .map(|i| {
                parallel_iter!(0..self.nrow)
                    .map(|j| self.entries[self.entry_index(j, i)].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let entries = rows.into_iter().flatten().collect::<Vec<_>>();
        Self { params: self.params.clone(), nrow: self.ncol, ncol: self.nrow, entries }
    }

    fn concat_columns(&self, others: &[&Self]) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.nrow != other.nrow {
                panic!(
                    "Concat error: while the shape of the first matrix is ({}, {}), that of the {}-th matrix is ({},{})",
                    self.nrow, self.ncol, idx, other.nrow, other.ncol
                );
            }
        }
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let rows = parallel_iter!(0..self.nrow)
            .map(|i| {
                let mut row = Vec::with_capacity(ncol);
                let self_start = i * self.ncol;
                let self_end = self_start + self.ncol;
                row.extend(self.entries[self_start..self_end].iter().cloned());
                for other in others {
                    let other_start = i * other.ncol;
                    let other_end = other_start + other.ncol;
                    row.extend(other.entries[other_start..other_end].iter().cloned());
                }
                row
            })
            .collect::<Vec<_>>();
        let entries = rows.into_iter().flatten().collect::<Vec<_>>();
        Self { params: self.params.clone(), nrow: self.nrow, ncol, entries }
    }

    fn concat_rows(&self, others: &[&Self]) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.ncol != other.ncol {
                panic!(
                    "Concat error: while the shape of the first matrix is ({}, {}), that of the {}-th matrix is ({},{})",
                    self.nrow, self.ncol, idx, other.nrow, other.ncol
                );
            }
        }
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let mut entries = Vec::with_capacity(nrow.saturating_mul(self.ncol));
        entries.extend(self.entries.iter().cloned());
        for other in others {
            entries.extend(other.entries.iter().cloned());
        }
        Self { params: self.params.clone(), nrow, ncol: self.ncol, entries }
    }

    fn concat_diag(&self, others: &[&Self]) -> Self {
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let zero_bytes = zero_rns_bytes(&self.params);
        struct DiagBlock<'a> {
            row_start: usize,
            col_start: usize,
            nrow: usize,
            ncol: usize,
            entries: &'a [PinnedHostBuffer<u8>],
        }

        let mut blocks = Vec::with_capacity(1 + others.len());
        let mut row_offset = 0usize;
        let mut col_offset = 0usize;
        blocks.push(DiagBlock {
            row_start: row_offset,
            col_start: col_offset,
            nrow: self.nrow,
            ncol: self.ncol,
            entries: &self.entries,
        });
        row_offset += self.nrow;
        col_offset += self.ncol;
        for other in others {
            blocks.push(DiagBlock {
                row_start: row_offset,
                col_start: col_offset,
                nrow: other.nrow,
                ncol: other.ncol,
                entries: &other.entries,
            });
            row_offset += other.nrow;
            col_offset += other.ncol;
        }

        let rows = parallel_iter!(0..nrow)
            .map(|row_idx| {
                let block = blocks
                    .iter()
                    .find(|b| row_idx >= b.row_start && row_idx < b.row_start + b.nrow)
                    .expect("row index should fall within a diagonal block");
                let local_row = row_idx - block.row_start;
                let mut row = Vec::with_capacity(ncol);
                if block.col_start > 0 {
                    row.extend(
                        (0..block.col_start).map(|_| PinnedHostBuffer::from_slice(&zero_bytes)),
                    );
                }
                let base = local_row * block.ncol;
                row.extend(block.entries[base..base + block.ncol].iter().cloned());
                let tail = ncol - block.col_start - block.ncol;
                if tail > 0 {
                    row.extend((0..tail).map(|_| PinnedHostBuffer::from_slice(&zero_bytes)));
                }
                row
            })
            .collect::<Vec<_>>();
        let entries = rows.into_iter().flatten().collect::<Vec<_>>();
        Self { params: self.params.clone(), nrow, ncol, entries }
    }

    fn tensor(&self, other: &Self) -> Self {
        debug_assert_eq!(self.params, other.params, "Tensor requires same params");
        let out_nrow = self.nrow * other.nrow;
        let out_ncol = self.ncol * other.ncol;
        let mut out = Self::new_empty(&self.params, out_nrow, out_ncol);
        if self.nrow == 0 || self.ncol == 0 || other.nrow == 0 || other.ncol == 0 {
            return out;
        }

        let block_nrow = block_size().min(other.nrow.max(1));
        let block_ncol = block_size().min(other.ncol.max(1));
        let row_offsets = block_offsets(0..other.nrow, block_nrow);
        let col_offsets = block_offsets(0..other.ncol, block_ncol);
        let mut block_b = GpuBlock::new(&self.params, block_nrow * block_ncol);
        let mut block_out = GpuBlock::new(&self.params, block_nrow * block_ncol);
        let params = Arc::new(self.params.clone());
        let level = params.crt_depth().saturating_sub(1);
        let mut scalar = GpuDCRTPoly::new_empty(params, level, false);

        for i in 0..self.nrow {
            for j in 0..self.ncol {
                scalar.load_rns_bytes(self.entry_bytes(i, j), GPU_POLY_FORMAT_EVAL);
                for row_pair in row_offsets.windows(2) {
                    let rows = row_pair[0]..row_pair[1];
                    for col_pair in col_offsets.windows(2) {
                        let cols = col_pair[0]..col_pair[1];
                        block_b.load_from_entries(
                            &other.entries,
                            other.ncol,
                            rows.clone(),
                            cols.clone(),
                        );
                        let rows_len = rows.end - rows.start;
                        let cols_len = cols.end - cols.start;
                        block_out.prepare(rows_len, cols_len);

                        let total = rows_len * cols_len;
                        block_out.polys[..total].par_iter_mut().enumerate().for_each(
                            |(idx, poly)| {
                                let r = idx / cols_len;
                                let c = idx % cols_len;
                                GpuDCRTPoly::mul_into(poly, block_b.poly(r, c), &scalar);
                            },
                        );

                        let out_rows = i * other.nrow + rows.start..i * other.nrow + rows.end;
                        let out_cols = j * other.ncol + cols.start..j * other.ncol + cols.end;
                        Self::write_block(
                            &mut out.entries,
                            out.ncol,
                            &out_rows,
                            &out_cols,
                            &mut block_out,
                        );
                    }
                }
            }
        }
        out
    }

    fn gadget_matrix(params: &<Self::P as Poly>::Params, size: usize) -> Self {
        let cpu_params = DCRTPolyParams::new(
            params.ring_dimension(),
            params.crt_depth(),
            params.crt_bits(),
            params.base_bits(),
        );
        let cpu_matrix = super::dcrt_poly::DCRTPolyMatrix::gadget_matrix(&cpu_params, size);
        Self::from_cpu_matrix(params, &cpu_matrix)
    }

    fn decompose(&self) -> Self {
        let cpu_matrix = self.to_cpu_matrix();
        let decomposed = cpu_matrix.decompose();
        Self::from_cpu_matrix(&self.params, &decomposed)
    }

    fn modulus_switch(
        &self,
        new_modulus: &<<Self::P as Poly>::Params as PolyParams>::Modulus,
    ) -> Self {
        let cpu_matrix = self.to_cpu_matrix();
        let switched = cpu_matrix.modulus_switch(new_modulus);
        Self::from_cpu_matrix(&self.params, &switched)
    }

    fn mul_tensor_identity(&self, other: &Self, identity_size: usize) -> Self {
        debug_assert_eq!(self.ncol, other.nrow * identity_size);
        let slice_width = other.nrow;

        let slices = parallel_iter!(0..identity_size)
            .map(|i| {
                let slice = self.slice(0, self.nrow, i * slice_width, (i + 1) * slice_width);
                slice * other
            })
            .collect::<Vec<_>>();

        let mut refs = Vec::with_capacity(identity_size - 1);
        for i in 1..identity_size {
            refs.push(&slices[i]);
        }
        slices[0].concat_columns(&refs)
    }

    fn mul_tensor_identity_decompose(&self, other: &Self, identity_size: usize) -> Self {
        let log_base_q = self.params.modulus_digits();
        debug_assert_eq!(self.ncol, other.nrow * identity_size * log_base_q);
        let slice_width = other.nrow * log_base_q;

        let outputs_rows = parallel_iter!(0..identity_size)
            .map(|i| {
                let slice = self.slice(0, self.nrow, i * slice_width, (i + 1) * slice_width);
                parallel_iter!(0..other.ncol)
                    .map(|j| &slice * &other.get_column_matrix_decompose(j))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let outputs = outputs_rows.into_iter().flatten().collect::<Vec<_>>();

        let mut refs = Vec::with_capacity(outputs.len() - 1);
        for i in 1..outputs.len() {
            refs.push(&outputs[i]);
        }
        outputs[0].concat_columns(&refs)
    }

    fn get_column_matrix_decompose(&self, j: usize) -> Self {
        let col = self.get_column(j);
        let column_matrix =
            Self::from_poly_vec(&self.params, col.into_iter().map(|poly| vec![poly]).collect());
        column_matrix.decompose()
    }

    fn read_from_files<P: AsRef<Path> + Send + Sync>(
        params: &<Self::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dir_path: P,
        id: &str,
    ) -> Self {
        let bsize = block_size().min(nrow.max(1)).min(ncol.max(1));
        let mut matrix = Self::new_empty(params, nrow, ncol);
        let row_offsets = block_offsets(0..nrow, bsize);
        let col_offsets = block_offsets(0..ncol, bsize);
        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                let mut path = dir_path.as_ref().to_path_buf();
                path.push(format!(
                    "{}_{}_{}.{}_{}.{}.matrix",
                    id, bsize, rows.start, rows.end, cols.start, cols.end
                ));
                let bytes = std::fs::read(&path)
                    .unwrap_or_else(|_| panic!("Failed to read matrix file {path:?}"));
                let entries_bytes: Vec<Vec<Vec<u8>>> =
                    bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap().0;
                let cols_len = cols.end - cols.start;
                matrix.entries.par_chunks_mut(ncol).enumerate().for_each(
                    |(row_idx, row_entries)| {
                        if row_idx < rows.start || row_idx >= rows.end {
                            return;
                        }
                        let local_row = row_idx - rows.start;
                        for j in 0..cols_len {
                            row_entries[cols.start + j].set_slice(&entries_bytes[local_row][j]);
                        }
                    },
                );
            }
        }
        matrix
    }

    fn block_entries(&self, rows: Range<usize>, cols: Range<usize>) -> Vec<Vec<Self::P>> {
        assert!(
            rows.start <= rows.end,
            "Invalid row range: start {} > end {}",
            rows.start,
            rows.end
        );
        assert!(
            cols.start <= cols.end,
            "Invalid column range: start {} > end {}",
            cols.start,
            cols.end
        );
        assert!(
            rows.end <= self.nrow,
            "Row range end {} exceeds matrix rows {}",
            rows.end,
            self.nrow
        );
        assert!(
            cols.end <= self.ncol,
            "Column range end {} exceeds matrix columns {}",
            cols.end,
            self.ncol
        );
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        parallel_iter!(0..rows_len)
            .map(|i| {
                let mut row = Vec::with_capacity(cols_len);
                let cols = parallel_iter!(0..cols_len)
                    .map(|j| {
                        self.poly_from_rns_bytes(self.entry_bytes(rows.start + i, cols.start + j))
                    })
                    .collect::<Vec<_>>();
                row.extend(cols);
                row
            })
            .collect::<Vec<_>>()
    }
}

impl Add for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add<&GpuDCRTPolyMatrix> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn add(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        &self + rhs
    }
}

impl Add<&GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn add(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        debug_assert!(
            self.nrow == rhs.nrow && self.ncol == rhs.ncol,
            "Addition requires matrices of same dimensions: self({}, {}) != rhs({}, {})",
            self.nrow,
            self.ncol,
            rhs.nrow,
            rhs.ncol
        );
        self.elementwise_block_binary(rhs, BlockOp::Add)
    }
}

impl Sub for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub<&GpuDCRTPolyMatrix> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        &self - rhs
    }
}

impl Sub<&GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        debug_assert!(
            self.nrow == rhs.nrow && self.ncol == rhs.ncol,
            "Subtraction requires matrices of same dimensions: self({}, {}) != rhs({}, {})",
            self.nrow,
            self.ncol,
            rhs.nrow,
            rhs.ncol
        );
        self.elementwise_block_binary(rhs, BlockOp::Sub)
    }
}

impl Mul for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<&GpuDCRTPolyMatrix> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        &self * rhs
    }
}

impl Mul<GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: GpuDCRTPolyMatrix) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        debug_assert!(
            self.ncol == rhs.nrow,
            "Multiplication condition failed: self.ncol ({}) must equal rhs.nrow ({})",
            self.ncol,
            rhs.nrow
        );
        debug_assert_eq!(self.params, rhs.params, "Multiplication requires same params");
        let mut out = GpuDCRTPolyMatrix::new_empty(&self.params, self.nrow, rhs.ncol);
        if self.nrow == 0 || rhs.ncol == 0 || self.ncol == 0 {
            return out;
        }
        let zero_bytes = zero_rns_bytes(&self.params);

        let block_nrow = block_size().min(self.nrow.max(1));
        let block_ncol = block_size().min(rhs.ncol.max(1));
        let block_nip = block_size().min(self.ncol.max(1));
        let row_offsets = block_offsets(0..self.nrow, block_nrow);
        let col_offsets = block_offsets(0..rhs.ncol, block_ncol);
        let ip_offsets = block_offsets(0..self.ncol, block_nip);

        let mut block_a = GpuBlock::new(&self.params, block_nrow * block_nip);
        let mut block_b = GpuBlock::new(&self.params, block_nip * block_ncol);
        let mut block_acc = GpuBlock::new(&self.params, block_nrow * block_ncol);
        let mut block_tmp = GpuBlock::new(&self.params, block_nrow * block_ncol);

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            // log_mem(format!("rows {:?}", rows));
            let rows_len = rows.end - rows.start;
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                // log_mem(format!("cols {:?}", cols));
                let cols_len = cols.end - cols.start;
                // let prepare_start = Instant::now();
                block_acc.prepare(rows_len, cols_len);
                let total = rows_len * cols_len;
                let bytes_per_poly = zero_bytes.len();
                if bytes_per_poly != 0 {
                    let zero_block = vec![0u8; total.saturating_mul(bytes_per_poly)];
                    GpuDCRTPoly::load_rns_bytes_batch(
                        &mut block_acc.polys[..total],
                        &zero_block,
                        bytes_per_poly,
                        GPU_POLY_FORMAT_EVAL,
                    );
                }

                for ip_pair in ip_offsets.windows(2) {
                    let ips = ip_pair[0]..ip_pair[1];
                    let ip_len = ips.end - ips.start;
                    if ip_len == 0 {
                        continue;
                    }
                    // let a_start = Instant::now();
                    block_a.load_from_entries(&self.entries, self.ncol, rows.clone(), ips.clone());
                    // log_mem(format!("load a with ips={:?} in {:?}", ips, a_start.elapsed()));
                    // let b_start = Instant::now();
                    block_b.load_from_entries(&rhs.entries, rhs.ncol, ips.clone(), cols.clone());
                    // log_mem(format!("load b with ips={:?} in {:?}", ips, b_start.elapsed()));
                    // let op_start = Instant::now();
                    // let prepare_start = Instant::now();
                    block_tmp.prepare(rows_len, cols_len);
                    // log_mem(format!("prepare with ips={:?} in {:?}", ips,
                    // prepare_start.elapsed())); let block_mul_start =
                    // Instant::now();
                    GpuDCRTPoly::block_mul_into(
                        &mut block_tmp.polys[..total],
                        &block_a.polys[..rows_len * ip_len],
                        &block_b.polys[..ip_len * cols_len],
                        rows_len,
                        ip_len,
                        cols_len,
                    );
                    // log_mem(format!(
                    //     "block_mul_into rows={:?} cols={:?} ips={:?} in {:?}",
                    //     rows,
                    //     cols,
                    //     ips,
                    //     block_mul_start.elapsed()
                    // ));
                    // let block_add_start = Instant::now();
                    GpuDCRTPoly::block_add_assign(
                        &mut block_acc.polys[..total],
                        &block_tmp.polys[..total],
                    );
                    // log_mem(format!(
                    //     "block_add_assign rows={:?} cols={:?} ips={:?} in {:?}",
                    //     rows,
                    //     cols,
                    //     ips,
                    //     block_add_start.elapsed()
                    // ));
                    // log_mem(format!(
                    //     "mul op rows={:?} cols={:?} ips={:?} in {:?}",
                    //     rows,
                    //     cols,
                    //     ips,
                    //     op_start.elapsed()
                    // ));
                }
                // let write_start = Instant::now();
                GpuDCRTPolyMatrix::write_block(
                    &mut out.entries,
                    out.ncol,
                    &rows,
                    &cols,
                    &mut block_acc,
                );
                // log_mem(format!(
                //     "mul write rows={:?} cols={:?} in {:?}",
                //     rows,
                //     cols,
                //     write_start.elapsed()
                // ));
            }
        }
        out
    }
}

impl Mul<GpuDCRTPoly> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: GpuDCRTPoly) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<&GpuDCRTPoly> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: &GpuDCRTPoly) -> Self::Output {
        &self * rhs
    }
}

impl Mul<GpuDCRTPoly> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: GpuDCRTPoly) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&GpuDCRTPoly> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: &GpuDCRTPoly) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl Neg for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Neg for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn neg(self) -> Self::Output {
        self.elementwise_unary_with_zero(GpuDCRTPoly::sub_into)
    }
}

fn block_offsets(range: Range<usize>, block: usize) -> Vec<usize> {
    let mut offsets = Vec::new();
    offsets.push(range.start);
    let mut cur = range.start;
    while cur < range.end {
        let next = (cur + block).min(range.end);
        offsets.push(next);
        cur = next;
    }
    offsets
}

fn rns_bytes_len(params: &GpuDCRTPolyParams) -> usize {
    let n = params.ring_dimension() as usize;
    let level = params.crt_depth().saturating_sub(1);
    (level + 1).saturating_mul(n).saturating_mul(std::mem::size_of::<u64>())
}

fn zero_rns_bytes(params: &GpuDCRTPolyParams) -> Vec<u8> {
    vec![0u8; rns_bytes_len(params)]
}

fn one_rns_bytes(params: &GpuDCRTPolyParams) -> Vec<u8> {
    let bytes_len = rns_bytes_len(params);
    if bytes_len == 0 {
        return Vec::new();
    }
    let mut poly = GpuDCRTPoly::const_one(params);
    poly.ntt_in_place();
    let mut bytes = vec![0u8; bytes_len];
    poly.store_rns_bytes(&mut bytes, GPU_POLY_FORMAT_EVAL);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        element::{PolyElem, finite_ring::FinRingElem},
        poly::dcrt::gpu::gpu_device_sync,
    };
    use num_bigint::BigUint;
    use std::sync::Arc;

    fn gpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 17, 1)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let _ = tracing_subscriber::fmt::try_init();
        let (moduli, _crt_bits, _crt_depth) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_gadget_matrix() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let size = 3;
        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, size);
        assert_eq!(gadget_matrix.size().0, size);
        assert_eq!(gadget_matrix.size().1, size * gpu_params.modulus_bits());
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_decompose_basic() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let bit_length = gpu_params.modulus_bits();

        let mut matrix_vec = Vec::with_capacity(2);
        let value = 5;

        let mut row1 = Vec::with_capacity(8);
        row1.push(GpuDCRTPoly::from_usize_to_constant(&gpu_params, value));
        for _ in 1..8 {
            row1.push(GpuDCRTPoly::const_zero(&gpu_params));
        }

        let mut row2 = Vec::with_capacity(8);
        row2.push(GpuDCRTPoly::const_zero(&gpu_params));
        row2.push(GpuDCRTPoly::from_usize_to_constant(&gpu_params, value));
        for _ in 2..8 {
            row2.push(GpuDCRTPoly::const_zero(&gpu_params));
        }

        matrix_vec.push(row1);
        matrix_vec.push(row2);

        let matrix = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix_vec);
        assert_eq!(matrix.size().0, 2);
        assert_eq!(matrix.size().1, 8);

        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, 2);
        assert_eq!(gadget_matrix.size().0, 2);
        assert_eq!(gadget_matrix.size().1, 2 * bit_length);

        let decomposed = matrix.decompose();
        assert_eq!(decomposed.size().0, 2 * bit_length);
        assert_eq!(decomposed.size().1, 8);

        let expected_matrix = &gadget_matrix * &decomposed;
        assert_eq!(expected_matrix.size().0, 2);
        assert_eq!(expected_matrix.size().1, 8);
        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_decompose_with_base8() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let digits_length = gpu_params.modulus_digits();

        let mut matrix_vec = Vec::with_capacity(2);
        let value = 5;

        let mut row1 = Vec::with_capacity(8);
        row1.push(GpuDCRTPoly::from_usize_to_constant(&gpu_params, value));
        for _ in 1..8 {
            row1.push(GpuDCRTPoly::const_zero(&gpu_params));
        }

        let mut row2 = Vec::with_capacity(8);
        row2.push(GpuDCRTPoly::const_zero(&gpu_params));
        row2.push(GpuDCRTPoly::from_usize_to_constant(&gpu_params, value));
        for _ in 2..8 {
            row2.push(GpuDCRTPoly::const_zero(&gpu_params));
        }

        matrix_vec.push(row1);
        matrix_vec.push(row2);

        let matrix = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix_vec);
        assert_eq!(matrix.size().0, 2);
        assert_eq!(matrix.size().1, 8);

        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, 2);
        assert_eq!(gadget_matrix.size().0, 2);
        assert_eq!(gadget_matrix.size().1, 2 * digits_length);

        let decomposed = matrix.decompose();
        assert_eq!(decomposed.size().0, 2 * digits_length);
        assert_eq!(decomposed.size().1, 8);

        let expected_matrix = &gadget_matrix * &decomposed;
        assert_eq!(expected_matrix.size().0, 2);
        assert_eq!(expected_matrix.size().1, 8);
        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_basic_operations() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);

        let zero = GpuDCRTPolyMatrix::zero(&gpu_params, 2, 2);
        let identity = GpuDCRTPolyMatrix::identity(&gpu_params, 2, None);

        let value = 5;

        let matrix_vec = vec![
            vec![
                GpuDCRTPoly::from_usize_to_constant(&gpu_params, value),
                GpuDCRTPoly::const_zero(&gpu_params),
            ],
            vec![
                GpuDCRTPoly::const_zero(&gpu_params),
                GpuDCRTPoly::from_usize_to_constant(&gpu_params, value),
            ],
        ];

        let matrix1 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix_vec);
        assert_eq!(matrix1.entry(0, 0).coeffs()[0].value(), &BigUint::from(value));
        let matrix2 = matrix1.clone();
        assert_eq!(matrix1, matrix2);

        let sum = matrix1.clone() + &matrix2;
        let value_10 = FinRingElem::new(10u32, gpu_params.modulus());
        assert_eq!(sum.entry(0, 0).coeffs()[0], value_10);

        let diff = matrix1.clone() - &matrix2;
        assert_eq!(diff, zero);

        let prod = matrix1 * &identity;
        assert_eq!(prod.size(), (2, 2));
        assert_eq!(prod.entry(0, 0).coeffs()[0].value(), &BigUint::from(value));
        assert_eq!(prod.entry(1, 1).coeffs()[0].value(), &BigUint::from(value));
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_concatenation() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let value = FinRingElem::new(5u32, gpu_params.modulus());

        let matrix1_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value),
                GpuDCRTPoly::const_zero(&gpu_params),
            ],
            vec![GpuDCRTPoly::const_zero(&gpu_params), GpuDCRTPoly::const_zero(&gpu_params)],
        ];

        let matrix1 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix1_vec);

        let matrix2_vec = vec![
            vec![GpuDCRTPoly::const_zero(&gpu_params), GpuDCRTPoly::const_zero(&gpu_params)],
            vec![
                GpuDCRTPoly::const_zero(&gpu_params),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value),
            ],
        ];

        let matrix2 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix2_vec);

        let col_concat = matrix1.concat_columns(&[&matrix2]);
        assert_eq!(col_concat.size().0, 2);
        assert_eq!(col_concat.size().1, 4);
        assert_eq!(col_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(col_concat.entry(1, 3).coeffs()[0], value);

        let row_concat = matrix1.concat_rows(&[&matrix2]);
        assert_eq!(row_concat.size().0, 4);
        assert_eq!(row_concat.size().1, 2);
        assert_eq!(row_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(row_concat.entry(3, 1).coeffs()[0], value);

        let diag_concat = matrix1.concat_diag(&[&matrix2]);
        assert_eq!(diag_concat.size().0, 4);
        assert_eq!(diag_concat.size().1, 4);
        assert_eq!(diag_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(diag_concat.entry(3, 3).coeffs()[0], value);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_tensor_product() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let value = FinRingElem::new(5u32, gpu_params.modulus());

        let matrix1_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value),
                GpuDCRTPoly::const_zero(&gpu_params),
            ],
            vec![GpuDCRTPoly::const_zero(&gpu_params), GpuDCRTPoly::const_zero(&gpu_params)],
        ];

        let matrix1 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix1_vec);

        let matrix2_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value),
                GpuDCRTPoly::const_zero(&gpu_params),
            ],
            vec![GpuDCRTPoly::const_zero(&gpu_params), GpuDCRTPoly::const_zero(&gpu_params)],
        ];

        let matrix2 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix2_vec);

        let tensor = matrix1.tensor(&matrix2);
        assert_eq!(tensor.size().0, 4);
        assert_eq!(tensor.size().1, 4);

        let value_25 = FinRingElem::new(25u32, gpu_params.modulus());
        assert_eq!(tensor.entry(0, 0).coeffs()[0], value_25);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_modulus_switch() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);

        let value00 =
            FinRingElem::new(1023782870921908217643761278891282178u128, gpu_params.modulus());
        let value01 =
            FinRingElem::new(8179012198875468938912873783289218738u128, gpu_params.modulus());
        let value10 =
            FinRingElem::new(2034903202902173762872163465127672178u128, gpu_params.modulus());
        let value11 =
            FinRingElem::new(1990091289902891278121564387120912660u128, gpu_params.modulus());

        let matrix_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value00),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value01),
            ],
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value10),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value11),
            ],
        ];

        let matrix = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix_vec);
        let new_modulus = Arc::new(BigUint::from(2u32));
        let switched = matrix.modulus_switch(&new_modulus);

        assert_eq!(switched.params.modulus(), gpu_params.modulus());

        let new_value00 = value00.modulus_switch(new_modulus.clone());
        let new_value01 = value01.modulus_switch(new_modulus.clone());
        let new_value10 = value10.modulus_switch(new_modulus.clone());
        let new_value11 = value11.modulus_switch(new_modulus.clone());

        let expected_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &new_value00),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &new_value01),
            ],
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &new_value10),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &new_value11),
            ],
        ];

        let expected = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, expected_vec);
        assert_eq!(switched, expected);
    }
}
