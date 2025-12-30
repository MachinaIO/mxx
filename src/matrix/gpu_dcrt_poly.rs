use crate::{
    matrix::PolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, PinnedHostBuffer},
            params::DCRTPolyParams,
        },
    },
    utils::block_size,
};
use std::{
    fmt::Debug,
    ops::{Add, Mul, Neg, Range, Sub},
    path::Path,
    sync::Arc,
};

#[derive(Clone)]
pub struct GpuDCRTPolyMatrix {
    pub params: GpuDCRTPolyParams,
    pub nrow: usize,
    pub ncol: usize,
    entries: PinnedHostBuffer<u64>,
    stride: usize,
}

impl Debug for GpuDCRTPolyMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDCRTPolyMatrix")
            .field("params", &self.params)
            .field("nrow", &self.nrow)
            .field("ncol", &self.ncol)
            .finish()
    }
}

impl PartialEq for GpuDCRTPolyMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params &&
            self.nrow == other.nrow &&
            self.ncol == other.ncol &&
            self.entries.as_slice() == other.entries.as_slice()
    }
}

impl Eq for GpuDCRTPolyMatrix {}

impl GpuDCRTPolyMatrix {
    fn new_empty(params: &GpuDCRTPolyParams, nrow: usize, ncol: usize) -> Self {
        let stride = entry_stride(params);
        let total = nrow.saturating_mul(ncol).saturating_mul(stride);
        let entries = PinnedHostBuffer::new_zeroed(total);
        Self { params: params.clone(), nrow, ncol, entries, stride }
    }

    fn entry_offset(&self, row: usize, col: usize) -> usize {
        (row * self.ncol + col) * self.stride
    }

    fn entry_slice(&self, row: usize, col: usize) -> &[u64] {
        let start = self.entry_offset(row, col);
        let end = start + self.stride;
        &self.entries.as_slice()[start..end]
    }

    fn entry_slice_mut(&mut self, row: usize, col: usize) -> &mut [u64] {
        let start = self.entry_offset(row, col);
        let end = start + self.stride;
        &mut self.entries.as_mut_slice()[start..end]
    }

    fn cpu_params(&self) -> DCRTPolyParams {
        DCRTPolyParams::new(
            self.params.ring_dimension(),
            self.params.crt_depth(),
            self.params.crt_bits(),
            self.params.base_bits(),
        )
    }

    fn to_cpu_matrix(&self) -> super::dcrt_poly::DCRTPolyMatrix {
        let cpu_params = self.cpu_params();
        let bytes = self.to_compact_bytes();
        super::dcrt_poly::DCRTPolyMatrix::from_compact_bytes(&cpu_params, &bytes)
    }

    fn from_cpu_matrix(params: &GpuDCRTPolyParams, matrix: &super::dcrt_poly::DCRTPolyMatrix) -> Self {
        let bytes = matrix.to_compact_bytes();
        Self::from_compact_bytes(params, &bytes)
    }

    fn write_block(out: &mut GpuDCRTPolyMatrix, rows: Range<usize>, cols: Range<usize>, block: &GpuBlock) {
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = block.idx(i, j);
                let slice = out.entry_slice_mut(rows.start + i, cols.start + j);
                block.polys[idx].store_rns_flat_async(slice);
            }
        }
        let count = rows_len * cols_len;
        for idx in 0..count {
            block.polys[idx].sync();
        }
    }

    fn elementwise_binary<F>(&self, rhs: &Self, op: F) -> Self
    where
        F: Fn(&mut GpuDCRTPoly, &GpuDCRTPoly, &GpuDCRTPoly),
    {
        debug_assert!(
            self.nrow == rhs.nrow && self.ncol == rhs.ncol,
            "Elementwise operation requires same dimensions"
        );
        debug_assert_eq!(self.params, rhs.params, "Elementwise operation requires same params");
        let mut out = Self::new_empty(&self.params, self.nrow, self.ncol);
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }
        let bsize = block_size();
        let row_offsets = block_offsets(0..self.nrow, bsize);
        let col_offsets = block_offsets(0..self.ncol, bsize);
        let total_blocks = (row_offsets.len().saturating_sub(1)) * (col_offsets.len().saturating_sub(1));
        let pipeline = pipeline_depth().min(total_blocks).max(1);
        let mut slots = (0..pipeline)
            .map(|_| {
                let slot_params = self.params.clone_with_new_context();
                ElementwiseSlot::new(slot_params, bsize * bsize)
            })
            .collect::<Vec<_>>();

        let mut slot_idx = 0usize;
        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                let slot = &mut slots[slot_idx];
                slot.flush();
                slot.start_binary(self, rhs, &mut out, rows.clone(), cols.clone(), &op);
                slot_idx = (slot_idx + 1) % slots.len();
            }
        }

        for slot in slots.iter_mut() {
            slot.flush();
        }

        out
    }

    fn elementwise_unary_with_zero<F>(&self, op: F) -> Self
    where
        F: Fn(&mut GpuDCRTPoly, &GpuDCRTPoly, &GpuDCRTPoly),
    {
        let mut out = Self::new_empty(&self.params, self.nrow, self.ncol);
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }
        let bsize = block_size();
        let row_offsets = block_offsets(0..self.nrow, bsize);
        let col_offsets = block_offsets(0..self.ncol, bsize);
        let total_blocks = (row_offsets.len().saturating_sub(1)) * (col_offsets.len().saturating_sub(1));
        let pipeline = pipeline_depth().min(total_blocks).max(1);
        let mut slots = (0..pipeline)
            .map(|_| {
                let slot_params = self.params.clone_with_new_context();
                ElementwiseSlot::new(slot_params, bsize * bsize)
            })
            .collect::<Vec<_>>();

        let mut slot_idx = 0usize;
        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                let slot = &mut slots[slot_idx];
                slot.flush();
                slot.start_unary(self, &mut out, rows.clone(), cols.clone(), &op);
                slot_idx = (slot_idx + 1) % slots.len();
            }
        }

        for slot in slots.iter_mut() {
            slot.flush();
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

        let bsize = block_size();
        let row_offsets = block_offsets(0..self.nrow, bsize);
        let col_offsets = block_offsets(0..self.ncol, bsize);

        let mut block_lhs = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_out = GpuBlock::new(&self.params, bsize * bsize);

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                block_lhs.load_from_matrix(self, rows.clone(), cols.clone());
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                block_out.prepare(rows_len, cols_len);

                for i in 0..rows_len {
                    for j in 0..cols_len {
                        let idx = block_out.idx(i, j);
                        GpuDCRTPoly::mul_into(
                            &mut block_out.polys[idx],
                            block_lhs.poly(i, j),
                            scalar_ref,
                        );
                    }
                }

                Self::write_block(&mut out, rows, cols, &block_out);
            }
        }

        out
    }
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

    fn load_from_matrix(
        &mut self,
        matrix: &GpuDCRTPolyMatrix,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        self.prepare(rows_len, cols_len);
        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = self.idx(i, j);
                let slice = matrix.entry_slice(rows.start + i, cols.start + j);
                self.polys[idx].load_from_rns_flat_async(slice);
            }
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    fn poly(&self, row: usize, col: usize) -> &GpuDCRTPoly {
        &self.polys[self.idx(row, col)]
    }
}

struct ElementwiseSlot {
    params: GpuDCRTPolyParams,
    block_lhs: GpuBlock,
    block_rhs: GpuBlock,
    block_out: GpuBlock,
    rows_len: usize,
    cols_len: usize,
    active: bool,
    zero: Option<GpuDCRTPoly>,
}

impl ElementwiseSlot {
    fn new(params: GpuDCRTPolyParams, capacity: usize) -> Self {
        let block_lhs = GpuBlock::new(&params, capacity);
        let block_rhs = GpuBlock::new(&params, capacity);
        let block_out = GpuBlock::new(&params, capacity);
        Self {
            params,
            block_lhs,
            block_rhs,
            block_out,
            rows_len: 0,
            cols_len: 0,
            active: false,
            zero: None,
        }
    }

    fn start_binary<F>(
        &mut self,
        lhs: &GpuDCRTPolyMatrix,
        rhs: &GpuDCRTPolyMatrix,
        out: &mut GpuDCRTPolyMatrix,
        rows: Range<usize>,
        cols: Range<usize>,
        op: &F,
    ) where
        F: Fn(&mut GpuDCRTPoly, &GpuDCRTPoly, &GpuDCRTPoly),
    {
        self.block_lhs.load_from_matrix(lhs, rows.clone(), cols.clone());
        self.block_rhs.load_from_matrix(rhs, rows.clone(), cols.clone());
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        self.block_out.prepare(rows_len, cols_len);

        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = self.block_out.idx(i, j);
                op(
                    &mut self.block_out.polys[idx],
                    self.block_lhs.poly(i, j),
                    self.block_rhs.poly(i, j),
                );
            }
        }

        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = self.block_out.idx(i, j);
                let slice = out.entry_slice_mut(rows.start + i, cols.start + j);
                self.block_out.polys[idx].store_rns_flat_async(slice);
            }
        }
        self.rows_len = rows_len;
        self.cols_len = cols_len;
        self.active = true;
    }

    fn start_unary<F>(
        &mut self,
        lhs: &GpuDCRTPolyMatrix,
        out: &mut GpuDCRTPolyMatrix,
        rows: Range<usize>,
        cols: Range<usize>,
        op: &F,
    ) where
        F: Fn(&mut GpuDCRTPoly, &GpuDCRTPoly, &GpuDCRTPoly),
    {
        self.block_lhs.load_from_matrix(lhs, rows.clone(), cols.clone());
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        self.block_out.prepare(rows_len, cols_len);
        if self.zero.is_none() {
            self.zero = Some(GpuDCRTPoly::const_zero(&self.params));
        }
        let zero = self.zero.as_ref().expect("zero poly should exist");

        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = self.block_out.idx(i, j);
                op(&mut self.block_out.polys[idx], zero, self.block_lhs.poly(i, j));
            }
        }

        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = self.block_out.idx(i, j);
                let slice = out.entry_slice_mut(rows.start + i, cols.start + j);
                self.block_out.polys[idx].store_rns_flat_async(slice);
            }
        }
        self.rows_len = rows_len;
        self.cols_len = cols_len;
        self.active = true;
    }

    fn flush(&mut self) {
        if !self.active {
            return;
        }
        for i in 0..self.rows_len {
            for j in 0..self.cols_len {
                let idx = i * self.cols_len + j;
                self.block_out.polys[idx].sync();
            }
        }
        self.active = false;
    }
}

struct MulSlot {
    block_a: GpuBlock,
    block_b: GpuBlock,
    block_acc: GpuBlock,
    scratch: Option<GpuDCRTPoly>,
    acc_init: Vec<bool>,
    zero_flat: Vec<u64>,
    rows_len: usize,
    cols_len: usize,
    active: bool,
}

impl MulSlot {
    fn new(params: GpuDCRTPolyParams, capacity: usize, stride: usize) -> Self {
        let block_a = GpuBlock::new(&params, capacity);
        let block_b = GpuBlock::new(&params, capacity);
        let block_acc = GpuBlock::new(&params, capacity);
        let zero_flat = vec![0u64; stride];
        Self {
            block_a,
            block_b,
            block_acc,
            scratch: None,
            acc_init: Vec::new(),
            zero_flat,
            rows_len: 0,
            cols_len: 0,
            active: false,
        }
    }

    fn scratch_poly(&mut self) -> &mut GpuDCRTPoly {
        if self.scratch.is_none() {
            let scratch = GpuDCRTPoly::new_empty(
                self.block_acc.params.clone(),
                self.block_acc.level,
                false,
            );
            self.scratch = Some(scratch);
        }
        self.scratch.as_mut().expect("scratch poly should exist")
    }

    fn start_mul(
        &mut self,
        lhs: &GpuDCRTPolyMatrix,
        rhs: &GpuDCRTPolyMatrix,
        out: &mut GpuDCRTPolyMatrix,
        rows: Range<usize>,
        cols: Range<usize>,
        ip_offsets: &[usize],
    ) {
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        self.block_acc.prepare(rows_len, cols_len);
        let count = rows_len * cols_len;
        self.acc_init.clear();
        self.acc_init.resize(count, false);

        for ip_pair in ip_offsets.windows(2) {
            let ips = ip_pair[0]..ip_pair[1];
            let ip_len = ips.end - ips.start;
            if ip_len == 0 {
                continue;
            }
            self.block_a.load_from_matrix(lhs, rows.clone(), ips.clone());
            self.block_b.load_from_matrix(rhs, ips.clone(), cols.clone());

            for i in 0..rows_len {
                for j in 0..cols_len {
                    let acc_idx = self.block_acc.idx(i, j);
                    for k in 0..ip_len {
                        let a = self.block_a.poly(i, k);
                        let b = self.block_b.poly(k, j);
                        if !self.acc_init[acc_idx] {
                            GpuDCRTPoly::mul_into(&mut self.block_acc.polys[acc_idx], a, b);
                            self.acc_init[acc_idx] = true;
                        } else {
                            {
                                let scratch = self.scratch_poly();
                                GpuDCRTPoly::mul_into(scratch, a, b);
                            }
                            let scratch_ref = self.scratch.as_ref().expect("scratch poly should exist");
                            self.block_acc.polys[acc_idx].add_in_place(scratch_ref);
                        }
                    }
                }
            }
        }

        for (idx, initialized) in self.acc_init.iter().enumerate() {
            if !*initialized {
                self.block_acc.polys[idx].load_from_rns_flat(&self.zero_flat);
            }
        }

        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = self.block_acc.idx(i, j);
                self.block_acc.polys[idx].intt_in_place();
                let slice = out.entry_slice_mut(rows.start + i, cols.start + j);
                self.block_acc.polys[idx].store_rns_flat_async(slice);
            }
        }
        self.rows_len = rows_len;
        self.cols_len = cols_len;
        self.active = true;
    }

    fn flush(&mut self) {
        if !self.active {
            return;
        }
        for i in 0..self.rows_len {
            for j in 0..self.cols_len {
                let idx = i * self.cols_len + j;
                self.block_acc.polys[idx].sync();
            }
        }
        self.active = false;
    }
}

fn pipeline_depth() -> usize {
    std::env::var("GPU_MATRIX_PIPELINE_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3)
}

impl PolyMatrix for GpuDCRTPolyMatrix {
    type P = GpuDCRTPoly;

    fn to_compact_bytes(&self) -> Vec<u8> {
        let level = self.params.crt_depth().saturating_sub(1);
        let mut entries_bytes = Vec::with_capacity(self.nrow);
        for i in 0..self.nrow {
            let mut row = Vec::with_capacity(self.ncol);
            for j in 0..self.ncol {
                let slice = self.entry_slice(i, j);
                row.push(GpuDCRTPoly::compact_bytes_from_rns_flat(
                    &self.params,
                    level,
                    slice,
                ));
            }
            entries_bytes.push(row);
        }
        bincode::encode_to_vec(&entries_bytes, bincode::config::standard())
            .expect("Failed to serialize matrix to compact bytes")
    }

    fn from_compact_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self {
        let entries_bytes: Vec<Vec<Vec<u8>>> =
            bincode::decode_from_slice(bytes, bincode::config::standard())
                .expect("Failed to deserialize matrix from compact bytes")
                .0;
        let nrow = entries_bytes.len();
        let ncol = if nrow > 0 { entries_bytes[0].len() } else { 0 };
        let mut matrix = Self::new_empty(params, nrow, ncol);
        let level = params.crt_depth().saturating_sub(1);
        for i in 0..nrow {
            for j in 0..ncol {
                let slice = matrix.entry_slice_mut(i, j);
                GpuDCRTPoly::fill_rns_flat_from_compact_bytes(params, level, &entries_bytes[i][j], slice);
            }
        }
        matrix
    }

    fn from_poly_vec(params: &<Self::P as Poly>::Params, vec: Vec<Vec<Self::P>>) -> Self {
        if vec.is_empty() {
            return Self::new_empty(params, 0, 0);
        }
        let nrow = vec.len();
        let ncol = vec[0].len();
        let mut matrix = Self::new_empty(params, nrow, ncol);
        for (i, row) in vec.into_iter().enumerate() {
            for (j, mut poly) in row.into_iter().enumerate() {
                let slice = matrix.entry_slice_mut(i, j);
                poly.store_rns_flat_async(slice);
                poly.sync();
            }
        }
        matrix
    }

    fn entry(&self, i: usize, j: usize) -> Self::P {
        let level = self.params.crt_depth().saturating_sub(1);
        GpuDCRTPoly::from_rns_flat(
            Arc::new(self.params.clone()),
            level,
            self.entry_slice(i, j),
            false,
        )
    }

    fn set_entry(&mut self, i: usize, j: usize, elem: Self::P) {
        let mut poly = elem;
        let slice = self.entry_slice_mut(i, j);
        poly.store_rns_flat_async(slice);
        poly.sync();
    }

    fn get_row(&self, i: usize) -> Vec<Self::P> {
        (0..self.ncol).map(|j| self.entry(i, j)).collect()
    }

    fn get_column(&self, j: usize) -> Vec<Self::P> {
        (0..self.nrow).map(|i| self.entry(i, j)).collect()
    }

    fn size(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    fn slice(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Self {
        let nrow = row_end - row_start;
        let ncol = col_end - col_start;
        let mut out = Self::new_empty(&self.params, nrow, ncol);
        for i in 0..nrow {
            for j in 0..ncol {
                let src = self.entry_slice(row_start + i, col_start + j);
                out.entry_slice_mut(i, j).copy_from_slice(src);
            }
        }
        out
    }

    fn zero(params: &<Self::P as Poly>::Params, nrow: usize, ncol: usize) -> Self {
        Self::new_empty(params, nrow, ncol)
    }

    fn identity(params: &<Self::P as Poly>::Params, size: usize, scalar: Option<Self::P>) -> Self {
        let mut matrix = Self::new_empty(params, size, size);
        if size == 0 {
            return matrix;
        }
        let level = params.crt_depth().saturating_sub(1);
        let stride = entry_stride(params);
        let mut diag = PinnedHostBuffer::<u64>::new_zeroed(stride);
        match scalar {
            Some(mut poly) => {
                poly.store_rns_flat_async(diag.as_mut_slice());
                poly.sync();
            }
            None => {
                fill_one_rns_flat(params, level, diag.as_mut_slice());
            }
        }
        for i in 0..size {
            matrix.entry_slice_mut(i, i).copy_from_slice(diag.as_slice());
        }
        matrix
    }

    fn transpose(&self) -> Self {
        let mut out = Self::new_empty(&self.params, self.ncol, self.nrow);
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let src = self.entry_slice(i, j);
                out.entry_slice_mut(j, i).copy_from_slice(src);
            }
        }
        out
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
        let mut out = Self::new_empty(&self.params, self.nrow, ncol);
        for i in 0..self.nrow {
            let mut col_offset = 0usize;
            for j in 0..self.ncol {
                let src = self.entry_slice(i, j);
                out.entry_slice_mut(i, col_offset + j).copy_from_slice(src);
            }
            col_offset += self.ncol;
            for other in others {
                for j in 0..other.ncol {
                    let src = other.entry_slice(i, j);
                    out.entry_slice_mut(i, col_offset + j).copy_from_slice(src);
                }
                col_offset += other.ncol;
            }
        }
        out
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
        let mut out = Self::new_empty(&self.params, nrow, self.ncol);
        let mut row_offset = 0usize;
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let src = self.entry_slice(i, j);
                out.entry_slice_mut(row_offset + i, j).copy_from_slice(src);
            }
        }
        row_offset += self.nrow;
        for other in others {
            for i in 0..other.nrow {
                for j in 0..other.ncol {
                    let src = other.entry_slice(i, j);
                    out.entry_slice_mut(row_offset + i, j).copy_from_slice(src);
                }
            }
            row_offset += other.nrow;
        }
        out
    }

    fn concat_diag(&self, others: &[&Self]) -> Self {
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let mut out = Self::new_empty(&self.params, nrow, ncol);
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let src = self.entry_slice(i, j);
                out.entry_slice_mut(i, j).copy_from_slice(src);
            }
        }

        let mut row_offset = self.nrow;
        let mut col_offset = self.ncol;
        for other in others {
            for i in 0..other.nrow {
                for j in 0..other.ncol {
                    let src = other.entry_slice(i, j);
                    out.entry_slice_mut(row_offset + i, col_offset + j).copy_from_slice(src);
                }
            }
            row_offset += other.nrow;
            col_offset += other.ncol;
        }
        out
    }

    fn tensor(&self, other: &Self) -> Self {
        debug_assert_eq!(self.params, other.params, "Tensor requires same params");
        let out_nrow = self.nrow * other.nrow;
        let out_ncol = self.ncol * other.ncol;
        let mut out = Self::new_empty(&self.params, out_nrow, out_ncol);
        if self.nrow == 0 || self.ncol == 0 || other.nrow == 0 || other.ncol == 0 {
            return out;
        }

        let bsize = block_size();
        let row_offsets = block_offsets(0..other.nrow, bsize);
        let col_offsets = block_offsets(0..other.ncol, bsize);
        let mut block_b = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_out = GpuBlock::new(&self.params, bsize * bsize);
        let params = Arc::new(self.params.clone());
        let level = params.crt_depth().saturating_sub(1);
        let mut scalar = GpuDCRTPoly::new_empty(params, level, false);

        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let slice = self.entry_slice(i, j);
                scalar.load_from_rns_flat(slice);
                for row_pair in row_offsets.windows(2) {
                    let rows = row_pair[0]..row_pair[1];
                    for col_pair in col_offsets.windows(2) {
                        let cols = col_pair[0]..col_pair[1];
                        block_b.load_from_matrix(other, rows.clone(), cols.clone());
                        let rows_len = rows.end - rows.start;
                        let cols_len = cols.end - cols.start;
                        block_out.prepare(rows_len, cols_len);

                        for r in 0..rows_len {
                            for c in 0..cols_len {
                                let idx = block_out.idx(r, c);
                                GpuDCRTPoly::mul_into(
                                    &mut block_out.polys[idx],
                                    block_b.poly(r, c),
                                    &scalar,
                                );
                            }
                        }

                        let out_rows = i * other.nrow + rows.start..i * other.nrow + rows.end;
                        let out_cols = j * other.ncol + cols.start..j * other.ncol + cols.end;
                        Self::write_block(&mut out, out_rows, out_cols, &block_out);
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
        Self::from_compact_bytes(&self.params, &switched.to_compact_bytes())
    }

    fn mul_tensor_identity(&self, other: &Self, identity_size: usize) -> Self {
        debug_assert_eq!(self.ncol, other.nrow * identity_size);
        let slice_width = other.nrow;

        let mut slices = Vec::with_capacity(identity_size);
        for i in 0..identity_size {
            let slice = self.slice(0, self.nrow, i * slice_width, (i + 1) * slice_width);
            slices.push(slice * other);
        }

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

        let mut outputs = Vec::with_capacity(identity_size * other.ncol);
        for i in 0..identity_size {
            let slice = self.slice(0, self.nrow, i * slice_width, (i + 1) * slice_width);
            for j in 0..other.ncol {
                outputs.push(&slice * &other.get_column_matrix_decompose(j));
            }
        }

        let mut refs = Vec::with_capacity(outputs.len() - 1);
        for i in 1..outputs.len() {
            refs.push(&outputs[i]);
        }
        outputs[0].concat_columns(&refs)
    }

    fn get_column_matrix_decompose(&self, j: usize) -> Self {
        let col = self.get_column(j);
        let column_matrix = Self::from_poly_vec(
            &self.params,
            col.into_iter().map(|poly| vec![poly]).collect(),
        );
        column_matrix.decompose()
    }

    fn read_from_files<P: AsRef<Path> + Send + Sync>(
        params: &<Self::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dir_path: P,
        id: &str,
    ) -> Self {
        let bsize = block_size();
        let mut matrix = Self::new_empty(params, nrow, ncol);
        let row_offsets = block_offsets(0..nrow, bsize);
        let col_offsets = block_offsets(0..ncol, bsize);
        let level = params.crt_depth().saturating_sub(1);
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
                    bincode::decode_from_slice(&bytes, bincode::config::standard())
                        .unwrap()
                        .0;
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                for i in 0..rows_len {
                    for j in 0..cols_len {
                        let slice = matrix.entry_slice_mut(rows.start + i, cols.start + j);
                        GpuDCRTPoly::fill_rns_flat_from_compact_bytes(
                            params,
                            level,
                            &entries_bytes[i][j],
                            slice,
                        );
                    }
                }
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
        let mut result = Vec::with_capacity(rows_len);
        let level = self.params.crt_depth().saturating_sub(1);
        let params = Arc::new(self.params.clone());
        for i in 0..rows_len {
            let mut row = Vec::with_capacity(cols_len);
            for j in 0..cols_len {
                row.push(GpuDCRTPoly::from_rns_flat(
                    params.clone(),
                    level,
                    self.entry_slice(rows.start + i, cols.start + j),
                    false,
                ));
            }
            result.push(row);
        }
        result
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
        self.elementwise_binary(rhs, GpuDCRTPoly::add_into)
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
        self.elementwise_binary(rhs, GpuDCRTPoly::sub_into)
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
        let bsize = block_size();
        let row_offsets = block_offsets(0..self.nrow, bsize);
        let col_offsets = block_offsets(0..rhs.ncol, bsize);
        let ip_offsets = block_offsets(0..self.ncol, bsize);
        let total_blocks = (row_offsets.len().saturating_sub(1)) * (col_offsets.len().saturating_sub(1));
        let pipeline = pipeline_depth().min(total_blocks).max(1);
        let stride = entry_stride(&self.params);

        let mut slots = (0..pipeline)
            .map(|_| {
                let slot_params = self.params.clone_with_new_context();
                MulSlot::new(slot_params, bsize * bsize, stride)
            })
            .collect::<Vec<_>>();

        let mut slot_idx = 0usize;
        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                let slot = &mut slots[slot_idx];
                slot.flush();
                slot.start_mul(self, rhs, &mut out, rows.clone(), cols.clone(), &ip_offsets);
                slot_idx = (slot_idx + 1) % slots.len();
            }
        }

        for slot in slots.iter_mut() {
            slot.flush();
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

fn entry_stride(params: &GpuDCRTPolyParams) -> usize {
    let level = params.crt_depth().saturating_sub(1);
    (level + 1) * params.ring_dimension() as usize
}

fn fill_one_rns_flat(params: &GpuDCRTPolyParams, level: usize, out: &mut [u64]) {
    out.fill(0);
    let n = params.ring_dimension() as usize;
    for limb in 0..=level {
        let modulus = params.moduli()[limb];
        out[limb * n] = 1 % modulus;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{element::finite_ring::FinRingElem, poly::dcrt::params::DCRTPolyParams};
    use num_bigint::BigUint;
    use std::sync::Arc;

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _crt_bits, _crt_depth) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    #[test]
    fn test_matrix_gadget_matrix() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::default();
        let gpu_params = gpu_params_from_cpu(&params);
        let size = 3;
        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, size);
        assert_eq!(gadget_matrix.size().0, size);
        assert_eq!(gadget_matrix.size().1, size * gpu_params.modulus_bits());
    }

    #[test]
    fn test_matrix_decompose() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::default();
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
    fn test_matrix_decompose_with_base8() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::new(4, 2, 17, 3);
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
    fn test_matrix_decompose_with_unaligned_base() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::new(4, 1, 52, 17);
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
    fn test_matrix_basic_operations() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::default();
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
    fn test_matrix_concatenation() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::default();
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
    fn test_matrix_tensor_product() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::default();
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
    fn test_matrix_modulus_switch() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::default();
        let gpu_params = gpu_params_from_cpu(&params);

        let value00 = FinRingElem::new(1023782870921908217643761278891282178u128, gpu_params.modulus());
        let value01 = FinRingElem::new(8179012198875468938912873783289218738u128, gpu_params.modulus());
        let value10 = FinRingElem::new(2034903202902173762872163465127672178u128, gpu_params.modulus());
        let value11 = FinRingElem::new(1990091289902891278121564387120912660u128, gpu_params.modulus());

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

    #[test]
    #[should_panic(expected = "Addition requires matrices of same dimensions")]
    #[cfg(debug_assertions)]
    fn test_matrix_addition_mismatch() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::default();
        let gpu_params = gpu_params_from_cpu(&params);
        let matrix1 = GpuDCRTPolyMatrix::zero(&gpu_params, 2, 2);
        let matrix2 = GpuDCRTPolyMatrix::zero(&gpu_params, 2, 3);
        let _sum = matrix1 + matrix2;
    }

    #[test]
    #[should_panic(expected = "Multiplication condition failed")]
    #[cfg(debug_assertions)]
    fn test_matrix_multiplication_mismatch() {
        let _guard = crate::poly::dcrt::gpu::gpu_test_lock();
        let params = DCRTPolyParams::default();
        let gpu_params = gpu_params_from_cpu(&params);
        let matrix1 = GpuDCRTPolyMatrix::zero(&gpu_params, 2, 2);
        let matrix2 = GpuDCRTPolyMatrix::zero(&gpu_params, 3, 2);
        let _prod = matrix1 * matrix2;
    }
}
