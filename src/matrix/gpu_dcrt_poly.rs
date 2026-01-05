use crate::{
    element::PolyElem,
    matrix::PolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, PinnedHostBuffer},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    utils::block_size,
};
#[cfg(test)]
use sequential_test::sequential;
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
    entries: Vec<PinnedHostBuffer<u8>>,
}

impl Debug for GpuDCRTPolyMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cpu_params = self.cpu_params();
        let mut coeffs = Vec::with_capacity(self.nrow);
        for row in 0..self.nrow {
            let mut row_coeffs = Vec::with_capacity(self.ncol);
            for col in 0..self.ncol {
                let bytes = self.entry_bytes(row, col);
                let poly = DCRTPoly::from_compact_bytes(&cpu_params, bytes);
                let values = poly
                    .coeffs()
                    .into_iter()
                    .map(|coeff| coeff.value().clone())
                    .collect::<Vec<_>>();
                row_coeffs.push(values);
            }
            coeffs.push(row_coeffs);
        }
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

        let cpu_params = self.cpu_params();
        let lhs = super::dcrt_poly::DCRTPolyMatrix::from_compact_bytes(
            &cpu_params,
            &self.to_compact_bytes(),
        );
        let rhs = super::dcrt_poly::DCRTPolyMatrix::from_compact_bytes(
            &cpu_params,
            &other.to_compact_bytes(),
        );
        lhs == rhs
    }
}

impl Eq for GpuDCRTPolyMatrix {}

impl GpuDCRTPolyMatrix {
    fn new_empty(params: &GpuDCRTPolyParams, nrow: usize, ncol: usize) -> Self {
        let zero_bytes = zero_compact_bytes(params);
        let mut entries = Vec::with_capacity(nrow.saturating_mul(ncol));
        for _ in 0..nrow.saturating_mul(ncol) {
            entries.push(PinnedHostBuffer::from_slice(&zero_bytes));
        }
        Self { params: params.clone(), nrow, ncol, entries }
    }

    fn entry_index(&self, row: usize, col: usize) -> usize {
        row * self.ncol + col
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

    fn to_cpu_matrix(&self) -> super::dcrt_poly::DCRTPolyMatrix {
        let cpu_params = self.cpu_params();
        let bytes = self.to_compact_bytes();
        super::dcrt_poly::DCRTPolyMatrix::from_compact_bytes(&cpu_params, &bytes)
    }

    pub fn from_cpu_matrix(
        params: &GpuDCRTPolyParams,
        matrix: &super::dcrt_poly::DCRTPolyMatrix,
    ) -> Self {
        let (nrow, ncol) = matrix.size();
        if nrow == 0 || ncol == 0 {
            return Self { params: params.clone(), nrow, ncol, entries: Vec::new() };
        }

        let mut entries = Vec::with_capacity(nrow.saturating_mul(ncol));
        for i in 0..nrow {
            let row = matrix.get_row(i);
            for poly in row.into_iter() {
                let bytes = poly.to_compact_bytes();
                entries.push(PinnedHostBuffer::from_slice(&bytes));
            }
        }

        Self { params: params.clone(), nrow, ncol, entries }
    }

    fn write_block(
        entries: &mut [PinnedHostBuffer<u8>],
        ncol: usize,
        rows: &Range<usize>,
        cols: &Range<usize>,
        block: &GpuBlock,
    ) {
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = block.idx(i, j);
                let entry_idx = (rows.start + i) * ncol + (cols.start + j);
                entries[entry_idx].set_slice(&block.polys[idx].to_compact_bytes());
            }
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

        let bsize = block_size().min(self.nrow.max(1)).min(self.ncol.max(1));
        let row_offsets = block_offsets(0..self.nrow, bsize);
        let col_offsets = block_offsets(0..self.ncol, bsize);

        let mut block_lhs = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_rhs = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_out = GpuBlock::new(&self.params, bsize * bsize);

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                block_lhs.load_from_entries(&self.entries, self.ncol, rows.clone(), cols.clone());
                block_rhs.load_from_entries(&rhs.entries, rhs.ncol, rows.clone(), cols.clone());
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                block_out.prepare(rows_len, cols_len);

                for i in 0..rows_len {
                    for j in 0..cols_len {
                        let idx = block_out.idx(i, j);
                        op(&mut block_out.polys[idx], block_lhs.poly(i, j), block_rhs.poly(i, j));
                    }
                }

                Self::write_block(&mut out.entries, out.ncol, &rows, &cols, &block_out);
            }
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

        let bsize = block_size().min(self.nrow.max(1)).min(self.ncol.max(1));
        let row_offsets = block_offsets(0..self.nrow, bsize);
        let col_offsets = block_offsets(0..self.ncol, bsize);

        let mut block_lhs = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_out = GpuBlock::new(&self.params, bsize * bsize);
        let zero = GpuDCRTPoly::const_zero(&self.params);

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                block_lhs.load_from_entries(&self.entries, self.ncol, rows.clone(), cols.clone());
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                block_out.prepare(rows_len, cols_len);

                for i in 0..rows_len {
                    for j in 0..cols_len {
                        let idx = block_out.idx(i, j);
                        op(&mut block_out.polys[idx], &zero, block_lhs.poly(i, j));
                    }
                }

                Self::write_block(&mut out.entries, out.ncol, &rows, &cols, &block_out);
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

        let bsize = block_size().min(self.nrow.max(1)).min(self.ncol.max(1));
        let row_offsets = block_offsets(0..self.nrow, bsize);
        let col_offsets = block_offsets(0..self.ncol, bsize);

        let mut block_lhs = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_out = GpuBlock::new(&self.params, bsize * bsize);

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                block_lhs.load_from_entries(&self.entries, self.ncol, rows.clone(), cols.clone());
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

                Self::write_block(&mut out.entries, out.ncol, &rows, &cols, &block_out);
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
        for i in 0..rows_len {
            for j in 0..cols_len {
                let idx = self.idx(i, j);
                let entry_idx = (rows.start + i) * ncol + (cols.start + j);
                self.polys[idx].load_from_compact_bytes(entries[entry_idx].as_slice());
                self.polys[idx].ntt_in_place();
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

impl PolyMatrix for GpuDCRTPolyMatrix {
    type P = GpuDCRTPoly;

    fn to_compact_bytes(&self) -> Vec<u8> {
        let mut entries = Vec::with_capacity(self.nrow);
        for i in 0..self.nrow {
            let mut row = Vec::with_capacity(self.ncol);
            for j in 0..self.ncol {
                row.push(self.entry_bytes(i, j).to_vec());
            }
            entries.push(row);
        }
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
        let mut entries = Vec::with_capacity(nrow.saturating_mul(ncol));
        for row in entries_bytes {
            for bytes in row {
                entries.push(PinnedHostBuffer::from_slice(&bytes));
            }
        }
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
        let mut entries = Vec::with_capacity(nrow.saturating_mul(ncol));
        for row in vec {
            for poly in row {
                let bytes = poly.to_compact_bytes();
                entries.push(PinnedHostBuffer::from_slice(&bytes));
            }
        }
        Self { params: params.clone(), nrow, ncol, entries }
    }

    fn entry(&self, i: usize, j: usize) -> Self::P {
        let poly = GpuDCRTPoly::from_compact_bytes(&self.params, self.entry_bytes(i, j));
        poly
    }

    fn set_entry(&mut self, i: usize, j: usize, elem: Self::P) {
        let bytes = elem.to_compact_bytes();
        self.set_entry_bytes(i, j, &bytes);
    }

    fn get_row(&self, i: usize) -> Vec<Self::P> {
        (0..self.ncol)
            .map(|j| {
                let mut poly =
                    GpuDCRTPoly::from_compact_bytes(&self.params, self.entry_bytes(i, j));
                poly.ntt_in_place();
                poly
            })
            .collect()
    }

    fn get_column(&self, j: usize) -> Vec<Self::P> {
        (0..self.nrow)
            .map(|i| {
                let mut poly =
                    GpuDCRTPoly::from_compact_bytes(&self.params, self.entry_bytes(i, j));
                poly.ntt_in_place();
                poly
            })
            .collect()
    }

    fn size(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    fn slice(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        let nrow = row_end - row_start;
        let ncol = col_end - col_start;
        let mut entries = Vec::with_capacity(nrow.saturating_mul(ncol));
        for i in row_start..row_end {
            for j in col_start..col_end {
                entries.push(self.entries[self.entry_index(i, j)].clone());
            }
        }
        Self { params: self.params.clone(), nrow, ncol, entries }
    }

    fn zero(params: &<Self::P as Poly>::Params, nrow: usize, ncol: usize) -> Self {
        Self::new_empty(params, nrow, ncol)
    }

    fn identity(params: &<Self::P as Poly>::Params, size: usize, scalar: Option<Self::P>) -> Self {
        let zero_bytes = zero_compact_bytes(params);
        let scalar_bytes = {
            match scalar {
                Some(poly) => poly.to_compact_bytes(),
                None => one_compact_bytes(params),
            }
        };
        let mut entries = Vec::with_capacity(size.saturating_mul(size));
        for _ in 0..size.saturating_mul(size) {
            entries.push(PinnedHostBuffer::from_slice(&zero_bytes));
        }
        let mut matrix = Self { params: params.clone(), nrow: size, ncol: size, entries };
        for i in 0..size {
            matrix.set_entry_bytes(i, i, &scalar_bytes);
        }
        matrix
    }

    fn transpose(&self) -> Self {
        let mut entries = Vec::with_capacity(self.nrow.saturating_mul(self.ncol));
        for i in 0..self.ncol {
            for j in 0..self.nrow {
                entries.push(self.entries[self.entry_index(j, i)].clone());
            }
        }
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
        let mut entries = Vec::with_capacity(self.nrow.saturating_mul(ncol));
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                entries.push(self.entries[self.entry_index(i, j)].clone());
            }
            for other in others {
                for j in 0..other.ncol {
                    entries.push(other.entries[other.entry_index(i, j)].clone());
                }
            }
        }
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
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                entries.push(self.entries[self.entry_index(i, j)].clone());
            }
        }
        for other in others {
            for i in 0..other.nrow {
                for j in 0..other.ncol {
                    entries.push(other.entries[other.entry_index(i, j)].clone());
                }
            }
        }
        Self { params: self.params.clone(), nrow, ncol: self.ncol, entries }
    }

    fn concat_diag(&self, others: &[&Self]) -> Self {
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let mut out = Self::new_empty(&self.params, nrow, ncol);

        for i in 0..self.nrow {
            for j in 0..self.ncol {
                out.set_entry_bytes(i, j, self.entry_bytes(i, j));
            }
        }

        let mut row_offset = self.nrow;
        let mut col_offset = self.ncol;
        for other in others {
            for i in 0..other.nrow {
                for j in 0..other.ncol {
                    out.set_entry_bytes(row_offset + i, col_offset + j, other.entry_bytes(i, j));
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

        let bsize = block_size().min(other.nrow.max(1)).min(other.ncol.max(1));
        let row_offsets = block_offsets(0..other.nrow, bsize);
        let col_offsets = block_offsets(0..other.ncol, bsize);
        let mut block_b = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_out = GpuBlock::new(&self.params, bsize * bsize);
        let params = Arc::new(self.params.clone());
        let level = params.crt_depth().saturating_sub(1);
        let mut scalar = GpuDCRTPoly::new_empty(params, level, false);

        for i in 0..self.nrow {
            for j in 0..self.ncol {
                scalar.load_from_compact_bytes(self.entry_bytes(i, j));
                scalar.ntt_in_place();
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
                        Self::write_block(
                            &mut out.entries,
                            out.ncol,
                            &out_rows,
                            &out_cols,
                            &block_out,
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
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                for i in 0..rows_len {
                    for j in 0..cols_len {
                        matrix.set_entry_bytes(
                            rows.start + i,
                            cols.start + j,
                            &entries_bytes[i][j],
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
        for i in 0..rows_len {
            let mut row = Vec::with_capacity(cols_len);
            for j in 0..cols_len {
                let mut poly = GpuDCRTPoly::from_compact_bytes(
                    &self.params,
                    self.entry_bytes(rows.start + i, cols.start + j),
                );
                poly.ntt_in_place();
                row.push(poly);
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
        let zero_bytes = zero_compact_bytes(&self.params);

        let bsize = block_size().min(self.nrow.max(1)).min(self.ncol.max(1)).min(rhs.ncol.max(1));
        let row_offsets = block_offsets(0..self.nrow, bsize);
        let col_offsets = block_offsets(0..rhs.ncol, bsize);
        let ip_offsets = block_offsets(0..self.ncol, bsize);

        let mut block_a = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_b = GpuBlock::new(&self.params, bsize * bsize);
        let mut block_acc = GpuBlock::new(&self.params, bsize * bsize);
        let mut scratch = GpuDCRTPoly::new_empty(block_acc.params.clone(), block_acc.level, false);

        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            let rows_len = rows.end - rows.start;
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                let cols_len = cols.end - cols.start;
                block_acc.prepare(rows_len, cols_len);
                let mut acc_init = vec![false; rows_len * cols_len];

                for ip_pair in ip_offsets.windows(2) {
                    let ips = ip_pair[0]..ip_pair[1];
                    let ip_len = ips.end - ips.start;
                    if ip_len == 0 {
                        continue;
                    }
                    block_a.load_from_entries(&self.entries, self.ncol, rows.clone(), ips.clone());
                    block_b.load_from_entries(&rhs.entries, rhs.ncol, ips.clone(), cols.clone());

                    for i in 0..rows_len {
                        for j in 0..cols_len {
                            let acc_idx = block_acc.idx(i, j);
                            for k in 0..ip_len {
                                let a = block_a.poly(i, k);
                                let b = block_b.poly(k, j);
                                if !acc_init[acc_idx] {
                                    GpuDCRTPoly::mul_into(&mut block_acc.polys[acc_idx], a, b);
                                    acc_init[acc_idx] = true;
                                } else {
                                    GpuDCRTPoly::mul_into(&mut scratch, a, b);
                                    block_acc.polys[acc_idx].add_in_place(&scratch);
                                }
                            }
                        }
                    }
                }

                for (idx, initialized) in acc_init.iter().enumerate() {
                    if !*initialized {
                        block_acc.polys[idx].load_from_compact_bytes(&zero_bytes);
                    }
                }
                GpuDCRTPolyMatrix::write_block(
                    &mut out.entries,
                    out.ncol,
                    &rows,
                    &cols,
                    &block_acc,
                );
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

fn zero_compact_bytes(params: &GpuDCRTPolyParams) -> Vec<u8> {
    let ring_dimension = params.ring_dimension() as usize;
    let bit_vector_byte_size = ring_dimension.div_ceil(8);
    let total_byte_size = 4 + bit_vector_byte_size + ring_dimension;
    let mut bytes = vec![0u8; total_byte_size];
    bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
    bytes
}

fn one_compact_bytes(params: &GpuDCRTPolyParams) -> Vec<u8> {
    let ring_dimension = params.ring_dimension() as usize;
    let bit_vector_byte_size = ring_dimension.div_ceil(8);
    let total_byte_size = 4 + bit_vector_byte_size + ring_dimension;
    let mut bytes = vec![0u8; total_byte_size];
    bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
    if ring_dimension > 0 {
        bytes[4 + bit_vector_byte_size] = 1;
    }
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
