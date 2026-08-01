use crate::atom::{AtomClass, AtomId, AtomTable, SelectionDomainRef, SourceKind};
use mxx_ir_core::{
    checks::{check_same_ring, multiplication_type},
    node::{ConcatAxis, ConstantMatrix},
    types::ConcreteMatrixType,
};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{cmp::Ordering, collections::HashMap};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SymbolicExprId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct IndexRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum SymbolicExprNode {
    Zero,
    Atom(AtomId),
    Add(Vec<SymbolicExprId>),
    Scale {
        #[serde(with = "crate::serde_support::bigint")]
        coefficient: BigInt,
        value: SymbolicExprId,
    },
    Mul(Vec<SymbolicExprId>),
    Tensor {
        left: SymbolicExprId,
        right: SymbolicExprId,
    },
    Concat {
        axis: ConcatAxis,
        inputs: Vec<SymbolicExprId>,
    },
    Select {
        domain: SelectionDomainRef,
        branches: Vec<SymbolicExprId>,
    },
    Transpose(SymbolicExprId),
    Slice {
        value: SymbolicExprId,
        rows: Option<IndexRange>,
        columns: Option<IndexRange>,
    },
    Reshape {
        value: SymbolicExprId,
        rows: usize,
        columns: usize,
    },
    ConstantCoefficient {
        value: SymbolicExprId,
        position: usize,
    },
    CrtRecompose {
        inputs: Vec<SymbolicExprId>,
        #[serde(with = "crate::serde_support::bigint_vec")]
        plaintext_moduli: Vec<BigInt>,
        #[serde(with = "crate::serde_support::bigint_vec")]
        reconstruction_coefficients: Vec<BigInt>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymbolicExprRecord {
    pub matrix_type: ConcreteMatrixType,
    pub node: SymbolicExprNode,
}

#[derive(Clone, Debug, Default)]
pub struct SymbolicExprArena {
    records: Vec<SymbolicExprRecord>,
    interner: HashMap<Vec<u8>, SymbolicExprId>,
    structural_digests: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ExpressionError {
    #[error("symbolic expression id is unavailable: {0:?}")]
    MissingExpression(SymbolicExprId),
    #[error("symbolic atom is unavailable: {0:?}")]
    MissingAtom(AtomId),
    #[error("symbolic expression matrix types are incompatible")]
    TypeMismatch,
    #[error("symbolic expression has an invalid structural operation")]
    InvalidStructure,
    #[error("symbolic expression arena exhausted its u32 id space")]
    ArenaExhausted,
    #[error("symbolic expression could not be canonically encoded: {0}")]
    Encoding(String),
}

impl SymbolicExprArena {
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[SymbolicExprRecord] {
        &self.records
    }

    pub fn get(&self, id: SymbolicExprId) -> Option<&SymbolicExprRecord> {
        self.records.get(id.0 as usize)
    }

    pub fn matrix_type(&self, id: SymbolicExprId) -> Result<&ConcreteMatrixType, ExpressionError> {
        Ok(&self.record(id)?.matrix_type)
    }

    pub fn zero(
        &mut self,
        matrix_type: ConcreteMatrixType,
    ) -> Result<SymbolicExprId, ExpressionError> {
        self.intern(SymbolicExprRecord { matrix_type, node: SymbolicExprNode::Zero })
    }

    pub fn atom(
        &mut self,
        atom: AtomId,
        atoms: &AtomTable,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let matrix_type = atoms
            .get(&atom)
            .ok_or_else(|| ExpressionError::MissingAtom(atom.clone()))?
            .matrix_type
            .clone();
        self.intern(SymbolicExprRecord { matrix_type, node: SymbolicExprNode::Atom(atom) })
    }

    pub fn add(
        &mut self,
        matrix_type: ConcreteMatrixType,
        values: impl IntoIterator<Item = SymbolicExprId>,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let mut coefficients = HashMap::<SymbolicExprId, BigInt>::new();
        for value in values {
            self.require_type(value, &matrix_type)?;
            let record = self.record(value)?.clone();
            match record.node {
                SymbolicExprNode::Zero => {}
                SymbolicExprNode::Add(children) => {
                    for child in children {
                        let (coefficient, value) = self.extract_scale(child);
                        self.add_coefficient(&mut coefficients, value, coefficient)?;
                    }
                }
                SymbolicExprNode::Scale { coefficient, value } => {
                    self.add_coefficient(&mut coefficients, value, coefficient)?;
                }
                _ => self.add_coefficient(&mut coefficients, value, BigInt::one())?,
            }
        }
        coefficients.retain(|_, coefficient| !coefficient.is_zero());
        if coefficients.is_empty() {
            return self.zero(matrix_type);
        }
        let mut entries = coefficients.into_iter().collect::<Vec<_>>();
        entries.sort_by(|(left, _), (right, _)| self.compare_structural(*left, *right));
        let mut children = Vec::with_capacity(entries.len());
        for (value, coefficient) in entries {
            children.push(if coefficient.is_one() {
                value
            } else {
                self.scale(coefficient, value)?
            });
        }
        if let [value] = children.as_slice() {
            return Ok(*value);
        }
        self.intern(SymbolicExprRecord { matrix_type, node: SymbolicExprNode::Add(children) })
    }

    pub fn subtract(
        &mut self,
        matrix_type: ConcreteMatrixType,
        left: SymbolicExprId,
        right: SymbolicExprId,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let right = self.scale(-BigInt::one(), right)?;
        self.add(matrix_type, [left, right])
    }

    pub fn scale(
        &mut self,
        coefficient: BigInt,
        value: SymbolicExprId,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let record = self.record(value)?.clone();
        if coefficient.is_zero() || matches!(record.node, SymbolicExprNode::Zero) {
            return self.zero(record.matrix_type);
        }
        if coefficient.is_one() {
            return Ok(value);
        }
        match record.node {
            SymbolicExprNode::Scale { coefficient: inner, value } => {
                self.scale(coefficient * inner, value)
            }
            SymbolicExprNode::Add(children) => {
                let scaled = children
                    .into_iter()
                    .map(|child| self.scale(coefficient.clone(), child))
                    .collect::<Result<Vec<_>, _>>()?;
                self.add(record.matrix_type, scaled)
            }
            _ => self.intern(SymbolicExprRecord {
                matrix_type: record.matrix_type,
                node: SymbolicExprNode::Scale { coefficient, value },
            }),
        }
    }

    pub fn multiply(
        &mut self,
        matrix_type: ConcreteMatrixType,
        values: impl IntoIterator<Item = SymbolicExprId>,
        atoms: &AtomTable,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let mut coefficient = BigInt::one();
        let mut children = Vec::new();
        for value in values {
            let record = self.record(value)?.clone();
            if matches!(record.node, SymbolicExprNode::Zero) {
                return self.zero(matrix_type);
            }
            match record.node {
                SymbolicExprNode::Mul(nested) => children.extend(nested),
                SymbolicExprNode::Scale { coefficient: scalar, value } => {
                    coefficient *= scalar;
                    match self.record(value)?.node.clone() {
                        SymbolicExprNode::Mul(nested) => children.extend(nested),
                        _ => children.push(value),
                    }
                }
                _ => children.push(value),
            }
        }
        if coefficient.is_zero() {
            return self.zero(matrix_type);
        }
        if children.is_empty() {
            return Err(ExpressionError::InvalidStructure);
        }
        self.require_product_type(&children, &matrix_type)?;

        let original = children.clone();
        let mut filtered = children
            .iter()
            .copied()
            .filter(|value| !self.is_identity(*value, atoms))
            .collect::<Vec<_>>();
        if filtered.is_empty() {
            if let Some(identity) = original
                .iter()
                .copied()
                .find(|value| self.matrix_type(*value).is_ok_and(|ty| ty == &matrix_type))
            {
                filtered.push(identity);
            } else {
                filtered = original;
            }
        } else if self.require_product_type(&filtered, &matrix_type).is_err() {
            filtered = original;
        }
        let product = if let [value] = filtered.as_slice() {
            *value
        } else {
            self.intern(SymbolicExprRecord { matrix_type, node: SymbolicExprNode::Mul(filtered) })?
        };
        self.scale(coefficient, product)
    }

    pub fn tensor(
        &mut self,
        matrix_type: ConcreteMatrixType,
        left: SymbolicExprId,
        right: SymbolicExprId,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let left_record = self.record(left)?.clone();
        let right_record = self.record(right)?.clone();
        let expected = ConcreteMatrixType {
            modulus: left_record.matrix_type.modulus.clone(),
            ring_dimension: left_record.matrix_type.ring_dimension,
            rows: left_record.matrix_type.rows.saturating_mul(right_record.matrix_type.rows),
            columns: left_record
                .matrix_type
                .columns
                .saturating_mul(right_record.matrix_type.columns),
        };
        check_same_ring(&left_record.matrix_type, &right_record.matrix_type)
            .map_err(|_| ExpressionError::TypeMismatch)?;
        if expected != matrix_type {
            return Err(ExpressionError::TypeMismatch);
        }
        if matches!(left_record.node, SymbolicExprNode::Zero) ||
            matches!(right_record.node, SymbolicExprNode::Zero)
        {
            return self.zero(matrix_type);
        }
        let (left_coefficient, left) = self.extract_scale(left);
        let (right_coefficient, right) = self.extract_scale(right);
        let tensor = self.intern(SymbolicExprRecord {
            matrix_type,
            node: SymbolicExprNode::Tensor { left, right },
        })?;
        self.scale(left_coefficient * right_coefficient, tensor)
    }

    pub fn concat(
        &mut self,
        matrix_type: ConcreteMatrixType,
        axis: ConcatAxis,
        inputs: Vec<SymbolicExprId>,
    ) -> Result<SymbolicExprId, ExpressionError> {
        if inputs.is_empty() {
            return Err(ExpressionError::InvalidStructure);
        }
        for input in &inputs {
            let ty = self.matrix_type(*input)?;
            check_same_ring(ty, &matrix_type).map_err(|_| ExpressionError::TypeMismatch)?;
        }
        if !self.concat_type_matches(&inputs, axis, &matrix_type)? {
            return Err(ExpressionError::TypeMismatch);
        }
        if inputs.iter().all(|input| self.is_zero(*input)) {
            return self.zero(matrix_type);
        }
        self.intern(SymbolicExprRecord {
            matrix_type,
            node: SymbolicExprNode::Concat { axis, inputs },
        })
    }

    pub fn select(
        &mut self,
        matrix_type: ConcreteMatrixType,
        domain: SelectionDomainRef,
        branches: Vec<SymbolicExprId>,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let (domain_modulus, domain_ring_dimension) = match &domain {
            SelectionDomainRef::Local(domain) | SelectionDomainRef::Imported { domain, .. } => {
                (&domain.modulus, domain.ring_dimension)
            }
        };
        if branches.len() != domain.count() ||
            domain_modulus != &matrix_type.modulus ||
            domain_ring_dimension != matrix_type.ring_dimension
        {
            return Err(ExpressionError::InvalidStructure);
        }
        for branch in &branches {
            self.require_type(*branch, &matrix_type)?;
        }
        self.intern(SymbolicExprRecord {
            matrix_type,
            node: SymbolicExprNode::Select { domain, branches },
        })
    }

    pub fn transpose(
        &mut self,
        matrix_type: ConcreteMatrixType,
        value: SymbolicExprId,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let record = self.record(value)?.clone();
        if matrix_type.modulus != record.matrix_type.modulus ||
            matrix_type.ring_dimension != record.matrix_type.ring_dimension ||
            matrix_type.rows != record.matrix_type.columns ||
            matrix_type.columns != record.matrix_type.rows
        {
            return Err(ExpressionError::TypeMismatch);
        }
        match record.node {
            SymbolicExprNode::Zero => self.zero(matrix_type),
            SymbolicExprNode::Transpose(inner) => Ok(inner),
            _ => self.intern(SymbolicExprRecord {
                matrix_type,
                node: SymbolicExprNode::Transpose(value),
            }),
        }
    }

    pub fn slice(
        &mut self,
        matrix_type: ConcreteMatrixType,
        value: SymbolicExprId,
        rows: Option<IndexRange>,
        columns: Option<IndexRange>,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let record = self.record(value)?.clone();
        validate_range(rows, record.matrix_type.rows)?;
        validate_range(columns, record.matrix_type.columns)?;
        let expected_rows = rows.map_or(record.matrix_type.rows, |range| range.end - range.start);
        let expected_columns =
            columns.map_or(record.matrix_type.columns, |range| range.end - range.start);
        if matrix_type.modulus != record.matrix_type.modulus ||
            matrix_type.ring_dimension != record.matrix_type.ring_dimension ||
            matrix_type.rows != expected_rows ||
            matrix_type.columns != expected_columns
        {
            return Err(ExpressionError::TypeMismatch);
        }
        if matches!(record.node, SymbolicExprNode::Zero) {
            return self.zero(matrix_type);
        }
        if let SymbolicExprNode::Slice {
            value: inner,
            rows: existing_rows,
            columns: existing_columns,
        } = record.node
        {
            return self.slice(
                matrix_type,
                inner,
                compose_range(existing_rows, rows),
                compose_range(existing_columns, columns),
            );
        }
        self.intern(SymbolicExprRecord {
            matrix_type,
            node: SymbolicExprNode::Slice { value, rows, columns },
        })
    }

    pub fn reshape(
        &mut self,
        matrix_type: ConcreteMatrixType,
        value: SymbolicExprId,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let record = self.record(value)?.clone();
        if matrix_type.modulus != record.matrix_type.modulus ||
            matrix_type.ring_dimension != record.matrix_type.ring_dimension ||
            matrix_type.rows.saturating_mul(matrix_type.columns) !=
                record.matrix_type.rows.saturating_mul(record.matrix_type.columns)
        {
            return Err(ExpressionError::TypeMismatch);
        }
        match record.node {
            SymbolicExprNode::Zero => self.zero(matrix_type),
            SymbolicExprNode::Reshape { value: inner, .. } => self.intern(SymbolicExprRecord {
                matrix_type: matrix_type.clone(),
                node: SymbolicExprNode::Reshape {
                    value: inner,
                    rows: matrix_type.rows,
                    columns: matrix_type.columns,
                },
            }),
            _ if record.matrix_type == matrix_type => Ok(value),
            _ => self.intern(SymbolicExprRecord {
                matrix_type: matrix_type.clone(),
                node: SymbolicExprNode::Reshape {
                    value,
                    rows: matrix_type.rows,
                    columns: matrix_type.columns,
                },
            }),
        }
    }

    pub fn constant_coefficient(
        &mut self,
        matrix_type: ConcreteMatrixType,
        value: SymbolicExprId,
        position: usize,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let record = self.record(value)?.clone();
        if !record.matrix_type.is_scalar() ||
            position >= record.matrix_type.ring_dimension ||
            matrix_type != record.matrix_type
        {
            return Err(ExpressionError::TypeMismatch);
        }
        if matches!(record.node, SymbolicExprNode::Zero) {
            return self.zero(matrix_type);
        }
        self.intern(SymbolicExprRecord {
            matrix_type,
            node: SymbolicExprNode::ConstantCoefficient { value, position },
        })
    }

    pub fn crt_recompose(
        &mut self,
        matrix_type: ConcreteMatrixType,
        inputs: Vec<SymbolicExprId>,
        plaintext_moduli: Vec<BigInt>,
        reconstruction_coefficients: Vec<BigInt>,
    ) -> Result<SymbolicExprId, ExpressionError> {
        if inputs.is_empty() ||
            inputs.len() != plaintext_moduli.len() ||
            inputs.len() != reconstruction_coefficients.len()
        {
            return Err(ExpressionError::InvalidStructure);
        }
        let input_type = self.matrix_type(inputs[0])?.clone();
        if input_type.rows != 1 || input_type != matrix_type {
            return Err(ExpressionError::TypeMismatch);
        }
        for input in &inputs[1..] {
            self.require_type(*input, &input_type)?;
        }
        if plaintext_moduli
            .iter()
            .any(|modulus| modulus <= &BigInt::one() || modulus >= &input_type.modulus) ||
            reconstruction_coefficients.iter().any(|coefficient| {
                coefficient < &BigInt::zero() || coefficient >= &input_type.modulus
            })
        {
            return Err(ExpressionError::InvalidStructure);
        }
        self.intern(SymbolicExprRecord {
            matrix_type,
            node: SymbolicExprNode::CrtRecompose {
                inputs,
                plaintext_moduli,
                reconstruction_coefficients,
            },
        })
    }

    pub fn replay(
        &mut self,
        record: SymbolicExprRecord,
        atoms: &AtomTable,
    ) -> Result<SymbolicExprId, ExpressionError> {
        let ty = record.matrix_type;
        Ok(match record.node {
            SymbolicExprNode::Zero => self.zero(ty)?,
            SymbolicExprNode::Atom(atom) => {
                let result = self.atom(atom, atoms)?;
                self.require_type(result, &ty)?;
                result
            }
            SymbolicExprNode::Add(values) => self.add(ty, values)?,
            SymbolicExprNode::Scale { coefficient, value } => {
                let result = self.scale(coefficient, value)?;
                self.require_type(result, &ty)?;
                result
            }
            SymbolicExprNode::Mul(values) => self.multiply(ty, values, atoms)?,
            SymbolicExprNode::Tensor { left, right } => self.tensor(ty, left, right)?,
            SymbolicExprNode::Concat { axis, inputs } => self.concat(ty, axis, inputs)?,
            SymbolicExprNode::Select { domain, branches } => self.select(ty, domain, branches)?,
            SymbolicExprNode::Transpose(value) => self.transpose(ty, value)?,
            SymbolicExprNode::Slice { value, rows, columns } => {
                self.slice(ty, value, rows, columns)?
            }
            SymbolicExprNode::Reshape { value, rows, columns } => {
                if rows != ty.rows || columns != ty.columns {
                    return Err(ExpressionError::TypeMismatch);
                }
                self.reshape(ty, value)?
            }
            SymbolicExprNode::ConstantCoefficient { value, position } => {
                self.constant_coefficient(ty, value, position)?
            }
            SymbolicExprNode::CrtRecompose {
                inputs,
                plaintext_moduli,
                reconstruction_coefficients,
            } => self.crt_recompose(ty, inputs, plaintext_moduli, reconstruction_coefficients)?,
        })
    }

    fn add_coefficient(
        &self,
        coefficients: &mut HashMap<SymbolicExprId, BigInt>,
        value: SymbolicExprId,
        coefficient: BigInt,
    ) -> Result<(), ExpressionError> {
        self.record(value)?;
        *coefficients.entry(value).or_insert_with(BigInt::zero) += coefficient;
        Ok(())
    }

    fn extract_scale(&self, value: SymbolicExprId) -> (BigInt, SymbolicExprId) {
        match &self.records[value.0 as usize].node {
            SymbolicExprNode::Scale { coefficient, value } => (coefficient.clone(), *value),
            _ => (BigInt::one(), value),
        }
    }

    fn is_zero(&self, value: SymbolicExprId) -> bool {
        matches!(self.records[value.0 as usize].node, SymbolicExprNode::Zero)
    }

    fn is_identity(&self, value: SymbolicExprId, atoms: &AtomTable) -> bool {
        let SymbolicExprNode::Atom(id) = &self.records[value.0 as usize].node else {
            return false;
        };
        atoms.get(id).is_some_and(|atom| {
            matches!(
                atom.class,
                AtomClass::Source {
                    source: SourceKind::ConstantMatrix { value: ConstantMatrix::Identity }
                }
            )
        })
    }

    fn require_product_type(
        &self,
        values: &[SymbolicExprId],
        expected: &ConcreteMatrixType,
    ) -> Result<(), ExpressionError> {
        let mut ty = self.matrix_type(values[0])?.clone();
        for value in &values[1..] {
            ty = multiplication_type(&ty, self.matrix_type(*value)?)
                .map_err(|_| ExpressionError::TypeMismatch)?;
        }
        if &ty != expected {
            return Err(ExpressionError::TypeMismatch);
        }
        Ok(())
    }

    fn require_type(
        &self,
        value: SymbolicExprId,
        expected: &ConcreteMatrixType,
    ) -> Result<(), ExpressionError> {
        if self.matrix_type(value)? != expected {
            return Err(ExpressionError::TypeMismatch);
        }
        Ok(())
    }

    fn concat_type_matches(
        &self,
        inputs: &[SymbolicExprId],
        axis: ConcatAxis,
        expected: &ConcreteMatrixType,
    ) -> Result<bool, ExpressionError> {
        let first = self.matrix_type(inputs[0])?;
        let mut rows = first.rows;
        let mut columns = first.columns;
        for input in &inputs[1..] {
            let ty = self.matrix_type(*input)?;
            match axis {
                ConcatAxis::Rows if ty.columns == columns => rows = rows.saturating_add(ty.rows),
                ConcatAxis::Columns if ty.rows == rows => {
                    columns = columns.saturating_add(ty.columns)
                }
                ConcatAxis::Diagonal => {
                    rows = rows.saturating_add(ty.rows);
                    columns = columns.saturating_add(ty.columns);
                }
                _ => return Ok(false),
            }
        }
        Ok(expected.rows == rows &&
            expected.columns == columns &&
            expected.modulus == first.modulus &&
            expected.ring_dimension == first.ring_dimension)
    }

    fn compare_structural(&self, left: SymbolicExprId, right: SymbolicExprId) -> Ordering {
        self.structural_digests[left.0 as usize]
            .cmp(&self.structural_digests[right.0 as usize])
            .then(left.cmp(&right))
    }

    fn record(&self, id: SymbolicExprId) -> Result<&SymbolicExprRecord, ExpressionError> {
        self.get(id).ok_or(ExpressionError::MissingExpression(id))
    }

    fn intern(&mut self, record: SymbolicExprRecord) -> Result<SymbolicExprId, ExpressionError> {
        let key = serde_json::to_vec(&record)
            .map_err(|error| ExpressionError::Encoding(error.to_string()))?;
        if let Some(id) = self.interner.get(&key) {
            return Ok(*id);
        }
        let id = SymbolicExprId(
            u32::try_from(self.records.len()).map_err(|_| ExpressionError::ArenaExhausted)?,
        );
        let digest = self.structural_digest(&record)?;
        self.records.push(record);
        self.structural_digests.push(digest);
        self.interner.insert(key, id);
        Ok(id)
    }

    fn structural_digest(&self, record: &SymbolicExprRecord) -> Result<[u8; 32], ExpressionError> {
        let mut hasher = Sha256::new();
        hasher.update(
            serde_json::to_vec(&record.matrix_type)
                .map_err(|error| ExpressionError::Encoding(error.to_string()))?,
        );
        macro_rules! hash_child {
            ($id:expr) => {{
                let id = $id;
                let digest = self
                    .structural_digests
                    .get(id.0 as usize)
                    .ok_or(ExpressionError::MissingExpression(id))?;
                hasher.update(digest);
            }};
        }
        match &record.node {
            SymbolicExprNode::Zero => hasher.update(b"zero"),
            SymbolicExprNode::Atom(atom) => {
                hasher.update(b"atom");
                hasher.update(
                    serde_json::to_vec(atom)
                        .map_err(|error| ExpressionError::Encoding(error.to_string()))?,
                );
            }
            SymbolicExprNode::Add(values) => {
                hasher.update(b"add");
                for value in values {
                    hash_child!(*value);
                }
            }
            SymbolicExprNode::Scale { coefficient, value } => {
                hasher.update(b"scale");
                hasher.update(coefficient.to_signed_bytes_be());
                hash_child!(*value);
            }
            SymbolicExprNode::Mul(values) => {
                hasher.update(b"mul");
                for value in values {
                    hash_child!(*value);
                }
            }
            SymbolicExprNode::Tensor { left, right } => {
                hasher.update(b"tensor");
                hash_child!(*left);
                hash_child!(*right);
            }
            SymbolicExprNode::Concat { axis, inputs } => {
                hasher.update(b"concat");
                hasher.update(
                    serde_json::to_vec(axis)
                        .map_err(|error| ExpressionError::Encoding(error.to_string()))?,
                );
                for input in inputs {
                    hash_child!(*input);
                }
            }
            SymbolicExprNode::Select { domain, branches } => {
                hasher.update(b"select");
                hasher.update(
                    serde_json::to_vec(domain)
                        .map_err(|error| ExpressionError::Encoding(error.to_string()))?,
                );
                for branch in branches {
                    hash_child!(*branch);
                }
            }
            SymbolicExprNode::Transpose(value) => {
                hasher.update(b"transpose");
                hash_child!(*value);
            }
            SymbolicExprNode::Slice { value, rows, columns } => {
                hasher.update(b"slice");
                hasher.update(
                    serde_json::to_vec(&(rows, columns))
                        .map_err(|error| ExpressionError::Encoding(error.to_string()))?,
                );
                hash_child!(*value);
            }
            SymbolicExprNode::Reshape { value, rows, columns } => {
                hasher.update(b"reshape");
                hasher.update(rows.to_le_bytes());
                hasher.update(columns.to_le_bytes());
                hash_child!(*value);
            }
            SymbolicExprNode::ConstantCoefficient { value, position } => {
                hasher.update(b"constant-coefficient");
                hasher.update(position.to_le_bytes());
                hash_child!(*value);
            }
            SymbolicExprNode::CrtRecompose {
                inputs,
                plaintext_moduli,
                reconstruction_coefficients,
            } => {
                hasher.update(b"crt-recompose");
                hasher.update(
                    serde_json::to_vec(&(plaintext_moduli, reconstruction_coefficients))
                        .map_err(|error| ExpressionError::Encoding(error.to_string()))?,
                );
                for input in inputs {
                    hash_child!(*input);
                }
            }
        }
        Ok(hasher.finalize().into())
    }
}

impl SelectionDomainRef {
    pub fn count(&self) -> usize {
        match self {
            Self::Local(domain) | Self::Imported { domain, .. } => domain.count as usize,
        }
    }
}

fn validate_range(range: Option<IndexRange>, length: usize) -> Result<(), ExpressionError> {
    if range.is_some_and(|range| range.start > range.end || range.end > length) {
        return Err(ExpressionError::InvalidStructure);
    }
    Ok(())
}

fn compose_range(existing: Option<IndexRange>, next: Option<IndexRange>) -> Option<IndexRange> {
    match (existing, next) {
        (Some(existing), Some(next)) => Some(IndexRange {
            start: existing.start.saturating_add(next.start),
            end: existing.start.saturating_add(next.end).min(existing.end),
        }),
        (Some(existing), None) => Some(existing),
        (None, next) => next,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::{Atom, AtomKind};
    use mxx_ir_core::{
        FrozenGraphScopeId, ScopedWireRef,
        types::{NodeId, Port, WireRef},
    };

    fn ty(rows: usize, columns: usize) -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: BigInt::from(97u32), ring_dimension: 8, rows, columns }
    }

    fn id(node: u64) -> AtomId {
        AtomId::Local(ScopedWireRef {
            scope: FrozenGraphScopeId::Root,
            wire: WireRef { node: NodeId(node), port: Port(0) },
        })
    }

    fn source(id: AtomId, matrix_type: ConcreteMatrixType) -> Atom {
        Atom {
            id,
            class: AtomClass::Source {
                source: SourceKind::UniformSample {
                    minimum: BigInt::from(-1),
                    maximum: BigInt::from(1),
                },
            },
            kind: AtomKind::Bounded,
            matrix_type,
        }
    }

    fn identity(id: AtomId, size: usize) -> Atom {
        Atom {
            id,
            class: AtomClass::Source {
                source: SourceKind::ConstantMatrix { value: ConstantMatrix::Identity },
            },
            kind: AtomKind::Bounded,
            matrix_type: ty(size, size),
        }
    }

    #[test]
    fn interning_and_add_cancellation_are_canonical() {
        let mut atoms = AtomTable::default();
        atoms.insert(source(id(1), ty(2, 2)));
        let mut arena = SymbolicExprArena::default();
        let value = arena.atom(id(1), &atoms).expect("atom");
        let repeated = arena.atom(id(1), &atoms).expect("same atom");
        assert_eq!(value, repeated);
        let negative = arena.scale(-BigInt::one(), value).expect("negative");
        let zero = arena.add(ty(2, 2), [value, negative]).expect("cancellation");
        assert!(matches!(arena.get(zero).unwrap().node, SymbolicExprNode::Zero));
    }

    #[test]
    fn flattened_addition_extracts_nested_scale_coefficients() {
        let mut atoms = AtomTable::default();
        atoms.insert(source(id(1), ty(2, 2)));
        atoms.insert(source(id(2), ty(2, 2)));
        let mut arena = SymbolicExprArena::default();
        let a = arena.atom(id(1), &atoms).expect("a");
        let b = arena.atom(id(2), &atoms).expect("b");
        let two_a = arena.scale(BigInt::from(2u8), a).expect("2a");
        let nested = arena.add(ty(2, 2), [two_a, b]).expect("2a + b");
        let minus_two_a = arena.scale(BigInt::from(-2), a).expect("-2a");
        assert_eq!(arena.add(ty(2, 2), [nested, minus_two_a]).expect("cancel"), b);
    }

    #[test]
    fn multiplication_does_not_distribute_over_addition() {
        let mut atoms = AtomTable::default();
        for node in 1..=3 {
            atoms.insert(source(id(node), ty(2, 2)));
        }
        let mut arena = SymbolicExprArena::default();
        let a = arena.atom(id(1), &atoms).unwrap();
        let b = arena.atom(id(2), &atoms).unwrap();
        let c = arena.atom(id(3), &atoms).unwrap();
        let sum = arena.add(ty(2, 2), [a, b]).unwrap();
        let product = arena.multiply(ty(2, 2), [sum, c], &atoms).unwrap();
        assert!(matches!(
            arena.get(product).unwrap().node,
            SymbolicExprNode::Mul(ref children)
                if matches!(arena.get(children[0]).unwrap().node, SymbolicExprNode::Add(_))
        ));
    }

    #[test]
    fn all_identity_product_keeps_the_concrete_output_type() {
        let mut atoms = AtomTable::default();
        atoms.insert(identity(id(1), 1));
        atoms.insert(identity(id(2), 2));
        let mut arena = SymbolicExprArena::default();
        let scalar = arena.atom(id(1), &atoms).expect("scalar identity");
        let matrix = arena.atom(id(2), &atoms).expect("matrix identity");
        let product = arena.multiply(ty(2, 2), [scalar, matrix], &atoms).expect("product");
        assert_eq!(product, matrix);
        assert_eq!(arena.matrix_type(product).expect("type"), &ty(2, 2));
    }

    #[test]
    fn zero_cancellation_scale_distribution_and_ordered_products_are_canonical() {
        let mut atoms = AtomTable::default();
        for node in 1..=2 {
            atoms.insert(source(id(node), ty(2, 2)));
        }
        let mut arena = SymbolicExprArena::default();
        let zero = arena.zero(ty(2, 2)).expect("zero");
        let a = arena.atom(id(1), &atoms).expect("a");
        let b = arena.atom(id(2), &atoms).expect("b");
        assert_eq!(arena.multiply(ty(2, 2), [zero, a], &atoms).unwrap(), zero);
        assert_eq!(arena.subtract(ty(2, 2), a, a).unwrap(), zero);

        let sum = arena.add(ty(2, 2), [a, b]).unwrap();
        let scaled = arena.scale(BigInt::from(3u8), sum).unwrap();
        assert!(matches!(
            arena.get(scaled).unwrap().node,
            SymbolicExprNode::Add(ref children)
                if children.len() == 2 && children.iter().all(|child| matches!(
                    arena.get(*child).unwrap().node,
                    SymbolicExprNode::Scale { ref coefficient, .. }
                        if coefficient == &BigInt::from(3u8)
                ))
        ));

        let ab = arena.multiply(ty(2, 2), [a, b], &atoms).unwrap();
        let ba = arena.multiply(ty(2, 2), [b, a], &atoms).unwrap();
        assert_ne!(ab, ba);
        assert!(matches!(
            arena.get(ab).unwrap().node,
            SymbolicExprNode::Mul(ref children) if children == &[a, b]
        ));
        assert!(matches!(
            arena.get(ba).unwrap().node,
            SymbolicExprNode::Mul(ref children) if children == &[b, a]
        ));
    }

    #[test]
    fn tensor_validates_type_extracts_scales_and_annihilates_zero() {
        let mut atoms = AtomTable::default();
        atoms.insert(source(id(1), ty(2, 1)));
        atoms.insert(source(id(2), ty(3, 2)));
        let mut arena = SymbolicExprArena::default();
        let left = arena.atom(id(1), &atoms).unwrap();
        let right = arena.atom(id(2), &atoms).unwrap();
        let output = ty(6, 2);
        let scaled_left = arena.scale(BigInt::from(2u8), left).unwrap();
        let scaled_right = arena.scale(BigInt::from(-3), right).unwrap();
        let tensor = arena.tensor(output.clone(), scaled_left, scaled_right).unwrap();
        assert!(matches!(
            arena.get(tensor).unwrap().node,
            SymbolicExprNode::Scale { ref coefficient, value }
                if coefficient == &BigInt::from(-6) && matches!(
                    arena.get(value).unwrap().node,
                    SymbolicExprNode::Tensor { left: inner_left, right: inner_right }
                        if inner_left == left && inner_right == right
                )
        ));
        let zero = arena.zero(ty(2, 1)).unwrap();
        let tensor_zero = arena.tensor(output.clone(), zero, right).unwrap();
        assert!(matches!(arena.get(tensor_zero).unwrap().node, SymbolicExprNode::Zero));
        assert!(matches!(arena.tensor(ty(5, 2), left, right), Err(ExpressionError::TypeMismatch)));
    }
}
