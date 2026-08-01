use crate::{
    atom::{AtomId, AtomTable, PreimageRelation},
    expression::{ExpressionError, SymbolicExprArena, SymbolicExprId, SymbolicExprNode},
};
use mxx_ir_core::node::ConcatAxis;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum RewriteError {
    #[error(transparent)]
    Expression(#[from] ExpressionError),
    #[error("symbolic rewrite encountered a cycle at {0:?}")]
    Cycle(SymbolicExprId),
}

pub fn rewrite_expression(
    root: SymbolicExprId,
    arena: &mut SymbolicExprArena,
    atoms: &AtomTable,
    relations: &[PreimageRelation],
) -> Result<SymbolicExprId, RewriteError> {
    let mut state = Rewriter {
        arena,
        atoms,
        relations: relations
            .iter()
            .map(|relation| {
                ((relation.left_matrix.clone(), relation.preimage.clone()), relation.product)
            })
            .collect(),
        memo: BTreeMap::new(),
        active: BTreeSet::new(),
    };
    state.rewrite(root)
}

struct Rewriter<'a> {
    arena: &'a mut SymbolicExprArena,
    atoms: &'a AtomTable,
    relations: BTreeMap<(AtomId, AtomId), SymbolicExprId>,
    memo: BTreeMap<SymbolicExprId, SymbolicExprId>,
    active: BTreeSet<SymbolicExprId>,
}

impl Rewriter<'_> {
    fn rewrite(&mut self, id: SymbolicExprId) -> Result<SymbolicExprId, RewriteError> {
        if let Some(rewritten) = self.memo.get(&id) {
            return Ok(*rewritten);
        }
        if !self.active.insert(id) {
            return Err(RewriteError::Cycle(id));
        }
        let record = self.arena.get(id).cloned().ok_or(ExpressionError::MissingExpression(id))?;
        let rewritten =
            match record.node {
                SymbolicExprNode::Zero | SymbolicExprNode::Atom(_) => id,
                SymbolicExprNode::Add(children) => {
                    let children = children
                        .into_iter()
                        .map(|child| self.rewrite(child))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.arena.add(record.matrix_type, children)?
                }
                SymbolicExprNode::Scale { coefficient, value } => {
                    let value = self.rewrite(value)?;
                    self.arena.scale(coefficient, value)?
                }
                SymbolicExprNode::Mul(children) => {
                    let children = children
                        .into_iter()
                        .map(|child| self.rewrite(child))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.rewrite_product(record.matrix_type, children)?
                }
                SymbolicExprNode::Tensor { left, right } => {
                    let left = self.rewrite(left)?;
                    let right = self.rewrite(right)?;
                    self.arena.tensor(record.matrix_type, left, right)?
                }
                SymbolicExprNode::Concat { axis, inputs } => {
                    let inputs = inputs
                        .into_iter()
                        .map(|input| self.rewrite(input))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.arena.concat(record.matrix_type, axis, inputs)?
                }
                SymbolicExprNode::Select { domain, branches } => {
                    let branches = branches
                        .into_iter()
                        .map(|branch| self.rewrite(branch))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.arena.select(record.matrix_type, domain, branches)?
                }
                SymbolicExprNode::Transpose(value) => {
                    let value = self.rewrite(value)?;
                    self.arena.transpose(record.matrix_type, value)?
                }
                SymbolicExprNode::Slice { value, rows, columns } => {
                    let value = self.rewrite(value)?;
                    self.arena.slice(record.matrix_type, value, rows, columns)?
                }
                SymbolicExprNode::Reshape { value, .. } => {
                    let value = self.rewrite(value)?;
                    self.arena.reshape(record.matrix_type, value)?
                }
                SymbolicExprNode::ConstantCoefficient { value, position } => {
                    let value = self.rewrite(value)?;
                    self.arena.constant_coefficient(record.matrix_type, value, position)?
                }
                SymbolicExprNode::CrtRecompose {
                    inputs,
                    plaintext_moduli,
                    reconstruction_coefficients,
                } => {
                    let inputs = inputs
                        .into_iter()
                        .map(|input| self.rewrite(input))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.arena.crt_recompose(
                        record.matrix_type,
                        inputs,
                        plaintext_moduli,
                        reconstruction_coefficients,
                    )?
                }
            };
        self.active.remove(&id);
        self.memo.insert(id, rewritten);
        Ok(rewritten)
    }

    fn rewrite_product(
        &mut self,
        matrix_type: mxx_ir_core::types::ConcreteMatrixType,
        children: Vec<SymbolicExprId>,
    ) -> Result<SymbolicExprId, RewriteError> {
        let normalized = self.arena.multiply(matrix_type.clone(), children, self.atoms)?;
        let (coefficient, mut children) =
            match self.arena.get(normalized).expect("newly interned product").node.clone() {
                SymbolicExprNode::Scale { coefficient, value } => {
                    match self.arena.get(value).expect("scaled value").node.clone() {
                        SymbolicExprNode::Mul(children) => (Some(coefficient), children),
                        _ => return Ok(normalized),
                    }
                }
                SymbolicExprNode::Mul(children) => (None, children),
                _ => return Ok(normalized),
            };
        let mut forced_distribution = None;
        loop {
            if let Some((position, replacement)) = self.preimage_pair(&children) {
                let replacement = self.rewrite(replacement)?;
                children.splice(position..position + 2, [replacement]);
                forced_distribution = Some(position);
                continue;
            }
            if let Some((position, replacement)) = self.block_pair(&children)? {
                children.splice(position..position + 2, [replacement]);
                forced_distribution = Some(position);
                continue;
            }
            let position = forced_distribution
                .take()
                .filter(|position| {
                    children.len() > 1 &&
                        matches!(
                            self.arena.get(children[*position]).map(|record| &record.node),
                            Some(SymbolicExprNode::Add(_))
                        )
                })
                .or_else(|| self.distributable_add(&children));
            if let Some(position) = position {
                let SymbolicExprNode::Add(branches) =
                    self.arena.get(children[position]).expect("checked expression").node.clone()
                else {
                    unreachable!()
                };
                let mut expanded = Vec::with_capacity(branches.len());
                for branch in branches {
                    let mut product = children.clone();
                    product[position] = branch;
                    let branch_type = product_type(self.arena, &product)?;
                    expanded.push(self.rewrite_product(branch_type, product)?);
                }
                let result = self.arena.add(matrix_type.clone(), expanded)?;
                let result = if let Some(coefficient) = coefficient {
                    self.arena.scale(coefficient, result)?
                } else {
                    result
                };
                return self.rewrite(result);
            }
            let result = self.arena.multiply(matrix_type, children, self.atoms)?;
            return Ok(if let Some(coefficient) = coefficient {
                self.arena.scale(coefficient, result)?
            } else {
                result
            });
        }
    }

    fn preimage_pair(&self, children: &[SymbolicExprId]) -> Option<(usize, SymbolicExprId)> {
        children.windows(2).enumerate().find_map(|(position, pair)| {
            let left = atom_id(self.arena, pair[0])?;
            let right = atom_id(self.arena, pair[1])?;
            self.relations.get(&(left, right)).copied().map(|product| (position, product))
        })
    }

    fn block_pair(
        &mut self,
        children: &[SymbolicExprId],
    ) -> Result<Option<(usize, SymbolicExprId)>, RewriteError> {
        for (position, pair) in children.windows(2).enumerate() {
            let Some((left, right)) = aligned_blocks(self.arena, pair[0], pair[1]) else {
                continue;
            };
            let mut terms = Vec::with_capacity(left.len());
            for (left, right) in left.into_iter().zip(right) {
                let ty = product_type(self.arena, &[left, right])?;
                terms.push(self.rewrite_product(ty, vec![left, right])?);
            }
            let ty = self.arena.matrix_type(pair[0])?.clone();
            let output_ty =
                mxx_ir_core::checks::multiplication_type(&ty, self.arena.matrix_type(pair[1])?)
                    .map_err(|_| ExpressionError::TypeMismatch)?;
            return Ok(Some((position, self.arena.add(output_ty, terms)?)));
        }
        Ok(None)
    }

    fn distributable_add(&self, children: &[SymbolicExprId]) -> Option<usize> {
        children.iter().enumerate().find_map(|(position, child)| {
            let SymbolicExprNode::Add(branches) = &self.arena.get(*child)?.node else {
                return None;
            };
            branches
                .iter()
                .any(|branch| self.branch_exposes_redex(children, position, *branch))
                .then_some(position)
        })
    }

    fn branch_exposes_redex(
        &self,
        children: &[SymbolicExprId],
        position: usize,
        branch: SymbolicExprId,
    ) -> bool {
        if position > 0 {
            let left = product_edge_atom(self.arena, children[position - 1], false);
            let right = product_edge_atom(self.arena, branch, true);
            if left.zip(right).is_some_and(|pair| self.relations.contains_key(&pair)) ||
                aligned_blocks(self.arena, children[position - 1], branch).is_some()
            {
                return true;
            }
        }
        if position + 1 < children.len() {
            let left = product_edge_atom(self.arena, branch, false);
            let right = product_edge_atom(self.arena, children[position + 1], true);
            if left.zip(right).is_some_and(|pair| self.relations.contains_key(&pair)) ||
                aligned_blocks(self.arena, branch, children[position + 1]).is_some()
            {
                return true;
            }
        }
        false
    }
}

fn atom_id(arena: &SymbolicExprArena, value: SymbolicExprId) -> Option<AtomId> {
    match &arena.get(value)?.node {
        SymbolicExprNode::Atom(atom) => Some(atom.clone()),
        _ => None,
    }
}

fn product_edge_atom(
    arena: &SymbolicExprArena,
    value: SymbolicExprId,
    first: bool,
) -> Option<AtomId> {
    match &arena.get(value)?.node {
        SymbolicExprNode::Atom(atom) => Some(atom.clone()),
        SymbolicExprNode::Scale { value, .. } => product_edge_atom(arena, *value, first),
        SymbolicExprNode::Mul(children) => product_edge_atom(
            arena,
            *if first { children.first()? } else { children.last()? },
            first,
        ),
        _ => None,
    }
}

fn aligned_blocks(
    arena: &SymbolicExprArena,
    left: SymbolicExprId,
    right: SymbolicExprId,
) -> Option<(Vec<SymbolicExprId>, Vec<SymbolicExprId>)> {
    let SymbolicExprNode::Concat { axis: ConcatAxis::Columns, inputs: left } =
        &arena.get(left)?.node
    else {
        return None;
    };
    let SymbolicExprNode::Concat { axis: ConcatAxis::Rows, inputs: right } =
        &arena.get(right)?.node
    else {
        return None;
    };
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    for (left, right) in left.iter().zip(right) {
        if mxx_ir_core::checks::multiplication_type(
            arena.matrix_type(*left).ok()?,
            arena.matrix_type(*right).ok()?,
        )
        .is_err()
        {
            return None;
        }
    }
    Some((left.clone(), right.clone()))
}

fn product_type(
    arena: &SymbolicExprArena,
    values: &[SymbolicExprId],
) -> Result<mxx_ir_core::types::ConcreteMatrixType, ExpressionError> {
    let mut ty = arena.matrix_type(values[0])?.clone();
    for value in &values[1..] {
        ty = mxx_ir_core::checks::multiplication_type(&ty, arena.matrix_type(*value)?)
            .map_err(|_| ExpressionError::TypeMismatch)?;
    }
    Ok(ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::{Atom, AtomClass, AtomKind, SourceKind};
    use mxx_ir_core::{
        FrozenGraphScopeId, ScopedWireRef,
        node::ConstantMatrix,
        types::{ConcreteMatrixType, NodeId, Port, WireRef},
    };
    use num_bigint::BigInt;

    fn ty() -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: BigInt::from(257u16), ring_dimension: 8, rows: 1, columns: 1 }
    }

    fn id(node: u64) -> AtomId {
        AtomId::Local(ScopedWireRef {
            scope: FrozenGraphScopeId::Root,
            wire: WireRef { node: NodeId(node), port: Port(0) },
        })
    }

    fn insert(atoms: &mut AtomTable, id: AtomId, kind: AtomKind) {
        atoms.insert(Atom {
            id,
            class: AtomClass::Source {
                source: SourceKind::ConstantMatrix {
                    value: ConstantMatrix::UnitRow { index: 0.into() },
                },
            },
            kind,
            matrix_type: ty(),
        });
    }

    fn factors(arena: &SymbolicExprArena, id: SymbolicExprId) -> Vec<AtomId> {
        try_factors(arena, id).expect("product contains only atom factors")
    }

    fn try_factors(arena: &SymbolicExprArena, id: SymbolicExprId) -> Option<Vec<AtomId>> {
        match &arena.get(id).expect("expression").node {
            SymbolicExprNode::Atom(atom) => Some(vec![atom.clone()]),
            SymbolicExprNode::Mul(children) => {
                children.iter().try_fold(Vec::new(), |mut factors, child| {
                    factors.extend(try_factors(arena, *child)?);
                    Some(factors)
                })
            }
            SymbolicExprNode::Scale { value, .. } => try_factors(arena, *value),
            _ => None,
        }
    }

    #[test]
    fn ggh15_preimage_rewrite_expands_only_the_required_path() {
        let s = id(1);
        let b = id(2);
        let e = id(3);
        let k = id(4);
        let s_prime = id(5);
        let p = id(6);
        let relation_error = id(7);
        let mut atoms = AtomTable::default();
        for atom in [&s, &e, &k, &s_prime, &relation_error] {
            insert(&mut atoms, atom.clone(), AtomKind::Bounded);
        }
        for atom in [&b, &p] {
            insert(&mut atoms, atom.clone(), AtomKind::Large);
        }
        let mut arena = SymbolicExprArena::default();
        let atom_expr = |arena: &mut SymbolicExprArena, atom: &AtomId| {
            arena.atom(atom.clone(), &atoms).expect("atom")
        };
        let s_expr = atom_expr(&mut arena, &s);
        let b_expr = atom_expr(&mut arena, &b);
        let e_expr = atom_expr(&mut arena, &e);
        let k_expr = atom_expr(&mut arena, &k);
        let s_prime_expr = atom_expr(&mut arena, &s_prime);
        let p_expr = atom_expr(&mut arena, &p);
        let relation_error_expr = atom_expr(&mut arena, &relation_error);
        let sb = arena.multiply(ty(), [s_expr, b_expr], &atoms).unwrap();
        let c = arena.add(ty(), [sb, e_expr]).unwrap();
        let root = arena.multiply(ty(), [c, k_expr], &atoms).unwrap();
        let signal = arena.multiply(ty(), [s_prime_expr, p_expr], &atoms).unwrap();
        let product = arena.add(ty(), [signal, relation_error_expr]).unwrap();
        let rewritten = rewrite_expression(
            root,
            &mut arena,
            &atoms,
            &[PreimageRelation { left_matrix: b.clone(), preimage: k.clone(), product }],
        )
        .expect("rewrite");
        let SymbolicExprNode::Add(alternatives) =
            &arena.get(rewritten).expect("rewritten expression").node
        else {
            panic!("preimage rewrite must expose signal and noise alternatives")
        };
        let products = alternatives
            .iter()
            .map(|alternative| factors(&arena, *alternative))
            .collect::<Vec<_>>();
        assert!(products.contains(&vec![s.clone(), s_prime, p]));
        assert!(products.contains(&vec![s, relation_error]));
        assert!(products.contains(&vec![e, k]));
        assert!(products.iter().flatten().all(|atom| atom != &b));
    }

    #[test]
    fn preimage_rewrite_requires_exact_atom_identity() {
        let declared_left = id(20);
        let regenerated_left = id(21);
        let preimage = id(22);
        let product_atom = id(23);
        let mut atoms = AtomTable::default();
        for atom in [&declared_left, &regenerated_left, &preimage, &product_atom] {
            insert(&mut atoms, atom.clone(), AtomKind::Bounded);
        }
        let mut arena = SymbolicExprArena::default();
        let regenerated = arena.atom(regenerated_left.clone(), &atoms).unwrap();
        let preimage_expr = arena.atom(preimage.clone(), &atoms).unwrap();
        let root = arena.multiply(ty(), [regenerated, preimage_expr], &atoms).unwrap();
        let product = arena.atom(product_atom, &atoms).unwrap();
        let rewritten = rewrite_expression(
            root,
            &mut arena,
            &atoms,
            &[PreimageRelation { left_matrix: declared_left, preimage, product }],
        )
        .expect("rewrite");

        assert_eq!(rewritten, root);
        assert_eq!(factors(&arena, rewritten), vec![regenerated_left, id(22)]);
    }

    #[test]
    fn repeated_preimage_rewrites_preserve_every_source_factor() {
        let s = id(30);
        let b0 = id(31);
        let e0 = id(32);
        let k0 = id(33);
        let s1 = id(34);
        let b1 = id(35);
        let e1 = id(36);
        let k1 = id(37);
        let s2 = id(38);
        let p = id(39);
        let e2 = id(40);
        let mut atoms = AtomTable::default();
        for atom in [&s, &e0, &k0, &s1, &e1, &k1, &s2, &e2] {
            insert(&mut atoms, atom.clone(), AtomKind::Bounded);
        }
        for atom in [&b0, &b1, &p] {
            insert(&mut atoms, atom.clone(), AtomKind::Large);
        }
        let mut arena = SymbolicExprArena::default();
        let mut atom = |id: &AtomId| arena.atom(id.clone(), &atoms).unwrap();
        let (s_expr, b0_expr, e0_expr, k0_expr) = (atom(&s), atom(&b0), atom(&e0), atom(&k0));
        let (s1_expr, b1_expr, e1_expr, k1_expr) = (atom(&s1), atom(&b1), atom(&e1), atom(&k1));
        let (s2_expr, p_expr, e2_expr) = (atom(&s2), atom(&p), atom(&e2));
        drop(atom);
        let sb0 = arena.multiply(ty(), [s_expr, b0_expr], &atoms).unwrap();
        let c = arena.add(ty(), [sb0, e0_expr]).unwrap();
        let first_step = arena.multiply(ty(), [c, k0_expr], &atoms).unwrap();
        let root = arena.multiply(ty(), [first_step, k1_expr], &atoms).unwrap();
        let s1b1 = arena.multiply(ty(), [s1_expr, b1_expr], &atoms).unwrap();
        let product0 = arena.add(ty(), [s1b1, e1_expr]).unwrap();
        let s2p = arena.multiply(ty(), [s2_expr, p_expr], &atoms).unwrap();
        let product1 = arena.add(ty(), [s2p, e2_expr]).unwrap();

        let rewritten = rewrite_expression(
            root,
            &mut arena,
            &atoms,
            &[
                PreimageRelation {
                    left_matrix: b0.clone(),
                    preimage: k0.clone(),
                    product: product0,
                },
                PreimageRelation {
                    left_matrix: b1.clone(),
                    preimage: k1.clone(),
                    product: product1,
                },
            ],
        )
        .expect("rewrite");
        let SymbolicExprNode::Add(alternatives) = &arena.get(rewritten).unwrap().node else {
            panic!("repeated rewrite must expose every recurrence term")
        };
        let products = alternatives
            .iter()
            .map(|alternative| factors(&arena, *alternative))
            .collect::<Vec<_>>();
        assert!(products.contains(&vec![s.clone(), s1.clone(), s2, p]));
        assert!(products.contains(&vec![s.clone(), s1, e2]));
        assert!(products.contains(&vec![s.clone(), e1, k1.clone()]));
        assert!(products.contains(&vec![e0, k0, k1]));
        assert!(products.iter().flatten().all(|factor| factor != &b0 && factor != &b1));
    }

    #[test]
    fn direct_chained_preimage_relations_rewrite_to_a_fixpoint() {
        let b0 = id(50);
        let k0 = id(51);
        let b1 = id(52);
        let k1 = id(53);
        let product_atom = id(54);
        let mut atoms = AtomTable::default();
        for atom in [&b0, &k0, &b1, &k1, &product_atom] {
            insert(&mut atoms, atom.clone(), AtomKind::Bounded);
        }
        let mut arena = SymbolicExprArena::default();
        let b0_expr = arena.atom(b0.clone(), &atoms).unwrap();
        let k0_expr = arena.atom(k0.clone(), &atoms).unwrap();
        let b1_expr = arena.atom(b1.clone(), &atoms).unwrap();
        let k1_expr = arena.atom(k1.clone(), &atoms).unwrap();
        let final_product = arena.atom(product_atom.clone(), &atoms).unwrap();
        let root = arena.multiply(ty(), [b0_expr, k0_expr], &atoms).unwrap();
        let intermediate = arena.multiply(ty(), [b1_expr, k1_expr], &atoms).unwrap();
        let rewritten = rewrite_expression(
            root,
            &mut arena,
            &atoms,
            &[
                PreimageRelation { left_matrix: b0, preimage: k0, product: intermediate },
                PreimageRelation { left_matrix: b1, preimage: k1, product: final_product },
            ],
        )
        .expect("rewrite fixpoint");
        assert_eq!(rewritten, final_product);
        assert_eq!(factors(&arena, rewritten), vec![product_atom]);
    }

    #[test]
    fn targeted_preimage_rewrite_can_create_exact_cancellation() {
        let b = id(60);
        let k = id(61);
        let product_atom = id(62);
        let mut atoms = AtomTable::default();
        for atom in [&b, &k, &product_atom] {
            insert(&mut atoms, atom.clone(), AtomKind::Bounded);
        }
        let mut arena = SymbolicExprArena::default();
        let b_expr = arena.atom(b.clone(), &atoms).unwrap();
        let k_expr = arena.atom(k.clone(), &atoms).unwrap();
        let product = arena.atom(product_atom, &atoms).unwrap();
        let bk = arena.multiply(ty(), [b_expr, k_expr], &atoms).unwrap();
        let negative_product = arena.scale(BigInt::from(-1), product).unwrap();
        let root = arena.add(ty(), [bk, negative_product]).unwrap();
        let rewritten = rewrite_expression(
            root,
            &mut arena,
            &atoms,
            &[PreimageRelation { left_matrix: b, preimage: k, product }],
        )
        .expect("rewrite cancellation");
        assert!(matches!(arena.get(rewritten).unwrap().node, SymbolicExprNode::Zero));
    }

    #[test]
    fn aligned_block_rewrite_expands_without_an_aggregate_label() {
        let scalar = ty();
        let block = ConcreteMatrixType { rows: 2, columns: 1, ..ty() };
        let left_block = ConcreteMatrixType { rows: 1, columns: 2, ..ty() };
        let s = id(10);
        let identity = id(11);
        let a = id(12);
        let g = id(13);
        let error = id(14);
        let mut atoms = AtomTable::default();
        for (atom, matrix_type, kind, value) in [
            (
                s.clone(),
                scalar.clone(),
                AtomKind::Bounded,
                ConstantMatrix::UnitRow { index: 0.into() },
            ),
            (identity.clone(), scalar.clone(), AtomKind::Bounded, ConstantMatrix::Identity),
            (
                a.clone(),
                scalar.clone(),
                AtomKind::Large,
                ConstantMatrix::UnitRow { index: 0.into() },
            ),
            (
                g.clone(),
                scalar.clone(),
                AtomKind::Bounded,
                ConstantMatrix::UnitRow { index: 0.into() },
            ),
            (
                error.clone(),
                block.clone(),
                AtomKind::Bounded,
                ConstantMatrix::UnitRow { index: 0.into() },
            ),
        ] {
            atoms.insert(Atom {
                id: atom,
                class: AtomClass::Source { source: SourceKind::ConstantMatrix { value } },
                kind,
                matrix_type,
            });
        }
        let mut arena = SymbolicExprArena::default();
        let s_expr = arena.atom(s.clone(), &atoms).unwrap();
        let identity_expr = arena.atom(identity, &atoms).unwrap();
        let a_expr = arena.atom(a.clone(), &atoms).unwrap();
        let g_expr = arena.atom(g.clone(), &atoms).unwrap();
        let error_expr = arena.atom(error.clone(), &atoms).unwrap();
        let left =
            arena.concat(left_block, ConcatAxis::Columns, vec![s_expr, identity_expr]).unwrap();
        let sg = arena.multiply(scalar.clone(), [s_expr, g_expr], &atoms).unwrap();
        let negative_sg = arena.scale(BigInt::from(-1), sg).unwrap();
        let right_signal =
            arena.concat(block.clone(), ConcatAxis::Rows, vec![a_expr, negative_sg]).unwrap();
        let right = arena.add(block, [right_signal, error_expr]).unwrap();
        let root = arena.multiply(scalar, [left, right], &atoms).unwrap();
        let rewritten = rewrite_expression(root, &mut arena, &atoms, &[]).unwrap();
        let SymbolicExprNode::Add(alternatives) = &arena.get(rewritten).unwrap().node else {
            panic!("aligned block product must become an additive expression")
        };
        assert_eq!(alternatives.len(), 3);
        assert!(alternatives.iter().any(|alternative| {
            try_factors(&arena, *alternative) == Some(vec![s.clone(), a.clone()])
        }));
        assert!(alternatives.iter().any(|alternative| {
            try_factors(&arena, *alternative) == Some(vec![s.clone(), g.clone()])
        }));
        assert!(alternatives.iter().any(|alternative| {
            matches!(
                &arena.get(*alternative).unwrap().node,
                SymbolicExprNode::Mul(children)
                    if children.len() == 2 && children[0] == left && children[1] == error_expr
            )
        }));
    }
}
