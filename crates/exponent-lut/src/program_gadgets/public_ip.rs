//! Inner product of a public vector with a hash-sampled public binary vector.
//!
//! [`PublicInnerProductProgram`] fixes a public vector `a in Z_Q^m`, an
//! exponent layout, and a hash seed and label. The hash sampler derives the
//! binary vector `t in {0,1}^m` from the seed and label, and the Program
//! multiplies the public exponent encodings of the coordinates `a_i` with
//! `t_i = 1`. Multiplying exponent encodings adds their exponents, so the
//! Program output encodes `<a, t> mod Q`.
//!
//! The products form a balanced binary tree of public-by-public
//! multiplications of depth `ceil(log2(<t, t>))`. Every tree level is one
//! parallel family whose body is a single multiplication, so the Program has
//! `O(log m)` bodies instead of one gate per product. Only the first level
//! lists the selected coordinates, as a `Select` index expression.
//!
//! This module only assembles the Program and prepares its plaintext inputs;
//! public keys and encodings are computed by the generic backend evaluators.

use crate::{
    api::PlaintextInputs,
    program::{
        AttributeKnowledge, ExpLayout, ExponentLutProgram, ExponentLutProgramBuilder, FamilyRange,
        InputBinding, Operand, ProgramValidationError, WireId,
    },
};
use mxx_backends::{
    poly::{Poly, PolyParams},
    sampler::hash::sample_hash_integers,
};
use mxx_ir_core::IntExpr;
use num_bigint::BigUint;
use std::collections::BTreeMap;

/// Errors in a public inner-product declaration or its plaintext bindings.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum PublicIpError {
    #[error("invalid public inner-product layout, vector, or plaintext ring")]
    InvalidDeclaration,
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
}

/// The Program computing `<a, t> mod Q` over public exponent encodings.
///
/// The exponent layout fixes the ring dimension `n = 2^(stride_log2 +
/// width_log2)` and the exponent modulus `Q = 2^(width_log2 + 1)`; the
/// encoding of `x in Z_Q` is `X^(x * 2^stride_log2)`. The Program's only
/// input is one public-payload family of `m + 1` members: member `i < m` is
/// the encoding of `a_i` and member `m` is the encoding of zero. The zero
/// member pads a selection of fewer than two coordinates, so the output is
/// always the scalar root of at least one multiplication.
#[derive(Clone, Debug)]
pub struct PublicInnerProductProgram {
    program: ExponentLutProgram,
    layout: ExpLayout,
    vector: Vec<usize>,
    selection: Vec<bool>,
    input_vector: WireId,
    output: WireId,
}

impl PublicInnerProductProgram {
    /// Builds the Program for the public vector `vector = a`, whose entries
    /// must lie in `[0, Q)`, and the binary vector sampled from `seed` and
    /// `label` (see [`sample_selection`]).
    pub fn new(
        layout: ExpLayout,
        vector: Vec<usize>,
        seed: [u8; 32],
        label: &[u8],
    ) -> Result<Self, PublicIpError> {
        let q = exponent_modulus(layout)?;
        if vector.is_empty() || vector.iter().any(|value| *value >= q) {
            return Err(PublicIpError::InvalidDeclaration);
        }
        let dimension = vector.len();
        let selection = sample_selection(seed, label, dimension);
        let mut builder = ExponentLutProgramBuilder::new();
        let input_vector = builder
            .input_family(IntExpr::constant(dimension + 1), AttributeKnowledge::PublicPayload)?;
        let mut leaves = selection
            .iter()
            .enumerate()
            .filter_map(|(coordinate, selected)| selected.then_some(coordinate))
            .collect::<Vec<_>>();
        // Pad with the zero member `m`, the multiplicative identity.
        leaves.resize(leaves.len().max(2), dimension);
        let output = balanced_public_product(&mut builder, input_vector, &leaves)?;
        let program = builder.finish(vec![Operand::Scalar(output)])?;
        Ok(Self { program, layout, vector, selection, input_vector, output })
    }

    /// The immutable Program.
    pub const fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    pub const fn layout(&self) -> ExpLayout {
        self.layout
    }

    /// The public vector `a`.
    pub fn vector(&self) -> &[usize] {
        &self.vector
    }

    /// The sampled binary vector `t`.
    pub fn selection(&self) -> &[bool] {
        &self.selection
    }

    /// The public input family: `a` followed by the encoding of zero.
    pub const fn input_vector(&self) -> WireId {
        self.input_vector
    }

    pub const fn output(&self) -> WireId {
        self.output
    }

    /// The exponent modulus `Q`.
    pub fn q(&self) -> usize {
        exponent_modulus(self.layout).expect("layout validated at construction")
    }

    /// The clear inner product `<a, t> mod Q`.
    pub fn inner_product(&self) -> usize {
        let q = self.q();
        self.vector
            .iter()
            .zip(&self.selection)
            .filter(|(_, selected)| **selected)
            .fold(0, |sum, (value, _)| (sum + value) % q)
    }

    /// The exponent encoding `X^(x * 2^stride_log2)` of `x in Z_Q` over a ring
    /// of the layout's dimension, negated when the rotation wraps past `X^n`.
    pub fn encode<P: Poly>(&self, parameters: &P::Params, x: usize) -> Result<P, PublicIpError> {
        let dimension = 1usize << (self.layout.stride_log2 + self.layout.width_log2);
        if parameters.ring_dimension() as usize != dimension {
            return Err(PublicIpError::InvalidDeclaration);
        }
        let rotation = (x % self.q()) * self.layout.stride();
        let monomial = P::const_rotate_poly(parameters, rotation % dimension);
        Ok(if rotation >= dimension { -monomial } else { monomial })
    }

    /// The LHS plaintext binding of the input family: the encodings of `a`
    /// followed by the encoding of zero.
    pub fn plaintext_inputs<P: Poly>(
        &self,
        parameters: &P::Params,
    ) -> Result<PlaintextInputs<WireId, P>, PublicIpError> {
        let members = self
            .vector
            .iter()
            .copied()
            .chain(std::iter::once(0))
            .map(|value| self.encode(parameters, value))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BTreeMap::from([(self.input_vector, InputBinding::Family(members))]))
    }
}

/// Samples `t in {0,1}^m` with the hash sampler: `t_i` is hash integer `i`
/// modulo two under key `seed` and tag `label`.
pub fn sample_selection(seed: [u8; 32], label: &[u8], dimension: usize) -> Vec<bool> {
    sample_hash_integers::<keccak_asm::Keccak256>(seed, label, dimension, &BigUint::from(2u8))
        .into_iter()
        .map(|bit| bit == BigUint::from(1u8))
        .collect()
}

fn exponent_modulus(layout: ExpLayout) -> Result<usize, PublicIpError> {
    let ring_log2 = layout
        .stride_log2
        .checked_add(layout.width_log2)
        .ok_or(PublicIpError::InvalidDeclaration)?;
    let dimension = 1usize.checked_shl(ring_log2).ok_or(PublicIpError::InvalidDeclaration)?;
    layout.validate(dimension).map_err(|_| PublicIpError::InvalidDeclaration)?;
    let bits = layout.width_log2.checked_add(1).ok_or(PublicIpError::InvalidDeclaration)?;
    1usize.checked_shl(bits).ok_or(PublicIpError::InvalidDeclaration)
}

/// Multiplies members `leaves` (at least two) of the public family `source`
/// in a balanced tree and returns the scalar root.
///
/// A level holds `count` members of a family and at most one extra operand.
/// The level's pairs `(2j, 2j + 1)` are multiplied by one parallel family. An
/// odd last member becomes the extra operand, or is multiplied into an
/// existing one; both have depth at most the level's. Every level halves the
/// operand count rounding up, so the depth is `ceil(log2(leaves.len()))`.
/// The final two operands are multiplied outside a family to give a scalar.
fn balanced_public_product(
    builder: &mut ExponentLutProgramBuilder,
    source: WireId,
    leaves: &[usize],
) -> Result<WireId, ProgramValidationError> {
    // First-level member index of pair `slot`: `leaves[2 * slot + offset]`.
    let leaf_index = |offset: usize, slot: &IntExpr, pairs: usize| IntExpr::Select {
        selector: Box::new(slot.clone()),
        branches: (0..pairs).map(|pair| IntExpr::constant(leaves[2 * pair + offset])).collect(),
    };
    let mut family = source;
    let mut first = true;
    let mut count = leaves.len();
    let mut extra: Option<Operand<WireId>> = None;
    let member = |family: WireId, first: bool, position: usize| Operand::Member {
        family,
        index: IntExpr::constant(if first { leaves[position] } else { position }),
    };
    loop {
        let mut operands = (0..count).map(|position| member(family, first, position));
        match (count, extra.clone()) {
            (2, None) => {
                let (lhs, rhs) = (operands.next().unwrap(), operands.next().unwrap());
                return builder.mul_public(lhs, rhs);
            }
            (1, Some(rhs)) => return builder.mul_public(operands.next().unwrap(), rhs),
            _ => {}
        }
        let pairs = count / 2;
        if count % 2 == 1 {
            let last = member(family, first, count - 1);
            extra = Some(match extra {
                None => last,
                Some(rhs) => Operand::Scalar(builder.mul_public(last, rhs)?),
            });
        }
        let range = FamilyRange::full(IntExpr::constant(pairs))?;
        family = builder.parallel(range, |builder, slot| {
            let (lhs, rhs) = if first {
                (leaf_index(0, &slot, pairs), leaf_index(1, &slot, pairs))
            } else {
                (slot.clone() * 2usize, slot * 2usize + 1usize)
            };
            builder
                .mul_public(
                    Operand::Member { family, index: lhs },
                    Operand::Member { family, index: rhs },
                )
                .map(Operand::Scalar)
        })?;
        first = false;
        count = pairs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        program::{
            Environment, ExponentLUTProgramLowering, FamilyExecution, LoweringSite,
            ProgramLowerError, ProgramLoweringContext, lower_program,
        },
        rns_bgg::{
            RnsBggParameters, RnsEncryptBuild, RnsOnlineEvalBuild, RnsOutputUse,
            RnsPreprocessBuild, decode_public_output, verify_output,
        },
    };
    use mxx_backends::{
        artifact::MemoryArtifactStore,
        authority::CpuExecution,
        backend::poly::cpu_backend,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use mxx_ir_core::ParamEnv;

    /// n = 16, Q = 16.
    const LAYOUT: ExpLayout = ExpLayout { stride_log2: 1, width_log2: 3 };

    #[derive(Debug, thiserror::Error)]
    #[error("depth lowering error")]
    struct DepthError;

    /// Lowers a public product Program symbolically: every wire is its
    /// multiplication depth, and a family is the depth of its one body.
    #[derive(Default)]
    struct Depth {
        site: Option<LoweringSite>,
        bodies: usize,
    }

    impl ExponentLUTProgramLowering for Depth {
        type Wire = usize;
        type Rhs = ();
        type Error = DepthError;

        fn add(&mut self, _: &usize, _: &usize) -> Result<usize, DepthError> {
            Err(DepthError)
        }
        fn mul_public(&mut self, lhs: &usize, rhs: &usize) -> Result<usize, DepthError> {
            Ok(lhs.max(rhs) + 1)
        }
        fn mul_private(&mut self, _: &usize, _: &()) -> Result<usize, DepthError> {
            Err(DepthError)
        }
        fn lut(&mut self, _: &usize, _: &crate::program::LutTable) -> Result<usize, DepthError> {
            Err(DepthError)
        }
    }

    impl ProgramLoweringContext for Depth {
        fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
            std::mem::replace(&mut self.site, site)
        }
    }

    struct DepthFamilies;

    impl FamilyExecution<Depth> for DepthFamilies {
        type WireFamily = usize;
        type RhsFamily = ();

        fn wire_at(
            &mut self,
            family: &usize,
            _: &IntExpr,
        ) -> Result<usize, ProgramLowerError<DepthError>> {
            Ok(*family)
        }
        fn rhs_at(&mut self, _: &(), _: &IntExpr) -> Result<(), ProgramLowerError<DepthError>> {
            Err(ProgramLowerError::Backend(DepthError))
        }
        fn rhs_selected(
            &mut self,
            _: &(),
            _: crate::program::DataId,
            _: &IntExpr,
        ) -> Result<(), ProgramLowerError<DepthError>> {
            Err(ProgramLowerError::Backend(DepthError))
        }
        fn parallel(
            &mut self,
            evaluator: &mut Depth,
            _: &FamilyRange,
            body: impl FnOnce(
                &mut Self,
                &mut Depth,
                IntExpr,
            ) -> Result<usize, ProgramLowerError<DepthError>>,
        ) -> Result<usize, ProgramLowerError<DepthError>> {
            evaluator.bodies += 1;
            body(self, evaluator, IntExpr::LoopIndex(0))
        }
        fn sum(
            &mut self,
            _: &mut Depth,
            _: &usize,
        ) -> Result<usize, ProgramLowerError<DepthError>> {
            Err(ProgramLowerError::Backend(DepthError))
        }
    }

    /// The multiplication depth of the output and the number of Program
    /// family bodies.
    fn depth_and_bodies(public_ip: &PublicInnerProductProgram) -> (usize, usize) {
        let mut environment = Environment::new();
        environment.bind_wire(public_ip.input_vector(), InputBinding::Family(0)).unwrap();
        let mut depth = Depth::default();
        let outputs =
            lower_program(public_ip.program(), &mut depth, &mut DepthFamilies, &mut environment)
                .unwrap();
        (outputs[0], depth.bodies)
    }

    #[test]
    fn selection_is_deterministic_in_seed_and_label() {
        let selection = sample_selection([3; 32], b"label", 256);
        assert_eq!(selection, sample_selection([3; 32], b"label", 256));
        assert_ne!(selection, sample_selection([4; 32], b"label", 256));
        assert_ne!(selection, sample_selection([3; 32], b"other", 256));
        let ones = selection.iter().filter(|bit| **bit).count();
        assert!((64..192).contains(&ones), "bits must be roughly balanced: {ones}");
    }

    #[test]
    fn tree_depth_is_logarithmic_and_bodies_grow_only_per_level() {
        for dimension in [1, 2, 3, 5, 8, 17, 64, 255, 1000] {
            let vector = (0..dimension).map(|i| (3 * i + 1) % 16).collect::<Vec<_>>();
            let program =
                PublicInnerProductProgram::new(LAYOUT, vector, [dimension as u8; 32], b"depth")
                    .unwrap();
            let selected = program.selection().iter().filter(|bit| **bit).count().max(2);
            let (depth, bodies) = depth_and_bodies(&program);
            let levels = selected.next_power_of_two().trailing_zeros() as usize;
            assert_eq!(depth, levels, "m = {dimension}, selected = {selected}");
            assert_eq!(bodies, levels - 1, "one family per level below the scalar root");
        }
    }

    #[test]
    fn rejects_out_of_range_vectors_and_layouts() {
        assert!(PublicInnerProductProgram::new(LAYOUT, vec![], [0; 32], b"x").is_err());
        assert!(PublicInnerProductProgram::new(LAYOUT, vec![1, 16], [0; 32], b"x").is_err());
        let bad = ExpLayout { stride_log2: 4, width_log2: 0 };
        assert!(PublicInnerProductProgram::new(bad, vec![0], [0; 32], b"x").is_err());
    }

    /// Preprocesses, encrypts, and evaluates the Program on the CPU, then
    /// verifies that the output encodes `<a, t> mod Q` and not its successor.
    fn check_encoding_inner_product(dimension: usize, seed: [u8; 32]) {
        let vector = (0..dimension).map(|i| (5 * i + 3) % 16).collect::<Vec<_>>();
        let public_ip = PublicInnerProductProgram::new(LAYOUT, vector, seed, b"public-ip").unwrap();
        let program = public_ip.program();
        let level = DCRTPolyParams::new(16, 4, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let mut execution = CpuExecution::new(cpu_backend([level.clone()]));
        let mut store = MemoryArtifactStore::default();
        let hash_key: [u8; 32] = rand::random();
        let context: [u8; 32] = rand::random();
        let payload = RnsEncryptBuild::new(&parameters, context)
            .sample_payload_secret(&mut execution, &mut store, |_| Ok(()))
            .unwrap();
        let (preprocess, online_eval) = crate::api::preprocess(
            program,
            &ParamEnv::default(),
            &PlaintextInputs::<_, DCRTPoly>::new(),
            Some(&payload),
            &hash_key,
            RnsPreprocessBuild::new(&parameters, context),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        let mask = RnsEncryptBuild::new(&parameters, rand::random())
            .sample_mask_secret(&mut execution, &mut store, |_| Ok(()))
            .unwrap();
        let lhs = public_ip.plaintext_inputs::<DCRTPoly>(&level).unwrap();
        let encrypted = crate::api::encrypt(
            &program.encryption_inputs(),
            &ParamEnv::default(),
            &lhs,
            Some(&preprocess),
            &payload,
            Some(&mask),
            &hash_key,
            RnsEncryptBuild::new(&parameters, context),
            &mut execution,
            &mut store,
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let result = crate::api::online_eval(
            program,
            &ParamEnv::default(),
            &online_eval,
            &encrypted,
            &hash_key,
            RnsOnlineEvalBuild::new(&parameters, context),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        let inner_product = public_ip.inner_product();
        let expected = public_ip.encode::<DCRTPoly>(&level, inner_product).unwrap();
        verify_output(
            program,
            &result,
            0,
            &expected,
            &parameters,
            &payload,
            Some(&mask),
            RnsOutputUse::DirectDecode,
            &mut execution,
            &mut store,
        )
        .unwrap();
        let wrong = public_ip.encode::<DCRTPoly>(&level, inner_product + 1).unwrap();
        assert!(
            verify_output(
                program,
                &result,
                0,
                &wrong,
                &parameters,
                &payload,
                Some(&mask),
                RnsOutputUse::DirectDecode,
                &mut execution,
                &mut store,
            )
            .is_err(),
            "a wrong inner product must be rejected"
        );
        let decode =
            |expected: &DCRTPoly, execution: &mut CpuExecution, store: &mut MemoryArtifactStore| {
                decode_public_output(
                    program,
                    &result,
                    0,
                    &parameters,
                    &payload,
                    Some(&mask),
                    expected,
                    execution,
                    store,
                    |_| Ok(()),
                )
                .unwrap()
            };
        assert_eq!(decode(&expected, &mut execution, &mut store), 0.into());
        assert_ne!(decode(&wrong, &mut execution, &mut store), 0.into());
    }

    #[test]
    fn encodings_evaluate_the_inner_product_with_a_sampled_selection() {
        // Search seeds for selections of every tree shape: padded singleton,
        // odd carries, and an even power of two.
        let mut covered = std::collections::BTreeSet::new();
        for seed in 0..=u8::MAX {
            let selected = sample_selection([seed; 32], b"public-ip", 9)
                .into_iter()
                .filter(|bit| *bit)
                .count();
            if [1, 4, 7].contains(&selected) && covered.insert(selected) {
                check_encoding_inner_product(9, [seed; 32]);
            }
            if covered.len() == 3 {
                return;
            }
        }
        panic!("seeds must cover every tree shape");
    }
}
