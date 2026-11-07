//! Inner product of a public vector with a hash-sampled public binary vector.
//!
//! [`PublicInnerProductProgram`] fixes a public vector `a in Z_Q^m`, an
//! exponent layout, and a hash seed and label. The hash sampler derives the
//! binary vector `t in {0,1}^m` from the seed and label, and the Program
//! multiplies the public exponent encodings of the coordinates `a_i` with
//! `t_i = 1`. Multiplying exponent encodings adds their exponents, so the
//! Program output encodes `<a, t> mod Q`.
//!
//! Products form a right-deep chain. Public multiplication propagates errors
//! asymmetrically: the left error receives a gadget action, while the right
//! error is multiplied by the left plaintext. For these monomial plaintexts,
//! keeping the accumulated product on the right avoids repeated gadget gains.
//! The Program has linear depth and size; DSL callers use the same schedule
//! with a loop rather than unrolling the chain.
//!
//! This module only assembles the Program and prepares its plaintext inputs;
//! public keys and encodings are computed by the generic backend evaluators.

use crate::{
    api::PlaintextInputs,
    program::{
        AttributeKnowledge, ExpLayout, ExponentLutProgram, ExponentLutProgramBuilder, InputBinding,
        Operand, ProgramValidationError, WireId,
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
        let output = public_product(
            &mut builder,
            leaves
                .iter()
                .map(|index| Operand::Member {
                    family: input_vector,
                    index: IntExpr::constant(*index),
                })
                .collect(),
            Operand::Member { family: input_vector, index: IntExpr::constant(dimension) },
        )?;
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

/// Declare a right-deep public product, padding zero/one operands with `one`.
/// The accumulated product stays on the right of every multiplication.
pub fn public_product(
    builder: &mut ExponentLutProgramBuilder,
    mut leaves: Vec<Operand<WireId>>,
    one: Operand<WireId>,
) -> Result<WireId, ProgramValidationError> {
    leaves.resize(leaves.len().max(2), one);
    let builder = std::cell::RefCell::new(builder);
    let product = right_deep_product(
        leaves.len(),
        |position| {
            Ok(leaves[usize::try_from(
                position.cancelled_constant().expect("static product position"),
            )
            .expect("nonnegative product position")]
            .clone())
        },
        |left, right| builder.borrow_mut().mul_public(left, right).map(Operand::Scalar),
        |count, mut product, step| {
            for iteration in 0..count {
                product = step(IntExpr::constant(iteration), product)?;
            }
            Ok(product)
        },
    )?;
    let Operand::Scalar(product) = product else { unreachable!("the root is a multiplication") };
    Ok(product)
}

/// Shared right-deep reduction schedule for Program nodes and runtime DSL loops.
///
/// `leaf` reads an ordered operand. `fold` performs `count - 1` iterations of
/// the supplied body, passing its result to the next iteration. Program callers
/// unroll this fold; DSL callers emit one sequential loop. This function fixes
/// both the descending leaf order and which operand receives the gadget action.
/// Runtime-prefix callers can attach an active flag to operands and bypass the
/// multiplication when the accumulated suffix is inactive.
///
/// Panics if `count < 2`; callers must pad with the one-encoding first.
pub fn right_deep_product<V, E>(
    count: usize,
    mut leaf: impl FnMut(IntExpr) -> Result<V, E>,
    mut multiply: impl FnMut(V, V) -> Result<V, E>,
    fold: impl FnOnce(usize, V, &mut dyn FnMut(IntExpr, V) -> Result<V, E>) -> Result<V, E>,
) -> Result<V, E> {
    assert!(count >= 2, "a public product requires at least two operands");
    let last = leaf(IntExpr::constant(count - 1))?;
    fold(count - 1, last, &mut |iteration, product| {
        let position = (IntExpr::constant(count - 2) - iteration).canonicalize();
        let left = leaf(position)?;
        multiply(left, product)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        program::{
            Environment, ExponentLUTProgramLowering, FamilyExecution, FamilyRange, LoweringSite,
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
        fn mul_small_constant(
            &mut self,
            _: &usize,
            _: &[i64],
            _: &IntExpr,
        ) -> Result<usize, DepthError> {
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

        fn wire_selected(
            &mut self,
            _evaluator: &mut Depth,
            family: &usize,
            _: crate::program::DataId,
            _: &IntExpr,
        ) -> Result<usize, ProgramLowerError<DepthError>> {
            Ok(*family)
        }

        fn wire_at(
            &mut self,
            _evaluator: &mut Depth,
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
            _: usize,
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

    /// Verify operand order and the asymmetric error recurrence independently of depth.
    #[test]
    fn right_deep_order_and_error_gain() {
        for count in 2usize..=32 {
            let (order, error) = right_deep_product(
                count,
                |position| {
                    Ok::<_, ()>((position.cancelled_constant().unwrap().to_string(), 1usize))
                },
                |(left, le), (right, re)| Ok((format!("({left},{right})"), 8 * le + re)),
                |iterations, mut value, step| {
                    for i in 0..iterations {
                        value = step(IntExpr::constant(i), value)?;
                    }
                    Ok(value)
                },
            )
            .unwrap();
            let mut expected = (count - 1).to_string();
            for i in (0..count - 1).rev() {
                expected = format!("({i},{expected})");
            }
            assert_eq!(order, expected);
            assert_eq!(error, 8 * (count - 1) + 1);
        }
    }

    /// Bypassing runtime-inactive suffixes must preserve the compact tree, including
    /// zero and one selected operand (which both need two-leaf padding).
    #[test]
    fn right_deep_runtime_prefix_matches_compact() {
        for selected in 0usize..=32 {
            let padded = selected.max(2);
            let capacity = 32;
            let leaf = |position: IntExpr| {
                let i = usize::try_from(position.cancelled_constant().unwrap()).unwrap();
                (if i < selected { i.to_string() } else { "one".to_owned() }, i < padded)
            };
            let fold = |count,
                        mut value,
                        step: &mut dyn FnMut(
                IntExpr,
                (String, bool),
            ) -> Result<(String, bool), ()>| {
                for i in 0..count {
                    value = step(IntExpr::constant(i), value)?;
                }
                Ok(value)
            };
            let runtime = right_deep_product(
                capacity,
                |i| Ok(leaf(i)),
                |a, b| Ok((if b.1 { format!("({},{})", a.0, b.0) } else { a.0 }, a.1)),
                fold,
            )
            .unwrap();
            let compact = right_deep_product(
                padded,
                |i| Ok(leaf(i)),
                |a, b| Ok((format!("({},{})", a.0, b.0), true)),
                fold,
            )
            .unwrap();
            assert_eq!(runtime, compact, "selected={selected}");
        }
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
    fn product_depth_is_linear_and_accumulator_stays_on_right() {
        for dimension in [1, 2, 3, 5, 8, 17, 64, 255, 1000] {
            let vector = (0..dimension).map(|i| (3 * i + 1) % 16).collect::<Vec<_>>();
            let program =
                PublicInnerProductProgram::new(LAYOUT, vector, [dimension as u8; 32], b"depth")
                    .unwrap();
            let selected = program.selection().iter().filter(|bit| **bit).count().max(2);
            let (depth, bodies) = depth_and_bodies(&program);
            assert_eq!(depth, selected - 1, "m = {dimension}, selected = {selected}");
            assert_eq!(bodies, 0, "the Program declares a scalar right-deep chain");
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
