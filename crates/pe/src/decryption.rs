//! Decryption: `Dec(sk_F, F, ct)`.
//!
//! The predicate is evaluated on the ciphertext's packages by the
//! exponent-lut Dynamic online evaluation with the gadget output use, giving
//! `c_F = s (A_F - F(x) G) + e_F`. The family values and constants
//!
//! `y_F = ((sigma_i(b))_i, (sigma_i(h_k))_(i,k), c_F)`,
//! `kappa_F = ((sigma_i(B^(2)))_i, (sigma_i(H_k^(2)) - G^(2))_(i,k), A_F^(2))`
//!
//! satisfy `y_F - kappa_F = s_bar_H M_F - F(x) (0, s G) + e_F^family`, so
//! with `M_F k = d_1 delta_1` the phase
//!
//! `c_m - d_2 - (y_F - kappa_F) k = floor(q/2) m + e_m - e_F^family k`
//!
//! when `F(x) = 0`; its coefficients are rounded to bits. The contraction is
//! accumulated by blocks: the anchor blocks as `sigma_i((b - B^(2)) k~_i)`
//! with the stored inverse-coordinate blocks, the helper blocks against
//! their extension coordinates, and the output block. Decryption reads no
//! secret: not `s`, the trapdoor, or any `Z_t`. For `F(x) != 0` the phase
//! keeps the term `-F(x) (s G)_2 v_F`, and the rounded output is not a
//! reliable rejection.
use crate::{
    PeCiphertext, PeError, PePredicateKey, PePublicKey,
    encryption::{ANCHOR_ENCODING, MESSAGE_ENCODING},
    keygen::{Group, HELPER_KEY, OUTPUT_KEY, anchor_key},
};
use mxx_backends::{RuntimeValue, SessionStore, poly::PolyParams};
use mxx_dsl::{DslContext, Int, Mat};
use mxx_exponent_lut::{
    api::{self, StageExecution, StageGraph, StageOutputs},
    rns_bgg::{RnsBggDynamicSetup, RnsOnlineEvalBuild, RnsOutputUse},
};
use mxx_ir_core::{IntExpr, ParamEnv, artifact::ArtifactAvailability, node::ConcatAxis};
use std::collections::BTreeMap;

const OUTPUT_ROW: &str = "mxx-pe/output-row";

/// Evaluate the key's predicate on the ciphertext: the gadget carrier `c_F`
/// and its key, as online evaluation outputs.
pub(crate) fn evaluate<E, S>(
    public_key: &PePublicKey,
    key: &PePredicateKey,
    ciphertext: &PeCiphertext,
    execution: &mut E,
    store: &mut S,
) -> Result<StageOutputs, PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    if key.public_key() != public_key.identity() || ciphertext.public_key() != public_key.identity()
    {
        return Err(PeError::Mismatch);
    }
    let parameters = public_key.parameters();
    Ok(api::online_eval(
        key.compiled().program(),
        &ParamEnv::default(),
        key.evaluation(),
        ciphertext.packages(),
        public_key.hash_key(),
        RnsOnlineEvalBuild::new(parameters.rns(), *public_key.context())
            .with_output_use(RnsOutputUse::Gadget)
            .with_dynamic_setup(public_key.setup(), key.compiled().binding().clone())?,
        execution,
        store,
        |_, outputs| Ok(outputs),
        |_| Ok(()),
    )?)
}

/// `Dec(sk_F, F, ct)`: the bits of the rounded phase, one per coefficient.
pub fn decrypt<E, S>(
    public_key: &PePublicKey,
    key: &PePredicateKey,
    ciphertext: &PeCiphertext,
    execution: &mut E,
    store: &mut S,
) -> Result<Vec<bool>, PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let evaluated = evaluate(public_key, key, ciphertext, execution, store)?;
    decode(public_key, key, ciphertext, &evaluated, execution, store)
}

/// The bits of the rounded phase of an evaluated predicate.
fn decode<E, S>(
    public_key: &PePublicKey,
    key: &PePredicateKey,
    ciphertext: &PeCiphertext,
    evaluated: &StageOutputs,
    execution: &mut E,
    store: &mut S,
) -> Result<Vec<bool>, PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let (graph, runtime, manifests) = phase_graph(public_key, key, ciphertext, evaluated)?;
    let dimension = public_key.parameters().level().ring_dimension() as usize;
    let mut decoded = DslContext::new("mxx-pe/decrypt");
    for (index, bit) in graph.threshold_decode_ints(2u8, dimension).into_iter().enumerate() {
        decoded = decoded.output(format!("mxx-pe/message/{index}"), bit)?;
    }
    let outputs = api::execute_graph(
        StageGraph::new(decoded.build()?, runtime, manifests),
        &ParamEnv::default(),
        execution,
        store,
        |_| Ok(()),
    )?;
    (0..dimension)
        .map(|index| {
            let value =
                outputs.outputs.get(&format!("mxx-pe/message/{index}")).ok_or(PeError::Mismatch)?;
            let [bit] = <[num_bigint::BigInt; 1]>::try_from(execution.download_integers(value)?)
                .map_err(|_| PeError::Mismatch)?;
            Ok(bit == num_bigint::BigInt::from(1u8))
        })
        .collect()
}

/// The phase `c_m - d_2 - (y_F - kappa_F) k` as a graph value, with its
/// runtime inputs and the manifests of the productions it reads.
#[allow(clippy::type_complexity)]
pub(crate) fn phase_graph(
    public_key: &PePublicKey,
    key: &PePredicateKey,
    ciphertext: &PeCiphertext,
    evaluated: &StageOutputs,
) -> Result<
    (
        Mat,
        BTreeMap<String, RuntimeValue>,
        BTreeMap<mxx_ir_core::artifact::ProductionId, mxx_ir_core::artifact::Manifest>,
    ),
    PeError,
> {
    let parameters = public_key.parameters();
    let rns = parameters.rns();
    let ring = public_key.ring()?;
    let columns = parameters.digits();
    let width = 2 * columns;
    let m_b = parameters.anchor_columns();
    let group = Group::of(public_key)?;
    let w = group.width();
    let mut runtime = BTreeMap::new();
    let hash_key = public_key.hash_key_input(&mut runtime)?;
    for port in [4, 5] {
        let value =
            evaluated.outputs.get(&format!("output-0.{port}")).ok_or(PeError::Mismatch)?.clone();
        runtime.insert(format!("{OUTPUT_ROW}.{port}"), value);
    }
    let output_row = Mat::concat(
        ConcatAxis::Columns,
        vec![
            ring.input(format!("{OUTPUT_ROW}.4"), (1, columns)),
            ring.input(format!("{OUTPUT_ROW}.5"), (1, columns)),
        ],
    );
    let output = key.output()?;
    let evaluation = key.evaluation().production().clone();
    let output_key = |port: usize| {
        ring.artifact_input(
            evaluation.clone(),
            format!("public-key/{output:?}.{port}"),
            (1, columns),
            ArtifactAvailability::Transferred,
        )
    };
    let output_second_row = Mat::concat(ConcatAxis::Columns, vec![output_key(2), output_key(3)]);
    let prelude = ciphertext.prelude().production().clone();
    let anchor = ring.artifact_input(
        prelude.clone(),
        ANCHOR_ENCODING,
        (1, m_b),
        ArtifactAvailability::Transferred,
    );
    let message =
        ring.artifact_input(prelude, MESSAGE_ENCODING, (1, 1), ArtifactAvailability::Transferred);
    let second = public_key.anchor_second_row(hash_key.clone())?;
    let (_, d_2) = public_key.message_row(hash_key.clone())?;
    let key_production = key.key().production().clone();
    let helper_keys = ring.family_artifact_input(
        key_production.clone(),
        HELPER_KEY,
        w * (w - 1),
        (width, 1),
        ArtifactAvailability::Transferred,
    );
    let output_key_block = ring.artifact_input(
        key_production,
        OUTPUT_KEY,
        (width, 1),
        ArtifactAvailability::Transferred,
    );
    let packages = ciphertext.packages().production().clone();
    let helper_rows = [0, 1].map(|port| {
        ring.family_artifact_input(
            packages.clone(),
            format!("{}.{port}", RnsBggDynamicSetup::automorphism_helper_role()),
            w - 1,
            (1, columns),
            ArtifactAvailability::Transferred,
        )
    });
    let gadget_second_row =
        Mat::concat(ConcatAxis::Columns, vec![ring.zero((1, columns)), rns.public_gadget(0)?]);
    let setup = public_key.setup();
    let masked = anchor - second;
    let anchors = group.elements.iter().enumerate().fold(ring.zero((1, 1)), |sum, (row, i)| {
        sum + masked
            .clone()
            .mul_small_rhs(anchor_key(&ring, parameters, key, row))
            .ring_automorphism(IntExpr::constant(*i))
    });
    let helpers = mxx_dsl::iterate(w * (w - 1), ring.zero((1, 1)), |pair, sum| {
        let pair = pair.expression()?;
        let quotient =
            IntExpr::FloorDiv(Box::new(pair.clone()), Box::new(IntExpr::constant(w - 1)))
                .canonicalize();
        let k_index =
            IntExpr::Rem(Box::new(pair.clone()), Box::new(IntExpr::constant(w - 1))).canonicalize();
        let i = Group::select(&group.elements, quotient);
        let k = Group::select(&group.elements[1..], k_index.clone());
        let public =
            setup.helper_key(rns, hash_key.clone(), k).map_err(|_| mxx_dsl::DslError::Schema)?;
        let member = Int::evaluate(k_index);
        let helper = Mat::concat(
            ConcatAxis::Columns,
            vec![helper_rows[0].at(member.clone()), helper_rows[1].at(member)],
        );
        let public_second_row = Mat::concat(ConcatAxis::Columns, vec![public.1.0, public.1.1]);
        let value = (helper - public_second_row).ring_automorphism(i) + gadget_second_row.clone();
        Ok(sum + value * helper_keys.at(Int::evaluate(pair)))
    })?;
    let outputs = (output_row - output_second_row) * output_key_block;
    let phase = message - d_2 - (anchors + helpers + outputs);
    let mut manifests = BTreeMap::new();
    for (production, manifest) in [
        (ciphertext.prelude().production(), ciphertext.prelude().manifest()),
        (ciphertext.packages().production(), ciphertext.packages().manifest()),
        (key.key().production(), key.key().manifest()),
        (key.evaluation().production(), key.evaluation().manifest()),
    ] {
        manifests.insert(production.clone(), manifest.clone());
    }
    Ok((phase, runtime, manifests))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BranchingProgram, PeClass, encryption::encrypt, keygen::keygen, setup::setup,
        test_support::*,
    };
    use mxx_backends::{MemoryArtifactStore, authority::CpuExecution};
    use num_bigint::BigUint;

    fn bits(value: usize, length: usize) -> Vec<usize> {
        (0..length).rev().map(|bit| value >> bit & 1).collect()
    }

    /// The largest centered coefficient of `phase - floor(q/2) m`: the
    /// measured decryption error.
    fn phase_error(
        public_key: &PePublicKey,
        key: &PePredicateKey,
        ciphertext: &PeCiphertext,
        evaluated: &StageOutputs,
        message: &[bool],
        execution: &mut CpuExecution,
        store: &mut MemoryArtifactStore,
    ) -> BigUint {
        use mxx_backends::poly::Poly;
        let (phase, runtime, manifests) =
            phase_graph(public_key, key, ciphertext, evaluated).unwrap();
        let graph = DslContext::new("mxx-pe/test-phase").output("phase", phase).unwrap();
        let outputs = api::execute_graph(
            StageGraph::new(graph.build().unwrap(), runtime, manifests),
            &ParamEnv::default(),
            execution,
            store,
            |_| Ok(()),
        )
        .unwrap();
        let phase = download(execution, &outputs, "phase").entry(0, 0);
        let q = public_key.parameters().modulus();
        let half = &q / 2u8;
        phase
            .coeffs_biguints()
            .into_iter()
            .zip(message.iter().copied().chain(std::iter::repeat(false)))
            .map(|(coefficient, bit)| {
                let shifted = (coefficient + &q - if bit { half.clone() } else { 0u8.into() }) % &q;
                shifted.clone().min(&q - shifted)
            })
            .max()
            .unwrap()
    }

    fn download(
        execution: &CpuExecution,
        outputs: &StageOutputs,
        name: &str,
    ) -> mxx_backends::matrix::dcrt_poly::DCRTPolyMatrix {
        <_ as StageExecution<MemoryArtifactStore>>::download_matrix(
            execution,
            &outputs.outputs[name],
        )
        .unwrap()
    }

    /// The largest centered coefficient of a downloaded row.
    fn norm(q: &BigUint, row: &mxx_backends::matrix::dcrt_poly::DCRTPolyMatrix) -> BigUint {
        use mxx_backends::{matrix::PolyMatrix, poly::Poly};
        (0..row.col_size())
            .flat_map(|column| row.entry(0, column).coeffs_biguints())
            .map(|coefficient| coefficient.clone().min(q - &coefficient))
            .max()
            .unwrap()
    }

    /// With the encryption secret, the family values satisfy
    /// `y_F - kappa_F = s_bar_H M_F - F(x) (0, s G) + e`, block by block:
    /// each anchor block `sigma_i(b - B^(2)) - sigma_i(s_bar) sigma_i(B^(1))`
    /// is within the error cutoff, each helper block
    /// `sigma_i(h_k - H_k^(2)) + G^(2) - sigma_i(s_bar) sigma_i(H_k^(1)) +
    /// sigma_(ik)(s_bar) G^(1)` within the helper cutoff, and the output
    /// block `c_F - A_F^(2) - s_bar A_F^(1) + F(x) s G` within the simulated
    /// `E_F`. One ciphertext of the all-one message serves an authorized
    /// predicate that reads attribute 0 twice (`F = 0` everywhere), whose
    /// key decrypts it, and an unauthorized one (`F(x) = 1`), whose output
    /// keeps the predicate term `-s G`. With the default noise-free test
    /// parameters, every cutoff and bound is zero, so each relation holds
    /// exactly, and the simulated bound guarantees correctness.
    #[test]
    fn family_values_open_to_the_secret_times_the_family_matrix() {
        let class = PeClass::new(2, 2, 2, 2).unwrap();
        let parameters = parameters(class);
        let mut execution = execution(&parameters);
        let mut store = MemoryArtifactStore::default();
        let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
        let dimension = parameters.level().ring_dimension() as usize;
        let attribute = bits(rand::random::<u32>() as usize % 4, 2);
        let message = vec![true; dimension];
        let (ciphertext, secret) = crate::encryption::encrypt_with_secret(
            &public_key,
            &attribute,
            &message,
            &mut execution,
            &mut store,
        )
        .unwrap();
        // `z_1 = x_0`, then `F = z_1 xor x_0`.
        let repeated =
            BranchingProgram::new(class, vec![0, 0], vec![vec![0, 0, 1, 1], vec![0, 1, 1, 0]])
                .unwrap();
        let complement = attribute.iter().map(|bit| 1 - bit).collect::<Vec<_>>();
        let bound = crate::noise::simulate(&parameters, &BranchingProgram::longest(class)).unwrap();
        assert!(bound.decrypts(), "the parameters must guarantee correctness");
        let q = parameters.modulus();
        let ring = public_key.ring().unwrap();
        let rns = parameters.rns();
        let columns = parameters.digits();
        let m_b = parameters.anchor_columns();
        let group = Group::of(&public_key).unwrap();
        let w = group.width();
        let order = 2 * dimension;
        let gadget = rns.public_gadget(0).unwrap();
        let zero = ring.zero((1, columns));
        let gadget_first_row = Mat::concat(ConcatAxis::Columns, vec![gadget.clone(), zero.clone()]);
        let gadget_second_row =
            Mat::concat(ConcatAxis::Columns, vec![zero.clone(), gadget.clone()]);
        for (predicate, value) in [(repeated, 0), (equals(class, &complement), 1)] {
            assert_eq!(predicate.evaluate(&attribute).unwrap(), value);
            let key = keygen(&public_key, &master, &predicate, &mut execution, &mut store).unwrap();
            let evaluated =
                evaluate(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap();
            if value == 0 {
                let decrypted =
                    decode(&public_key, &key, &ciphertext, &evaluated, &mut execution, &mut store)
                        .unwrap();
                assert_eq!(decrypted, message);
            }
            let mut runtime = BTreeMap::new();
            secret.bind_runtime(&mut runtime);
            let hash_key = public_key.hash_key_input(&mut runtime).unwrap();
            for port in [4, 5] {
                runtime.insert(
                    format!("{OUTPUT_ROW}.{port}"),
                    evaluated.outputs[&format!("output-0.{port}")].clone(),
                );
            }
            let s_bar = secret.row().0.clone();
            let sigma = |value: Mat, i: usize| value.ring_automorphism(IntExpr::constant(i));
            let prelude = ciphertext.prelude().production().clone();
            let anchor = ring.artifact_input(
                prelude,
                ANCHOR_ENCODING,
                (1, m_b),
                ArtifactAvailability::Transferred,
            );
            let first = public_key.anchor_row().unwrap();
            let second = public_key.anchor_second_row(hash_key.clone()).unwrap();
            let packages = ciphertext.packages().production().clone();
            let helper_rows = [0, 1].map(|port| {
                ring.family_artifact_input(
                    packages.clone(),
                    format!("{}.{port}", RnsBggDynamicSetup::automorphism_helper_role()),
                    w - 1,
                    (1, columns),
                    ArtifactAvailability::Transferred,
                )
            });
            let mut graph = DslContext::new("mxx-pe/test-family");
            for i in &group.elements {
                let anchor_block = sigma(anchor.clone() - second.clone(), *i) -
                    sigma(s_bar.clone(), *i) * sigma(first.clone(), *i);
                graph = graph.output(format!("anchor-{i}"), anchor_block).unwrap();
                for (member, k) in group.elements[1..].iter().enumerate() {
                    let public = public_key
                        .setup()
                        .helper_key(rns, hash_key.clone(), IntExpr::constant(*k))
                        .unwrap();
                    let member = Int::constant(member);
                    let helper = Mat::concat(
                        ConcatAxis::Columns,
                        vec![helper_rows[0].at(member.clone()), helper_rows[1].at(member)],
                    );
                    let public_first =
                        Mat::concat(ConcatAxis::Columns, vec![public.0.0, public.0.1]);
                    let public_second =
                        Mat::concat(ConcatAxis::Columns, vec![public.1.0, public.1.1]);
                    let helper_block = sigma(helper - public_second, *i) +
                        gadget_second_row.clone() -
                        sigma(s_bar.clone(), *i) * sigma(public_first, *i) +
                        sigma(s_bar.clone(), i * k % order) * gadget_first_row.clone();
                    graph = graph.output(format!("helper-{i}-{k}"), helper_block).unwrap();
                }
            }
            let output = key.output().unwrap();
            let evaluation = key.evaluation().production().clone();
            let output_key = |port: usize| {
                ring.artifact_input(
                    evaluation.clone(),
                    format!("public-key/{output:?}.{port}"),
                    (1, columns),
                    ArtifactAvailability::Transferred,
                )
            };
            let output_row = Mat::concat(
                ConcatAxis::Columns,
                [4, 5]
                    .map(|port| ring.input(format!("{OUTPUT_ROW}.{port}"), (1, columns)))
                    .to_vec(),
            );
            let output_first = Mat::concat(ConcatAxis::Columns, vec![output_key(0), output_key(1)]);
            let output_second =
                Mat::concat(ConcatAxis::Columns, vec![output_key(2), output_key(3)]);
            let mut output_block = output_row - output_second - s_bar.clone() * output_first;
            if value == 1 {
                output_block = output_block +
                    s_bar.clone() * gadget_first_row.clone() +
                    gadget_second_row.clone();
            }
            graph = graph.output("output", output_block).unwrap();
            let mut manifests = BTreeMap::from([public_key.anchor_production().clone()]);
            for (production, manifest) in [
                (ciphertext.prelude().production(), ciphertext.prelude().manifest()),
                (ciphertext.packages().production(), ciphertext.packages().manifest()),
                (key.evaluation().production(), key.evaluation().manifest()),
            ] {
                manifests.insert(production.clone(), manifest.clone());
            }
            let outputs = api::execute_graph(
                StageGraph::new(graph.build().unwrap(), runtime, manifests),
                &ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let error = parameters.error().cutoff();
            let helper_cutoff = parameters.distributions().helper.cutoff().clone();
            for i in &group.elements {
                let measured = norm(&q, &download(&execution, &outputs, &format!("anchor-{i}")));
                assert!(&measured <= error, "anchor {i}: {measured} > {error}");
                for k in &group.elements[1..] {
                    let measured =
                        norm(&q, &download(&execution, &outputs, &format!("helper-{i}-{k}")));
                    assert!(measured <= helper_cutoff, "helper ({i}, {k}): {measured}");
                }
            }
            let measured = norm(&q, &download(&execution, &outputs, "output"));
            assert!(measured <= bound.evaluation, "output, F(x) = {value}: {measured}");
        }
    }

    /// One setup, ciphertexts of a random and of the all-zero message
    /// encrypted before any key exists, and keys for every equality
    /// predicate of the class: the parameters guarantee correctness
    /// (`4 E_PE + 2 < q`), the one authorized key (`F(x) = 0`) decrypts
    /// both, and its measured phase error is within the simulated bound
    /// `E_PE`; with the default noise-free test parameters it is exactly
    /// zero. Unauthorized keys return the rounded phase, which this test does
    /// not constrain.
    #[test]
    fn authorized_keys_of_several_predicates_decrypt_one_ciphertext() {
        let class = PeClass::new(2, 2, 2, 2).unwrap();
        let parameters = parameters(class);
        let mut execution = execution(&parameters);
        let mut store = MemoryArtifactStore::default();
        let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
        let dimension = parameters.level().ring_dimension() as usize;
        let value = rand::random::<u32>() as usize % 4;
        let attribute = bits(value, 2);
        let message = (0..dimension).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
        let zeros = vec![false; dimension];
        let [ciphertext, zero_ciphertext] = [&message, &zeros].map(|message| {
            encrypt(&public_key, &attribute, message, &mut execution, &mut store).unwrap()
        });
        let bound = crate::noise::simulate(&parameters, &BranchingProgram::longest(class)).unwrap();
        assert!(bound.decrypts(), "the parameters must guarantee correctness");
        let mut authorized = 0;
        for pattern in 0..4 {
            let predicate = equals(class, &bits(pattern, 2));
            let key = keygen(&public_key, &master, &predicate, &mut execution, &mut store).unwrap();
            if predicate.evaluate(&attribute).unwrap() != 0 {
                decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap();
                continue;
            }
            for (ciphertext, message) in [(&ciphertext, &message), (&zero_ciphertext, &zeros)] {
                let evaluated =
                    evaluate(&public_key, &key, ciphertext, &mut execution, &mut store).unwrap();
                let decrypted =
                    decode(&public_key, &key, ciphertext, &evaluated, &mut execution, &mut store)
                        .unwrap();
                assert_eq!(&decrypted, message, "pattern {pattern}, value {value}");
                let measured = phase_error(
                    &public_key,
                    &key,
                    ciphertext,
                    &evaluated,
                    message,
                    &mut execution,
                    &mut store,
                );
                assert!(measured <= bound.decryption, "{measured} > {}", bound.decryption);
            }
            authorized += 1;
        }
        assert_eq!(authorized, 1);
    }
}
