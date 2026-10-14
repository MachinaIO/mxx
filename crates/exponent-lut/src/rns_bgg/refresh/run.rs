//! Running a refresh to the top level (see [`crate::refresh`]) on RNS-BGG stages.
//!
//! [`RnsBggRefresh`] holds a declared refresh with its source Program, its keys and its chains:
//! the reference chain ([`RnsBggParameters::refresh_chain`] with `None`) and, per refreshed slot,
//! the source chain, its pool and the chains of its fresh-noise and smudge children
//! ([`RefreshChain`]). [`refresh_noise_accepted`] checks the simulated noise of every part before
//! any stage runs, given the error of the caller's encodings of `sigma_0 G`, and
//! [`RnsBggRefresh::security`] estimates the security of its encodings and of its output.
//!
//! [`RnsBggRefresh::run`] samples the payload and mask secrets on the reference chain and, on
//! every chain, first asks the caller for the chain's encoding of `sigma_0 G` under those secrets
//! ([`RefreshSigmaRequest`], [`RefreshSigma`]: the caller runs and records its own
//! preprocessing, encryption and online evaluation, see
//! [Caller inputs](crate::refresh#caller-inputs)), then runs the chain's two parts (one per
//! component, both reading that encoding), each through the three artifact stages:
//!
//! - preprocessing prepares the source Program on the chain (under the chain's hash namespace), the
//!   fresh-noise child, one smudge child per decoder, the decoded keys (with the caller's `K_g` and
//!   `A'`), one unmask trapdoor and the preimages;
//! - encryption encrypts the source input on the chain, the children's inputs on the shared hashed
//!   PRF label, the one encoding and the unmask hint;
//! - online evaluation combines `[g | one encoding]` (`g` the caller's row), the target component
//!   and the fresh noise into the decoded row of `q`, unmasks and rounds every column with its
//!   smudge, and outputs the residue `y_j mod q`.
//!
//! Every stage's validated graph is recorded in the returned [`RefreshRun`], whose
//! [`RefreshRun::error`] and [`RefreshRun::decode`] recombine the residues of every chain
//! modulo the product `P` of the refreshed primes and compare them with
//! `sigma A'_(j) - Delta_0 r_j x G^st_0` for the caller's `sigma_0`, and whose
//! [`RefreshRun::protocol`] states the claim of the recorded graphs.

use crate::{
    api::{self, ApiError, PlaintextInputs, StageArtifacts, StageExecution, StageOutputs},
    decoder::simulation::AverageCaseConfig,
    program::{ExponentLutProgram, ProgramData, RhsId, WireId},
    refresh::{FreshNoiseProgram, RefreshDeclaration, RefreshKeyLayouts},
    rns_bgg::{
        RefreshSigmaNoise, RnsBggMaskSecret, RnsBggNoiseBindings, RnsBggNoiseValue,
        RnsBggParameters, RnsBggPayloadSecret, RnsBggPreprocessEncryptDescriptor,
        RnsBggPreprocessOnlineEvalDescriptor, RnsEncryptBuild, RnsOnlineEvalBuild,
        RnsPreprocessBuild, decode_refresh, refresh_error, simulate_refresh_noise,
    },
    utils::refresh_output_security_bits,
};
use bigdecimal::BigDecimal;
use mxx_backends::{
    RuntimeValue, SessionStore,
    poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_ir_core::{Graph, ParamEnv, ValidatedGraph, artifact::ProductionId};
use num_bigint::{BigInt, BigUint};
use std::{cell::RefCell, collections::BTreeMap};

/// The online evaluation output port of the row `g` of the caller's encoding of `sigma_0 G` when
/// it is the first output of a Program: output `output-0` flattens its public key (ports 0 to 3)
/// and its two component rows (ports 4 and 5), and `g` is component 1.
pub const SIGMA_ROW_OUTPUT: &str = "output-0.5";

/// The chains of one refreshed slot: its source chain, the pool hosting it and its children, and
/// the chains of its fresh-noise and smudge children.
pub struct RefreshChain {
    pub pool: DCRTPolyParams,
    pub source: RnsBggParameters,
    pub fresh: RnsBggParameters,
    pub smudge: RnsBggParameters,
}

impl RefreshChain {
    /// Chain `chain` of `top` for the source Program `source`, with encoding error width
    /// `encoding_error_sigma` ([`RnsBggParameters::refresh_chain`] and the children's
    /// parameters).
    pub fn new(
        top: &RefreshDeclaration,
        source: &ExponentLutProgram,
        chain: usize,
        encoding_error_sigma: &BigDecimal,
    ) -> Result<Self, ApiError> {
        let (pool, parameters) =
            RnsBggParameters::refresh_chain(top, Some(chain), encoding_error_sigma.clone(), 1)
                .map_err(ApiError::scheme)?;
        let decoder =
            top.decoder((chain, 0)).ok_or_else(|| ApiError::input("the refresh has no chain"))?;
        Ok(Self {
            fresh: parameters
                .refresh_fresh_parameters(top, chain, source, &pool, 1)
                .map_err(ApiError::scheme)?,
            smudge: parameters
                .decoder_smudge_parameters(decoder, source, &pool, 1)
                .map_err(ApiError::scheme)?,
            source: parameters,
            pool,
        })
    }

    /// The source chain and the children's chains.
    pub fn parameters(&self) -> [&RnsBggParameters; 3] {
        [&self.source, &self.fresh, &self.smudge]
    }
}

/// Whether the simulated noise of every part of `top` (on the source Program `source`, encoding
/// error width `encoding_error_sigma`, unmask trapdoor width `trapdoor_sigma`) fits its decoding
/// margin and hides by `statistical_bits` under the average-case noise model with failure
/// probability `2^-failure_exponent` per accepted bound ([`simulate_refresh_noise`]).
/// `sigma(chain, chains, average)` is the error of the caller's encodings of `sigma_0 G` on chain
/// `chain` (both components' parts read it) under the average-case model `average`, e.g.
/// [`RnsBggParameters::refresh_carrier_noise`] of its Program.
pub fn refresh_noise_accepted(
    top: &RefreshDeclaration,
    source: &ExponentLutProgram,
    encoding_error_sigma: &BigDecimal,
    trapdoor_sigma: f64,
    statistical_bits: usize,
    failure_exponent: u32,
    sigma: impl Fn(usize, &RefreshChain, &AverageCaseConfig) -> Result<RefreshSigmaNoise, ApiError>,
) -> Result<bool, ApiError> {
    let average = AverageCaseConfig {
        failure_exponent,
        allow_average_acceptance: true,
        ..AverageCaseConfig::default()
    };
    for chain in 0..top.chains() {
        let chains = RefreshChain::new(top, source, chain, encoding_error_sigma)?;
        let one = [BigInt::from(1u8)];
        let bindings = RnsBggNoiseBindings::from_inputs(
            source,
            &[RnsBggNoiseValue::sampled_encoding(&chains.source, b"top-target", &one)],
            &[],
        )
        .map_err(ApiError::scheme)?;
        let sigma = sigma(chain, &chains, &average)?;
        for component in 0..2 {
            let noise = simulate_refresh_noise(
                &chains.source,
                &sigma,
                &chains.fresh,
                &chains.smudge,
                source,
                &bindings,
                top,
                (chain, component),
                0,
                trapdoor_sigma,
                Some(&average),
            )?;
            if !(noise.fits() && noise.hides(statistical_bits)) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// The estimated security of a refresh ([`RnsBggRefresh::security`]).
#[derive(Clone, Copy, Debug)]
pub struct RefreshSecurity {
    /// The bits of the largest level-0 modulus of every chain and child.
    pub largest_modulus_bits: u64,
    /// The primes of that modulus.
    pub largest_modulus_primes: usize,
    /// The estimated security of the encodings on it, in bits.
    pub encoding: u64,
    /// The estimated security of the refreshed output, in bits.
    pub output: u64,
}

/// A declared refresh ready to run on RNS-BGG stages: the declaration, its source Program, its
/// generated keys (`None` when they are only declared, see
/// [`crate::refresh::PrfSettings::declare_refresh`]), the reference chain, every refreshed slot's
/// chains and the unmask trapdoor width.
pub struct RnsBggRefresh {
    pub top: RefreshDeclaration,
    pub source: ExponentLutProgram,
    pub keys: Option<RefreshKeyLayouts>,
    pub reference: RnsBggParameters,
    pub chains: Vec<RefreshChain>,
    pub trapdoor_sigma: BigDecimal,
}

impl RnsBggRefresh {
    /// The refresh `top` of `source` with keys `keys`, its reference chain and every slot's
    /// chains with encoding error width `encoding_error_sigma`, and unmask trapdoor width
    /// `trapdoor_sigma`.
    pub fn new(
        top: RefreshDeclaration,
        source: ExponentLutProgram,
        keys: Option<RefreshKeyLayouts>,
        encoding_error_sigma: &BigDecimal,
        trapdoor_sigma: BigDecimal,
    ) -> Result<Self, ApiError> {
        let (_, reference) =
            RnsBggParameters::refresh_chain(&top, None, encoding_error_sigma.clone(), 1)
                .map_err(ApiError::scheme)?;
        let chains = (0..top.chains())
            .map(|chain| RefreshChain::new(&top, &source, chain, encoding_error_sigma))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { top, source, keys, reference, chains, trapdoor_sigma })
    }

    /// The estimated security of the encodings on the largest level-0 modulus of every chain and
    /// child (ternary secret, Gaussian error of the reference's encoding width) and of the
    /// refreshed output (binary mask, CBD(1) error, `2 w_0 n` samples over `Q_0`), by the lattice
    /// estimator.
    pub fn security(&self) -> Result<RefreshSecurity, String> {
        let largest = self
            .chains
            .iter()
            .flat_map(RefreshChain::parameters)
            .map(|parameters| parameters.level(0).map_err(|error| error.to_string()).cloned())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max_by_key(|level| level.modulus().bits())
            .ok_or("the refresh has no chain")?;
        let encoding = mxx_backends::lattice_estimator::lattice_security_bits(
            largest.ring_dimension(),
            largest.modulus().as_ref(),
            self.reference.encoding_error_sigma(),
        )?;
        let reference = self.top.reference();
        let modulus = reference.top.iter().map(|prime| BigUint::from(*prime)).product::<BigUint>();
        let n = reference.ring_dimension;
        let samples = (2 * reference.columns() * n as usize) as u64;
        let output = refresh_output_security_bits(n, &modulus, 1, samples)?;
        Ok(RefreshSecurity {
            largest_modulus_bits: largest.modulus().bits(),
            largest_modulus_primes: largest.to_crt().0.len(),
            encoding,
            output,
        })
    }

    /// Run the refresh on `execution`: sample the payload and mask secrets on the reference
    /// chain, then run every part's preprocessing, encryption and online evaluation on its chain,
    /// in the session context `context` (byte 30 the part's chain, byte 31 its component), under
    /// the stage hash key `hash_key` and the invocation's label tag `label_tag`
    /// ([`RefreshDeclaration::label`]). `lhs` gives the source Program's inputs on a chain's
    /// level 0. `a_prime` is the caller's `A'` (see
    /// [Caller inputs](crate::refresh#caller-inputs)), and `sigma` gives each chain's encoding of
    /// `sigma_0 G` (see [`RefreshSigmaRequest`]) before the chain's parts run; both parts read
    /// it. With
    /// `record_only`, the children bind their plaintexts and tables by shape only, so the stages
    /// can be recorded but not executed (as for an export; `a_prime` and the encodings' rows may
    /// then be absent); otherwise they bind the generated keys, which `self.keys` must hold.
    #[allow(clippy::too_many_arguments)]
    pub fn run<E, S>(
        &self,
        (context, hash_key, label_tag): ([u8; 32], [u8; 32], &[u8]),
        record_only: bool,
        a_prime: Option<RuntimeValue>,
        lhs: impl Fn(&DCRTPolyParams) -> PlaintextInputs<WireId, DCRTPoly>,
        mut sigma: impl FnMut(RefreshSigmaRequest<'_>, &mut E, &mut S) -> Result<RefreshSigma, ApiError>,
        execution: &mut E,
        store: &mut S,
    ) -> Result<RefreshRun, ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
    {
        let keys = match (record_only, &self.keys, &a_prime) {
            (true, _, _) => None,
            (false, Some(keys), Some(_)) => Some(keys),
            (false, _, _) => {
                return Err(ApiError::input("an executed refresh needs its keys and its A'"));
            }
        };
        let graphs = StageGraphs::default();
        let payload = RnsEncryptBuild::new(&self.reference, context).sample_payload_secret(
            execution,
            store,
            record(&graphs, "payload".to_owned()),
        )?;
        let mask = RnsEncryptBuild::new(&self.reference, context).sample_mask_secret(
            execution,
            store,
            record(&graphs, "mask".to_owned()),
        )?;
        let invocation = Invocation { context, hash_key, label_tag, a_prime: a_prime.as_ref() };
        let mut outputs = Vec::with_capacity(self.chains.len());
        let mut productions = BTreeMap::new();
        let mut sigmas = Vec::with_capacity(self.chains.len());
        for (chain, chains) in self.chains.iter().enumerate() {
            let lhs = lhs(&chains.source.levels()[0]);
            // The chain's encoding of `sigma_0 G`, in its own session (byte 31 two).
            let mut sigma_context = context;
            sigma_context[30] = chain as u8;
            sigma_context[31] = 2;
            let request = RefreshSigmaRequest {
                chain,
                chains,
                context: sigma_context,
                payload: &payload,
                mask: &mask,
                record_only,
                graphs: &graphs,
            };
            let provided = sigma(request, execution, store)?;
            if provided.preprocess.descriptor().binding().program() != Some(provided.program.id()) {
                return Err(ApiError::input("the sigma artifacts are not of its Program"));
            }
            let row = provided.online_eval.outputs.get(SIGMA_ROW_OUTPUT).cloned();
            if row.is_none() && !record_only {
                return Err(ApiError::input("an executed refresh needs every sigma row"));
            }
            productions.insert(
                provided.preprocess.production().clone(),
                format!("sigma_preprocess_{chain}"),
            );
            productions.insert(provided.encryption.clone(), format!("sigma_encrypt_{chain}"));
            let mut parts = Vec::with_capacity(2);
            for component in 0..2 {
                let part = (chain, component);
                let name = |stage: &str| format!("{stage}_{chain}_{component}");
                let (outputs, [preprocess, encrypt]) = self.run_part(
                    keys,
                    chains,
                    &lhs,
                    (&payload, &mask),
                    part,
                    (&provided, row.clone()),
                    &invocation,
                    execution,
                    store,
                    &graphs,
                )?;
                productions.insert(preprocess, name("preprocess"));
                productions.insert(encrypt, name("encrypt"));
                parts.push(outputs);
            }
            sigmas.push((provided.program, provided.namespace));
            tracing::info!(chain, "top refresh chain evaluated");
            let [first, second] = <[StageOutputs; 2]>::try_from(parts)
                .map_err(|_| ApiError::input("a chain has two parts"))?;
            outputs.push([first, second]);
        }
        Ok(RefreshRun {
            payload,
            mask,
            outputs: (!record_only).then_some(outputs),
            graphs: graphs.into_inner(),
            productions,
            sigmas,
            a_prime,
            hash_key,
            label_tag: label_tag.to_vec(),
        })
    }

    /// One part's preprocessing, encryption and online evaluation on its chain `chains`, with
    /// the chain's encoding of `sigma_0 G` `sigma` (its artifacts and its row), its outputs and
    /// the productions of its preprocessing and encryption. Without `keys`, the children bind
    /// their plaintexts and tables by shape only.
    #[allow(clippy::too_many_arguments)]
    fn run_part<E, S>(
        &self,
        keys: Option<&RefreshKeyLayouts>,
        chains: &RefreshChain,
        lhs: &PlaintextInputs<WireId, DCRTPoly>,
        (payload, mask): (&RnsBggPayloadSecret, &RnsBggMaskSecret),
        (chain, component): (usize, usize),
        (sigma, sigma_row): (&RefreshSigma, Option<RuntimeValue>),
        invocation: &Invocation<'_>,
        execution: &mut E,
        store: &mut S,
        graphs: &StageGraphs,
    ) -> Result<(StageOutputs, [ProductionId; 2]), ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
    {
        let Self { top, source, trapdoor_sigma, .. } = self;
        let Invocation { hash_key, label_tag, .. } = *invocation;
        let part = (chain, component);
        let name = |stage: &str| format!("{stage}_{chain}_{component}");
        let missing = |what: &'static str| move || ApiError::input(what);
        let fresh_program = top.fresh(part).ok_or_else(missing("the refresh has no such part"))?;
        let decoder = top.decoder(part).ok_or_else(missing("the refresh has no such part"))?;
        // One preprocessing session per part.
        let mut context = invocation.context;
        context[30] = chain as u8;
        context[31] = component as u8;
        let env = ParamEnv::default();
        let fresh_level = chains.fresh.levels()[0].clone();
        let smudge_level = chains.smudge.levels()[0].clone();
        let chain_namespace = top.chain_namespace(chain);
        let fresh_namespace =
            top.fresh_namespace(part).ok_or_else(missing("the refresh has no such part"))?;
        let sampler_error = |error| ApiError::scheme(crate::refresh::RefreshError::Sampler(error));
        let prf_error = |error| ApiError::scheme(crate::refresh::RefreshError::Prf(error));
        let child = ChildSession { context, hash_key, payload };

        let fresh_preprocess = child.preprocess(
            &chains.fresh,
            fresh_namespace,
            fresh_program.program(),
            keys.map(|keys| {
                fresh_program.sampler().key_inputs(&fresh_level, &keys.fresh[component])
            })
            .transpose()
            .map_err(sampler_error)?,
        )?;
        let smudge_keys = keys
            .map(|keys| {
                keys.smudge.get(&part).ok_or_else(missing("the refresh keys miss a part's smudge"))
            })
            .transpose()?;
        let mut smudge_preprocess = child.preprocess(
            &chains.smudge,
            *decoder.commitment(),
            decoder.smudge().program(),
            smudge_keys
                .map(|smudge| decoder.smudge().prf().key_inputs(&smudge_level, smudge))
                .transpose()
                .map_err(prf_error)?,
        )?;
        smudge_preprocess.bind_decoder_smudge(decoder)?;
        // The caller's `A'`, read by preprocessing and online evaluation.
        let preprocess_build =
            RnsPreprocessBuild::new(&chains.source, context).with_hash_namespace(chain_namespace);
        let preprocess_build = match invocation.a_prime {
            Some(a_prime) => preprocess_build.with_refresh_matrix(a_prime.clone()),
            None => preprocess_build,
        };
        let no_rhs = PlaintextInputs::<RhsId, DCRTPoly>::new();
        let chain_payload = payload.at_level(&chains.source, 0).map_err(ApiError::scheme)?;
        let (preprocess, online_eval) = api::preprocess(
            source,
            &env,
            &no_rhs,
            Some(&chain_payload),
            &hash_key,
            preprocess_build,
            execution,
            store,
            |build, outputs| {
                build.apply_refresh(
                    0,
                    top,
                    part,
                    outputs[0].clone(),
                    (&sigma.preprocess, sigma.program.program()),
                    fresh_preprocess,
                    smudge_preprocess,
                    trapdoor_sigma,
                )?;
                Ok(outputs)
            },
            record(graphs, name("preprocess")),
        )?;

        let a = top.label(hash_key, label_tag).map_err(ApiError::scheme)?;
        let fresh_encrypt = child.encrypt(
            &preprocess,
            mask,
            &chains.fresh,
            fresh_namespace,
            fresh_program.program(),
            keys.map(|_| fresh_program.sampler().plaintext_inputs(&fresh_level, &a))
                .transpose()
                .map_err(sampler_error)?,
        )?;
        let mut smudge_encrypt = child.encrypt(
            &preprocess,
            mask,
            &chains.smudge,
            *decoder.commitment(),
            decoder.smudge().program(),
            keys.map(|_| decoder.smudge().prf().plaintext_inputs(&smudge_level, &a))
                .transpose()
                .map_err(prf_error)?,
        )?;
        smudge_encrypt.bind_decoder_smudge(decoder).map_err(ApiError::scheme)?;
        let encrypted = api::encrypt(
            &source.encryption_inputs(),
            &env,
            lhs,
            Some(&preprocess),
            &chain_payload,
            Some(&mask.at_level(&chains.source, 0).map_err(ApiError::scheme)?),
            &hash_key,
            RnsEncryptBuild::new(&chains.source, context).with_hash_namespace(chain_namespace),
            execution,
            store,
            |build| build.bind_refresh(top, part, fresh_encrypt, smudge_encrypt),
            record(graphs, name("encrypt")),
        )?;

        let child_online = |parameters, namespace| {
            RnsOnlineEvalBuild::new(parameters, context).with_hash_namespace(namespace)
        };
        let mut fresh_online = child_online(&chains.fresh, fresh_namespace);
        fresh_online.bind_refresh_fresh_child(
            top,
            part,
            label_tag,
            &online_eval,
            &encrypted,
            &hash_key,
        )?;
        let mut smudge_online = child_online(&chains.smudge, *decoder.commitment());
        smudge_online.bind_refresh_smudge_child(
            top,
            part,
            label_tag,
            &online_eval,
            &encrypted,
            &hash_key,
        )?;
        let online =
            RnsOnlineEvalBuild::new(&chains.source, context).with_hash_namespace(chain_namespace);
        let online = match invocation.a_prime {
            Some(a_prime) => online.with_refresh_matrix(a_prime.clone()),
            None => online,
        };
        let outputs = api::online_eval(
            &source.without_lut_values(),
            &env,
            &online_eval,
            &encrypted,
            &hash_key,
            online,
            execution,
            store,
            |build, outputs| {
                build.apply_refresh(0, top, part, sigma_row, fresh_online, smudge_online)?;
                Ok(outputs)
            },
            record(graphs, name("online_eval")),
        )?;
        Ok((outputs, [preprocess.production().clone(), encrypted.production().clone()]))
    }
}

/// What the caller's `sigma` of [`RnsBggRefresh::run`] receives for one chain: the chain, its
/// parameters, the session context of its encoding of `sigma_0 G` and the run's secrets. The
/// caller returns the chain's encoding of `sigma_0 G` under these secrets, which both parts of
/// the chain read (see
/// [Caller inputs](crate::refresh#caller-inputs)) as a [`RefreshSigma`]: typically it
/// preprocesses, encrypts and evaluates a binary [`FreshNoiseProgram`] of `n` samples modulo the
/// chain's output prime on [`RnsBggParameters::refresh_child_parameters`] of `chains.source`,
/// with the output use [`crate::rns_bgg::RnsOutputUse::Gadget`] and the payload and mask secrets
/// at that chain's level 0, observing its stages with [`Self::record`] so that the claim
/// ([`RefreshRun::protocol`]) includes them. With `record_only`, the stages are recorded but not
/// executed (bind the plaintexts and data by shape, see `with_inputs_by_shape`).
pub struct RefreshSigmaRequest<'a> {
    pub chain: usize,
    pub chains: &'a RefreshChain,
    pub context: [u8; 32],
    pub payload: &'a RnsBggPayloadSecret,
    pub mask: &'a RnsBggMaskSecret,
    pub record_only: bool,
    graphs: &'a StageGraphs,
}

impl<'a> RefreshSigmaRequest<'a> {
    /// The stage observer recording the graph of the chain's sigma stage `phase`
    /// (`preprocess`, `encrypt` or `online_eval`) as stage `sigma_<phase>_<chain>` of the run.
    pub fn record(
        &self,
        phase: &str,
    ) -> impl FnOnce(&ValidatedGraph) -> Result<(), ApiError> + 'a + use<'a> {
        record(self.graphs, format!("sigma_{phase}_{}", self.chain))
    }
}

/// The caller's encoding of `sigma_0 G` for one chain (see [`RefreshSigmaRequest`]): the Program
/// that computes it (its first output, a binary [`FreshNoiseProgram`] for the claim's tie of
/// `sigma_0` to its PRF bits) and the hash namespace its stages ran under, its preprocessing
/// artifacts (the chain's parts' preprocessing reads its key), its encryption production, and its
/// online evaluation's outputs (the parts' online evaluation reads [`SIGMA_ROW_OUTPUT`]; empty
/// for a record-only run).
pub struct RefreshSigma {
    pub program: FreshNoiseProgram,
    pub namespace: [u8; 32],
    pub preprocess: StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
    pub encryption: ProductionId,
    pub online_eval: StageOutputs,
}

/// The session identities of one run: the session context, the stage hash key, the label tag and
/// the caller's `A'`.
#[derive(Clone, Copy)]
struct Invocation<'a> {
    context: [u8; 32],
    hash_key: [u8; 32],
    label_tag: &'a [u8],
    a_prime: Option<&'a RuntimeValue>,
}

/// What every child stage of one part shares: the part's session context, the stage hash key and
/// the payload secret.
struct ChildSession<'a> {
    context: [u8; 32],
    hash_key: [u8; 32],
    payload: &'a RnsBggPayloadSecret,
}

impl ChildSession<'_> {
    /// A child's preprocessing on `parameters` under the hash namespace `namespace`, bound to its
    /// RHS plaintexts and coordinate tables `values`, or, without them, to their shapes only.
    fn preprocess<'p>(
        &self,
        parameters: &'p RnsBggParameters,
        namespace: [u8; 32],
        program: &ExponentLutProgram,
        values: Option<(PlaintextInputs<RhsId, DCRTPoly>, ProgramData)>,
    ) -> Result<RnsPreprocessBuild<'p>, ApiError> {
        let (rhs, data) = values.unzip();
        let mut child = RnsPreprocessBuild::new(parameters, self.context)
            .with_hash_namespace(namespace)
            .with_data(data.unwrap_or_default());
        child.bind_rhs_inputs(
            program,
            &ParamEnv::default(),
            rhs.as_ref(),
            Some(&self.payload.at_level(parameters, 0).map_err(ApiError::scheme)?),
            &self.hash_key,
        )?;
        Ok(child)
    }

    /// A child's encryption on `parameters` under the hash namespace `namespace`, after the
    /// part's preprocessing `preprocess`, bound to its input plaintexts `lhs`, or, without them,
    /// to their shapes only.
    fn encrypt<'p>(
        &self,
        preprocess: &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
        mask: &RnsBggMaskSecret,
        parameters: &'p RnsBggParameters,
        namespace: [u8; 32],
        program: &ExponentLutProgram,
        lhs: Option<PlaintextInputs<WireId, DCRTPoly>>,
    ) -> Result<RnsEncryptBuild<'p>, ApiError> {
        let mut child =
            RnsEncryptBuild::new(parameters, self.context).with_hash_namespace(namespace);
        child.bind_lhs_inputs(
            &program.encryption_inputs(),
            &ParamEnv::default(),
            lhs.as_ref(),
            &self.payload.at_level(parameters, 0).map_err(ApiError::scheme)?,
            Some(&mask.at_level(parameters, 0).map_err(ApiError::scheme)?),
            &self.hash_key,
        )?;
        child.bind_preprocess_artifacts(preprocess)?;
        Ok(child)
    }
}

/// Graphs of the executed stages by stage name, recorded for the claim.
type StageGraphs = RefCell<BTreeMap<String, Graph>>;

/// A stage observer that records the validated graph of `stage`.
fn record<'a>(
    graphs: &'a StageGraphs,
    stage: String,
) -> impl FnOnce(&ValidatedGraph) -> Result<(), ApiError> + 'a {
    move |validated| {
        graphs.borrow_mut().insert(stage, validated.source.clone());
        Ok(())
    }
}

/// One run of a refresh ([`RnsBggRefresh::run`]): its secrets, every part's outputs and the
/// recorded stage graphs. Stage `payload` and `mask` sample the secrets, stage
/// `<phase>_<chain>_<component>` is a part's phase (`preprocess`, `encrypt`, `online_eval`),
/// stage `sigma_<phase>_<chain>` the phase of the chain's encoding of `sigma_0 G`
/// it recorded, and [`Self::decode`] records stage `decode`.
pub struct RefreshRun {
    pub payload: RnsBggPayloadSecret,
    pub mask: RnsBggMaskSecret,
    /// Every chain's two parts' outputs (`outputs[chain][component]`); `None` for a record-only
    /// run.
    pub outputs: Option<Vec<[StageOutputs; 2]>>,
    pub(super) graphs: BTreeMap<String, Graph>,
    /// The stage name of every preprocessing and encryption production.
    pub(super) productions: BTreeMap<ProductionId, String>,
    /// Every chain's sigma Program and hash namespace.
    pub(super) sigmas: Vec<(FreshNoiseProgram, [u8; 32])>,
    /// The caller's `A'`.
    pub(super) a_prime: Option<RuntimeValue>,
    pub(super) hash_key: [u8; 32],
    pub(super) label_tag: Vec<u8>,
}

impl RefreshRun {
    /// The parts' outputs as the decode graphs read them.
    fn results(&self) -> Option<Vec<[&StageOutputs; 2]>> {
        self.outputs
            .as_ref()
            .map(|outputs| outputs.iter().map(|[first, second]| [first, second]).collect())
    }

    /// The error rows `e'_j` of the executed `refresh`, per column the `n` coefficients centered
    /// modulo `P`, for the caller's mask polynomial `sigma` (`sigma_0`, binary) and the expected
    /// plaintext `expected` on the reference chain's level 0 (see [`refresh_error`]). A
    /// record-only run has no error rows.
    pub fn error<E, S>(
        &self,
        refresh: &RnsBggRefresh,
        sigma: &DCRTPoly,
        expected: &DCRTPoly,
        execution: &mut E,
        store: &mut S,
    ) -> Result<[Vec<Vec<BigInt>>; 2], ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
    {
        let results =
            self.results().ok_or_else(|| ApiError::input("a record-only run has no errors"))?;
        let a_prime =
            self.a_prime.as_ref().ok_or_else(|| ApiError::input("an executed run has its A'"))?;
        refresh_error(
            &refresh.top,
            &results,
            &refresh.reference,
            &self.payload,
            sigma,
            expected,
            a_prime,
            execution,
            store,
        )
    }

    /// The decode stage of `refresh` for the caller's mask polynomial `sigma` and the expected
    /// plaintext `expected` (see [`decode_refresh`]): the number of refreshed coefficients that
    /// fail to decode, zero for a correct refresh. Its graph is recorded as stage `decode`; for a
    /// record-only run it is built and recorded but not executed (any `sigma` serves), so the
    /// call then returns the execution's error.
    pub fn decode<E, S>(
        &mut self,
        refresh: &RnsBggRefresh,
        sigma: &DCRTPoly,
        expected: &DCRTPoly,
        execution: &mut E,
        store: &mut S,
    ) -> Result<BigInt, ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
    {
        let graphs = StageGraphs::default();
        let decoded = decode_refresh(
            &refresh.top,
            self.results().as_deref(),
            &refresh.reference,
            &self.payload,
            sigma,
            expected,
            self.a_prime.as_ref(),
            execution,
            store,
            record(&graphs, "decode".to_owned()),
        );
        self.graphs.extend(graphs.into_inner());
        decoded
    }
}
