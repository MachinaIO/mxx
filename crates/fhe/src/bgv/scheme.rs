//! The binding of BGV to the public-key homomorphic encryption syntax ([`crate::spec`]).
//!
//! The levels count from the top level `0`; level `i + 1` is `modswitch_steps` CRT levels below
//! level `i`, and the depth is the number of levels but one. A ciphertext carries its level in its
//! ring.
//!
//! | Syntax | BGV |
//! | --- | --- |
//! | depth | the number of levels but one |
//! | gate | slot-wise multiplication modulo the plaintext modulus |
//! | `KeyGen` | [`BgvParams::keygen`] and the relinearization key at the top level |
//! | `Enc` | [`BgvParams::encrypt`] at the top level |
//! | `Dec` | [`BgvParams::decrypt`] at the ciphertext's level |
//! | gate on ciphertexts | a gate on levels `i` and `j` switches the argument of the higher level down to level `max(i, j)` and aligns its correction factor, then multiplies, relinearizes, and switches down to the next level |
//!
//! Every gate relinearizes with the one key of the top level ([`BgvParams::relinearize`] reduces
//! it to the gate's level), so no key beyond that of the security statement is needed.
//!
//! The adversary of the security statement sees the public key and the relinearization key. The
//! statement uses three allowed assumptions, in the order of the game hops of the proof:
//!
//! 1. `RlweKeySwitchCircular`: the relinearization key, which encrypts `s²` under `s` over the
//!    extended modulus `PQ`, is indistinguishable from uniform next to the public key.
//! 2. `DecisionRlwe` with one sample: the public key `(a, s a + t e)` is indistinguishable from
//!    uniform. The simulator scales a standard sample `(a, s a + e)` by the plaintext modulus `t`.
//! 3. `DecisionRlwe` with two samples: under a uniform public key, the ciphertext `(pk₀ u + t e₀,
//!    pk₁ u + t e₁ + m)` is indistinguishable from uniform plus the message.
use crate::{
    BgvCiphertext, BgvParams,
    spec::{Correctness, Depth, Fhe, Reduction, SecurityStatement, Statement, roles},
    utils,
};
use mxx_backends::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use mxx_dsl::{
    BuiltGraph, DslContext, Family, FamilyType, GraphValue, Int, IntType, Mat, Ring, concat_rows,
    parallel,
};
use mxx_ir_core::{
    Graph, ParamEnv, Rational,
    node::IndexRange,
    protocol::InputValueContract,
    security::{AssumptionParams, SamplerLaw},
};
use num_bigint::{BigInt, BigUint};

/// BGV at `params`, whose gates switch down `modswitch_steps` levels.
pub struct BgvScheme<'a> {
    pub params: &'a BgvParams,
    pub modswitch_steps: usize,
}

fn built(context: DslContext) -> Graph {
    let built: BuiltGraph = context.build().unwrap();
    built.graph
}

fn dsl_ring(parameters: &DCRTPolyParams) -> Ring {
    Ring::from_crt_moduli(
        parameters.to_crt().0.into_iter().map(Into::into).collect(),
        parameters.ring_dimension(),
    )
}

/// Column `index` of a row.
fn column(row: &Mat, index: usize) -> Mat {
    row.clone().slice(None, Some(IndexRange { start: index.into(), end: (index + 1).into() }))
}

impl BgvScheme<'_> {
    /// The top-level ring, the ring of the relinearization key, and the key's width.
    fn rings(&self) -> (Ring, Ring, usize) {
        let common = &self.params.common;
        let top = common.ring.crt_depth() - 1;
        let (key_params, key_width) = self.params.key_switch_parameters(top).unwrap();
        (dsl_ring(&common.ring), dsl_ring(&key_params), key_width)
    }

    /// The CRT level of each ciphertext level, from the top.
    fn crt_levels(&self) -> Vec<usize> {
        assert!(self.modswitch_steps > 0, "every gate switches down");
        let top = self.params.common.ring.crt_depth() - 1;
        (0..=top / self.modswitch_steps).map(|level| top - level * self.modswitch_steps).collect()
    }

    fn level_ring(&self, crt_level: usize) -> Ring {
        dsl_ring(&self.params.common.parameters_at(crt_level).unwrap())
    }

    fn slots(&self) -> usize {
        self.params.common.ring.ring_dimension() as usize
    }

    /// `Enc`, also the challenge graph of the first two reductions.
    fn encryption_graph(&self, name: &str) -> Graph {
        let (ring, ..) = self.rings();
        let context = DslContext::new(name);
        let message = context.int_family_input(roles::MESSAGE, self.slots());
        let ct = self.params.encrypt(&ring.input(roles::PUBLIC_KEY, (2, 1)), &message).unwrap();
        built(context.output(roles::CIPHERTEXT, ct.components).unwrap())
    }

    /// A fresh ciphertext of the top level.
    fn fresh_template(&self) -> BgvCiphertext {
        let (ring, ..) = self.rings();
        self.params
            .encrypt(
                &ring.input(roles::PUBLIC_KEY, (2, 1)),
                &DslContext::new("template").int_family_input(roles::MESSAGE, self.slots()),
            )
            .unwrap()
    }

    /// The level of `ct`.
    fn level(&self, ct: &BgvCiphertext) -> usize {
        let crt_level = self.params.level_of(&ct.components).unwrap();
        self.crt_levels().iter().position(|&level| level == crt_level).expect("a ciphertext level")
    }

    /// Multiplies two ciphertexts of level `level`, relinearizes with `key`, and switches down to
    /// the next level.
    fn multiply(
        &self,
        level: usize,
        key: &Mat,
        x: &BgvCiphertext,
        y: &BgvCiphertext,
    ) -> BgvCiphertext {
        let quadratic = self.params.mul_unrelinearized(x, y).unwrap();
        let relinearized = self.params.relinearize(&quadratic, key).unwrap();
        self.params.mod_switch_to(&relinearized, self.crt_levels()[level + 1]).unwrap()
    }

    /// The ciphertexts of every level, as fresh encryption and multiplication produce them.
    fn templates(&self) -> Vec<BgvCiphertext> {
        let (_, key_ring, key_width) = self.rings();
        let key = key_ring.input(roles::EVALUATION_KEY, (2, key_width));
        let mut templates = vec![self.fresh_template()];
        for level in 0..self.max_level() {
            let input = BgvCiphertext {
                components: self
                    .level_ring(self.crt_levels()[level])
                    .input(roles::CIPHERTEXT, (2, 1)),
                ..templates[level].clone()
            };
            let output = self.multiply(level, &key, &input, &input);
            templates.push(output);
        }
        templates
    }

    /// `ct`, of level at most `level`, switched down to level `level` with the correction factor
    /// of `templates[level]`.
    fn switch_down(
        &self,
        ct: &BgvCiphertext,
        level: usize,
        templates: &[BgvCiphertext],
    ) -> BgvCiphertext {
        if self.level(ct) == level {
            return ct.clone();
        }
        let output = self.params.mod_switch_to(ct, self.crt_levels()[level]).unwrap();
        if output.correction_factor == templates[level].correction_factor {
            return output;
        }
        self.params.match_correction_factor(&output, templates[level].correction_factor).unwrap()
    }

    /// The deepest level, and the depth of the circuits the scheme evaluates: the number of levels
    /// but one.
    fn max_level(&self) -> usize {
        self.crt_levels().len() - 1
    }

    fn t_scalar(&self) -> Mat {
        utils::scalar(&self.params.common.ring, self.params.plaintext_modulus)
    }
}

impl Fhe for BgvScheme<'_> {
    type Message = Family<Int>;
    type PublicKey = Mat;
    type EvaluationKey = Mat;
    type SecretKey = Mat;
    type Ciphertext = BgvCiphertext;

    fn depth(&self) -> Depth {
        Depth::Bounded(self.max_level())
    }

    fn message_space(&self) -> (FamilyType<IntType>, InputValueContract) {
        let schema =
            DslContext::new("message").int_family_input(roles::MESSAGE, self.slots()).schema();
        let slot = InputValueContract::IntegerRange {
            lower: 0.into(),
            upper: (self.params.plaintext_modulus - 1).into(),
        };
        (schema, InputValueContract::Family { count: self.slots().into(), element: Box::new(slot) })
    }

    fn key_generation(&self) -> (Mat, Mat, Mat) {
        let bgv = self.params;
        let (sk, pk) = bgv.keygen().unwrap();
        let rk = bgv.relinearization_key(&sk, bgv.common.ring.crt_depth() - 1).unwrap();
        (pk, rk, sk)
    }

    fn encrypt(&self, public_key: &Mat, message: &Family<Int>) -> BgvCiphertext {
        self.params.encrypt(public_key, message).unwrap()
    }

    fn plain_gate(&self, x: &Family<Int>, y: &Family<Int>) -> Family<Int> {
        let modulus = Int::constant(self.params.plaintext_modulus);
        parallel(self.slots(), |slot| Ok(x.at(slot.clone()).mul(y.at(slot)).rem(modulus.clone())))
            .unwrap()
    }

    fn eval_gate(
        &self,
        evaluation_key: &Mat,
        x: &BgvCiphertext,
        y: &BgvCiphertext,
    ) -> BgvCiphertext {
        let templates = self.templates();
        let level = self.level(x).max(self.level(y));
        assert!(level < self.max_level(), "no level below {level}");
        let (x, y) =
            (self.switch_down(x, level, &templates), self.switch_down(y, level, &templates));
        self.multiply(level, evaluation_key, &x, &y)
    }

    fn decrypt(&self, secret_key: &Mat, ciphertext: &BgvCiphertext) -> Family<Int> {
        self.params.decrypt(secret_key, ciphertext).unwrap()
    }
}

impl BgvScheme<'_> {
    /// The statement this crate proves for the scheme, and the simulators of its reductions,
    /// one per assumption in the order of the game hops.
    pub fn security(&self) -> (Statement, Vec<Reduction>) {
        let bgv = self.params;
        let common = &bgv.common;
        let n = self.slots();
        let (ring, key_ring, key_width) = self.rings();
        let top = common.ring.crt_depth() - 1;
        let (key_params, _) = bgv.key_switch_parameters(top).unwrap();
        let t = bgv.plaintext_modulus;
        let secret = SamplerLaw::Interval {
            minimum: common.secret_range.minimum.evaluate(&ParamEnv::default()).unwrap(),
            maximum: common.secret_range.maximum.evaluate(&ParamEnv::default()).unwrap(),
        };
        let error = SamplerLaw::Gaussian {
            sigma: Rational::from_f64_exact(common.error_sigma).unwrap(),
            cutoff: BigInt::from(common.error_cutoff.clone()),
        };
        let q = common.ring.modulus().as_ref().clone();
        let pq = key_params.modulus().as_ref().clone();
        let p = &pq / &q;
        let factors = common
            .ring
            .to_crt()
            .0
            .chunks(bgv.hybrid.digit_size)
            .map(|group| {
                let digit = group.iter().map(|&prime| BigUint::from(prime)).product::<BigUint>();
                &p * (&q / digit)
            })
            .collect::<Vec<_>>();
        assert_eq!(factors.len(), key_width, "one key column per digit");
        let keys = |context: DslContext, pk: Mat, ek: Mat| {
            context
                .output(roles::PUBLIC_KEY, pk)
                .unwrap()
                .output(roles::EVALUATION_KEY, ek)
                .unwrap()
        };
        // Hop 1: the key rows come from the challenge, next to its public key.
        let circular_setup = built(keys(
            DslContext::new("bgv-scheme-sim-circular-setup"),
            concat_rows![ring.input("pk_a", (1, 1)), ring.input("pk_b", (1, 1))],
            concat_rows![
                key_ring.input("key_a", (1, key_width)),
                key_ring.input("key_b", (1, key_width))
            ],
        ));
        // Hop 2: the public key is a standard sample scaled by `t`; the key is uniform.
        let public_key_setup = built(keys(
            DslContext::new("bgv-scheme-sim-public-key-setup"),
            concat_rows![
                self.t_scalar() * ring.input("a", (1, 1)),
                self.t_scalar() * ring.input("b", (1, 1))
            ],
            key_ring.uniform_residue((2, key_width)),
        ));
        // Hop 3: the public key is the challenge's public row scaled by `t`; the challenge
        // ciphertext is its other row scaled by `t`, plus the encoded message.
        let ciphertext_setup = {
            let a = ring.input("a", (1, 2));
            let pk = concat_rows![self.t_scalar() * column(&a, 0), self.t_scalar() * column(&a, 1)];
            built(
                keys(
                    DslContext::new("bgv-scheme-sim-ciphertext-setup"),
                    pk,
                    key_ring.uniform_residue((2, key_width)),
                )
                .output("b", ring.input("b", (1, 2)))
                .unwrap(),
            )
        };
        let ciphertext_challenge = {
            let context = DslContext::new("bgv-scheme-sim-ciphertext-challenge");
            let message = context.int_family_input(roles::MESSAGE, n);
            let encoded = bgv.encode_message(&message).unwrap();
            let b = ring.input("b", (1, 2));
            let ct = concat_rows![
                self.t_scalar() * column(&b, 0),
                self.t_scalar() * column(&b, 1) + encoded
            ];
            built(context.output(roles::CIPHERTEXT, ct).unwrap())
        };
        let rlwe = |samples| AssumptionParams::DecisionRlwe {
            modulus: q.clone(),
            ring_dimension: n as u32,
            secret: secret.clone(),
            error: error.clone(),
            samples,
        };
        let statement = Statement {
            correctness: Correctness::WorstCase,
            security: Some(SecurityStatement {
                assumptions: vec![
                    AssumptionParams::RlweKeySwitchCircular {
                        modulus: q.clone(),
                        extended_modulus: pq.clone(),
                        ring_dimension: n as u32,
                        secret: secret.clone(),
                        error: error.clone(),
                        plaintext_modulus: BigUint::from(t),
                        factors,
                    },
                    rlwe(1),
                    rlwe(2),
                ],
                // Each simulator costs about 2^29.2 units at the fixture parameters (dominated by
                // the encryption graph's slot encoding); the proof checks the exact cost.
                cost_bound: 1 << 30,
            }),
        };
        let reductions = vec![
            Reduction {
                setup: circular_setup,
                challenge: self.encryption_graph("bgv-scheme-sim-circular-challenge"),
            },
            Reduction {
                setup: public_key_setup,
                challenge: self.encryption_graph("bgv-scheme-sim-public-key-challenge"),
            },
            Reduction { setup: ciphertext_setup, challenge: ciphertext_challenge },
        ];
        (statement, reductions)
    }
}
