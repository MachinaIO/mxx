//! Parameters of a Diamond iO 2 obfuscation (see `crates/io/DIO2-SPEC.md`, §3 and §4).

use super::Dio2Error;
use bigdecimal::BigDecimal;
use mxx_backends::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use mxx_exponent_lut::{
    SparseLwrPrfProfile,
    refresh::{PrfSettings, TopReference},
    rns_bgg::{RnsBggParameters, RnsBggRoot},
};

/// The ring, the root modulus and its chain, the sparse-LWR PRF and the decoders of an
/// obfuscation.
///
/// - The root modulus `Q_0` is the product of `primes`, the modulus of every refreshed row; there
///   are no extension primes (spec §4.2).
/// - The chain has `levels` levels, every one of `Q_0`: the evaluation never drops a modulus, so
///   one Program per step serves every prime's decoder. The decoding level is `L = levels - 2`
///   (level `L + 1` holds the state gadget of `L`), and `Delta_l = 1` on every level but the last.
/// - The PRF of every sampler and smudge (`prf`) has its input modulus `Q` as the label modulus,
///   its LWR dimension `nu` as the label height `v`, and `m_v = v log2 Q`.
#[derive(Clone, Debug)]
pub struct Dio2Parameters {
    pub ring_dimension: u32,
    pub crt_bits: usize,
    pub base_bits: u32,
    pub primes: Vec<u64>,
    pub levels: usize,
    pub prf: PrfSettings,
    /// The smudge digits `d` of every decoder (the smudge range is `P^d`).
    pub smudge_digits: usize,
    /// The width of the anchor trapdoor's Gaussian.
    pub trapdoor_sigma: u32,
    /// The width of the Static preprocessing ciphertexts' errors (zero for noise-free runs).
    pub encoding_error_sigma: BigDecimal,
}

impl Dio2Parameters {
    /// Check the chain and the PRF against the ring.
    pub fn validate(&self) -> Result<(), Dio2Error> {
        let invalid = |reason| Err(Dio2Error::Parameters(reason));
        if self.levels < 3 {
            return invalid("the chain needs a decoding level and two levels after it");
        }
        if self.smudge_digits == 0 {
            return invalid("a decoder smudges with at least one digit");
        }
        self.profile()?;
        Ok(())
    }

    /// The parameters of the root modulus `Q_0`.
    pub fn pool(&self) -> Result<DCRTPolyParams, Dio2Error> {
        DCRTPolyParams::try_new(
            self.ring_dimension,
            self.primes.len(),
            self.crt_bits,
            self.base_bits,
            Some(self.primes.clone()),
            None,
        )
        .map_err(|_| Dio2Error::Parameters("the primes do not make a CRT basis"))
    }

    /// Every ring a backend running an obfuscation hosts: the root modulus.
    pub fn rings(&self) -> Result<Vec<DCRTPolyParams>, Dio2Error> {
        Ok(vec![self.pool()?])
    }

    /// The root of a step's chain under the step's root hash key `hash_key`.
    pub fn root(&self, hash_key: [u8; 32]) -> Result<RnsBggRoot, Dio2Error> {
        Ok(RnsBggRoot::new(self.pool()?, hash_key))
    }

    /// The columns `w_0` of a refreshed row: the digits of `Q_0`.
    pub fn columns(&self) -> Result<usize, Dio2Error> {
        Ok(self.pool()?.modulus_digits())
    }

    /// The decoding level `L`.
    pub fn decoding_level(&self) -> usize {
        self.levels - 2
    }

    /// The chain rooted at `root`, its Static preprocessing errors of width
    /// `encoding_error_sigma`.
    pub fn chain(&self, root: &RnsBggRoot) -> Result<RnsBggParameters, Dio2Error> {
        RnsBggParameters::new(vec![self.pool()?; self.levels], self.encoding_error_sigma.clone(), 1)
            .and_then(|parameters| parameters.with_root(root.clone()))
            .map_err(Dio2Error::from)
    }

    /// The index of `primes[prime]` in the basis of `Q_0`.
    pub fn prime_index(&self, prime: usize) -> Result<usize, Dio2Error> {
        self.pool()?
            .to_crt()
            .0
            .iter()
            .position(|tower| *tower == self.primes[prime])
            .ok_or(Dio2Error::Parameters("a prime is outside the basis"))
    }

    /// The refreshed format: level 0 and level 1 both of `Q_0`.
    pub fn reference(&self) -> TopReference {
        TopReference {
            top: self.primes.clone(),
            next: self.primes.clone(),
            ring_dimension: self.ring_dimension,
            base_bits: self.base_bits,
        }
    }

    /// The sparse-LWR profile of every PRF.
    pub fn profile(&self) -> Result<SparseLwrPrfProfile, Dio2Error> {
        self.prf
            .profile(self.ring_dimension as usize)
            .map_err(|_| Dio2Error::Parameters("the PRF does not fit the ring"))
    }

    /// The label modulus `Q` (the PRF's input modulus).
    pub fn label_modulus(&self) -> usize {
        self.prf.input_modulus
    }

    /// The label height `v` (the PRF's LWR dimension).
    pub fn label_rows(&self) -> usize {
        self.prf.lwr_dimension
    }

    /// `m_v = v log2 Q`: the label matrix has `m_v + 1` columns.
    pub fn label_width(&self) -> usize {
        self.label_rows() * self.label_modulus().ilog2() as usize
    }
}
