//! Shared unit-test helpers: small, environment-overridable parameters and
//! branching programs.
use crate::{
    BranchingProgram, PeClass, PeParameterRequest, PeParameters, budget::KeyWidthPolicy,
    params::TrapdoorMode,
};
use mxx_backends::{
    authority::CpuExecution,
    backend::poly::cpu_backend,
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::bounds::TrapdoorPreimageCutoffPolicy,
};
use mxx_exponent_lut::rns_bgg::RnsBggGaussian;

fn read<T: std::str::FromStr>(name: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    std::env::var(format!("MXX_PE_TEST_{name}")).map_or(default, |value| value.parse().unwrap())
}

/// Test parameters of `class`: ring dimension, CRT depth and prime bits,
/// base bits, trapdoor sigma, and the error, flooding, and programming
/// sigmas overridable by `MXX_PE_TEST_{RING_DIMENSION,CRT_DEPTH,CRT_BITS,
/// BASE_BITS,TRAPDOOR_SIGMA,ERROR_SIGMA,FLOODING_SIGMA,PROGRAMMING_SIGMA}`,
/// and the instantiation by `MXX_PE_TEST_TRAPDOOR_INSTANTIATION`
/// (`computational`, the default, or `statistical`, see
/// [`statistical_parameters`]). They are functional, not secure. By default
/// every error (`chi` and `chi_fl`) is zero, so the evaluation is exact, the
/// simulated bounds are zero, and correctness holds without a noise margin;
/// the entries of `Z_t` are encryption randomness, not noise, and keep sigma
/// one.
pub(crate) fn parameters(class: PeClass) -> PeParameters {
    match read("TRAPDOOR_INSTANTIATION", "computational".to_owned()).as_str() {
        "computational" => computational_parameters(class),
        "statistical" => statistical_parameters(class),
        other => panic!("unknown trapdoor instantiation {other}"),
    }
}

fn level() -> DCRTPolyParams {
    level_at(read("CRT_DEPTH", 4))
}

fn level_at(depth: usize) -> DCRTPolyParams {
    DCRTPolyParams::new(
        read("RING_DIMENSION", 16),
        depth,
        read("CRT_BITS", 30),
        read("BASE_BITS", 10),
        None,
        None,
    )
}

fn contract(name: &str, default: &str) -> RnsBggGaussian {
    RnsBggGaussian::with_default_cutoff(read(name, default.to_owned()).parse().unwrap()).unwrap()
}

/// The computational test parameters of [`parameters`].
pub(crate) fn computational_parameters(class: PeClass) -> PeParameters {
    computational_parameters_at(class, level())
}

/// [`computational_parameters`] over `level`.
fn computational_parameters_at(class: PeClass, level: DCRTPolyParams) -> PeParameters {
    PeParameters::new(
        class,
        level,
        read("TRAPDOOR_SIGMA", 4.578),
        contract("ERROR_SIGMA", "0"),
        contract("FLOODING_SIGMA", "0"),
        contract("PROGRAMMING_SIGMA", "1"),
        TrapdoorPreimageCutoffPolicy::Default,
    )
    .unwrap()
}

/// Statistical test parameters: the contracts of [`parameters`] with a
/// statistical trapdoor of `m̄ = MXX_PE_TEST_M_BAR` (default `2 D`) and the
/// minimum key width. The production selection rejects their zero widths, so
/// this fixture bypasses the budget ([`PeParameters::statistical_unchecked`]).
pub(crate) fn statistical_parameters(class: PeClass) -> PeParameters {
    statistical_parameters_at(class, level())
}

/// [`statistical_parameters`] over `level`.
fn statistical_parameters_at(class: PeClass, level: DCRTPolyParams) -> PeParameters {
    let m_bar = read("M_BAR", 2 * level.modulus_digits());
    PeParameters::statistical_unchecked(
        class,
        level,
        read("TRAPDOOR_SIGMA", 4.578),
        m_bar,
        contract("ERROR_SIGMA", "0"),
        contract("FLOODING_SIGMA", "0"),
        contract("PROGRAMMING_SIGMA", "1"),
    )
    .unwrap()
}

/// The test parameters of `class` at the smallest CRT depth from `4` whose simulated bound
/// guarantees correctness after `adapt` (which may enable options that add noise); over a
/// statistical trapdoor if `statistical`.
pub(crate) fn correct_parameters(
    class: PeClass,
    statistical: bool,
    adapt: impl Fn(PeParameters) -> PeParameters,
) -> PeParameters {
    (4..=16)
        .map(|depth| {
            let level = level_at(depth);
            adapt(if statistical {
                statistical_parameters_at(class, level)
            } else {
                computational_parameters_at(class, level)
            })
        })
        .find(|parameters| {
            crate::noise::simulate(parameters, &crate::BranchingProgram::longest(class))
                .unwrap()
                .decrypts()
        })
        .expect("a CRT depth up to 16 decrypts")
}

/// A statistical selection request of `class` with small, insecure sizes; the
/// key-width policy and `σ_R` are overridable by
/// `MXX_PE_TEST_{KEY_WIDTH_POLICY,RANDOMNESS_SIGMA}` (`practical` or `strict`).
pub(crate) fn statistical_request(class: PeClass) -> PeParameterRequest {
    PeParameterRequest {
        class,
        ring_dimension: read("RING_DIMENSION", 16),
        crt_bits: read("CRT_BITS", 50),
        base_bits: read("BASE_BITS", 25),
        crt_depths: 1..=8,
        trapdoor_instantiation: TrapdoorMode::Statistical,
        key_width_policy: match read("KEY_WIDTH_POLICY", "practical".to_owned()).as_str() {
            "practical" => KeyWidthPolicy::Practical,
            "strict" => KeyWidthPolicy::Strict,
            other => panic!("unknown key width policy {other}"),
        },
        trapdoor_sigma: read("TRAPDOOR_SIGMA", 4.578),
        error_sigma: "3.19".parse().unwrap(),
        flooding_sigma: "3.19".parse().unwrap(),
        enforce_flooding: true,
        programming_sigma: "64".parse().unwrap(),
        randomness_sigma: read("RANDOMNESS_SIGMA", "64".to_owned()).parse().unwrap(),
        statistical_security_bits: crate::budget::DEFAULT_STATISTICAL_BITS,
        flooding_statistical_bits: crate::budget::DEFAULT_FLOODING_BITS,
        key_queries: 1 << 20,
        max_anchor_columns: 4096,
        lut_radix: None,
        compact_encodings: false,
        failure_exponent: None,
        lattice_security_bits: None,
    }
}

/// A CPU execution registering the one ring of `parameters`.
pub(crate) fn execution(parameters: &PeParameters) -> CpuExecution {
    CpuExecution::new(cpu_backend(parameters.rings().unwrap()))
}

/// The comparison DFA of `x > threshold` over bits, most significant first,
/// with states `0 = equal`, `1 = greater`, `2 = smaller`; the final table
/// outputs zero (authorizes) exactly on `greater`.
pub(crate) fn greater_than(class: PeClass, threshold: &[usize]) -> BranchingProgram {
    let step = |bit: usize, last: bool| {
        let mut table = vec![0; 6];
        for x in 0..2 {
            for z in 0..3 {
                let next = match z {
                    0 if x > bit => 1,
                    0 if x < bit => 2,
                    state => state,
                };
                table[z + 3 * x] = if last { usize::from(next != 1) } else { next };
            }
        }
        table
    };
    let length = threshold.len();
    BranchingProgram::new(
        class,
        (0..length).collect(),
        threshold.iter().enumerate().map(|(i, bit)| step(*bit, i + 1 == length)).collect(),
    )
    .unwrap()
}

/// The equality DFA of `x = pattern` over bits, with states `0 = equal so
/// far` and `1 = differs`; the final table outputs zero (authorizes) exactly
/// when every symbol matched.
pub(crate) fn equals(class: PeClass, pattern: &[usize]) -> BranchingProgram {
    let length = pattern.len();
    BranchingProgram::new(
        class,
        (0..length).collect(),
        pattern
            .iter()
            .map(|bit| {
                // Index `z + 2 x`: the state stays 0 only on a match.
                (0..4).map(|index| usize::from(index % 2 == 1 || index / 2 != *bit)).collect()
            })
            .collect(),
    )
    .unwrap()
}

/// A random program of the class's maximal length whose output on `attribute` is `value`: random
/// reads and intermediate tables, and a random final table with its reached entry set.
pub(crate) fn random_program(
    class: PeClass,
    attribute: &[usize],
    value: usize,
) -> BranchingProgram {
    let random = |bound: usize| rand::random::<u32>() as usize % bound;
    let entries = class.state_width() * class.alphabet_size();
    let steps = class.max_steps();
    let reads = (0..steps).map(|_| random(class.attribute_length())).collect::<Vec<_>>();
    let mut tables = (0..steps)
        .map(|step| {
            let bound = if step + 1 == steps { 2 } else { class.state_width() };
            (0..entries).map(|_| random(bound)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut state = 0;
    for (read, table) in reads.iter().zip(&tables).take(steps - 1) {
        state = table[state + class.state_width() * attribute[*read]];
    }
    tables[steps - 1][state + class.state_width() * attribute[reads[steps - 1]]] = value;
    let program = BranchingProgram::new(class, reads, tables).unwrap();
    assert_eq!(program.evaluate(attribute).unwrap(), value);
    program
}
