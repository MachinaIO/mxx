//! Shared unit-test helpers: small, environment-overridable parameters and
//! branching programs.
use crate::{BranchingProgram, PeClass, PeParameters};
use mxx_backends::{
    authority::CpuExecution, backend::poly::cpu_backend, poly::dcrt::params::DCRTPolyParams,
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
/// BASE_BITS,TRAPDOOR_SIGMA,ERROR_SIGMA,FLOODING_SIGMA,PROGRAMMING_SIGMA}`.
/// They are functional, not secure. By default every error (`chi` and
/// `chi_fl`) is zero, so the evaluation is exact, the simulated bounds are
/// zero, and correctness holds without a noise margin; the entries of `Z_t`
/// are encryption randomness, not noise, and keep sigma one.
pub(crate) fn parameters(class: PeClass) -> PeParameters {
    let level = DCRTPolyParams::new(
        read("RING_DIMENSION", 16),
        read("CRT_DEPTH", 4),
        read("CRT_BITS", 30),
        read("BASE_BITS", 10),
        None,
        None,
    );
    let contract = |name: &str, default: &str| {
        RnsBggGaussian::with_default_cutoff(read(name, default.to_owned()).parse().unwrap())
            .unwrap()
    };
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

/// A CPU execution registering the one ring of `parameters`.
pub(crate) fn execution(parameters: &PeParameters) -> CpuExecution {
    CpuExecution::new(cpu_backend([parameters.level().clone()]))
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
