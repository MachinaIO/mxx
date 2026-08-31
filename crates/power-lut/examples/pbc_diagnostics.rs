use std::{collections::BTreeMap, env};

use mxx_power_lut::pbc::{
    PbcDiagnosticAggregator, PbcDiagnosticSample, PbcError, PbcParameters, PbcProfile, PbcRootSeed,
    measure_key_layout,
};
use rand::{Rng, seq::SliceRandom};

fn usage() -> &'static str {
    "usage: pbc_diagnostics --nu N --h H --trials N --profile Conservative|PaperEvaluation [--width-limit N] [--format text|json]\n       pbc_diagnostics --nu N --h H --trials N --profile Custom --c C --k K --max-seed-attempts N [--width-limit N] [--format text|json]"
}

fn parse<T: std::str::FromStr>(
    options: &BTreeMap<String, String>,
    name: &str,
) -> Result<T, String> {
    options
        .get(name)
        .ok_or_else(|| format!("missing --{name}\n{}", usage()))?
        .parse()
        .map_err(|_| format!("invalid --{name}\n{}", usage()))
}

fn parse_options() -> Result<(PbcParameters, usize, String), String> {
    const ALLOWED: &[&str] =
        &["nu", "h", "trials", "profile", "width-limit", "format", "c", "k", "max-seed-attempts"];
    let mut options = BTreeMap::new();
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let Some(name) = argument.strip_prefix("--") else {
            return Err(format!("unexpected argument {argument}\n{}", usage()));
        };
        if !ALLOWED.contains(&name) {
            return Err(format!("unsupported option --{name}\n{}", usage()));
        }
        if options.contains_key(name) {
            return Err(format!("duplicate --{name}\n{}", usage()));
        }
        let value =
            arguments.next().ok_or_else(|| format!("missing value for --{name}\n{}", usage()))?;
        if value.starts_with("--") {
            return Err(format!("missing value for --{name}\n{}", usage()));
        }
        options.insert(name.to_owned(), value);
    }
    let universe_size = parse(&options, "nu")?;
    let support_weight = parse(&options, "h")?;
    let trials = parse(&options, "trials")?;
    if trials == 0 {
        return Err("--trials must be positive".to_owned());
    }
    let profile_name =
        options.get("profile").ok_or_else(|| format!("missing --profile\n{}", usage()))?;
    let bucket_width_limit =
        options.get("width-limit").map(|_| parse(&options, "width-limit")).transpose()?;
    let profile = match profile_name.as_str() {
        "Conservative" => PbcProfile::Conservative,
        "PaperEvaluation" => PbcProfile::PaperEvaluation,
        "Custom" => PbcProfile::Custom,
        _ => return Err(format!("unsupported profile {profile_name}\n{}", usage())),
    };
    let parameters = match profile {
        PbcProfile::Conservative => {
            reject_custom_options(&options)?;
            PbcParameters::conservative(universe_size, support_weight)
        }
        PbcProfile::PaperEvaluation => {
            reject_custom_options(&options)?;
            PbcParameters::paper_evaluation(universe_size, support_weight)
        }
        PbcProfile::Custom => PbcParameters::custom(
            universe_size,
            support_weight,
            parse(&options, "c")?,
            parse(&options, "k")?,
            parse(&options, "max-seed-attempts")?,
            bucket_width_limit,
        ),
    };
    let parameters = if profile == PbcProfile::Custom {
        parameters
    } else if bucket_width_limit.is_some() {
        PbcParameters { bucket_width_limit, ..parameters }
    } else {
        parameters
    };
    parameters.validate().map_err(|error| error.to_string())?;
    let format = options.get("format").cloned().unwrap_or_else(|| "text".to_owned());
    if format != "text" && format != "json" {
        return Err("--format must be text or json".to_owned());
    }
    Ok((parameters, trials, format))
}

fn reject_custom_options(options: &BTreeMap<String, String>) -> Result<(), String> {
    for option in ["c", "k", "max-seed-attempts"] {
        if options.contains_key(option) {
            return Err(format!("--{option} is valid only with --profile Custom\n{}", usage()));
        }
    }
    Ok(())
}

fn random_support(universe_size: usize, support_weight: usize) -> Vec<usize> {
    let mut coordinates: Vec<usize> = (0..universe_size).collect();
    coordinates.shuffle(&mut rand::rng());
    coordinates.truncate(support_weight);
    coordinates.sort_unstable();
    coordinates
}

fn fresh_root_seed() -> PbcRootSeed {
    let mut bytes = [0_u8; 32];
    rand::rng().fill(&mut bytes);
    PbcRootSeed(bytes)
}

fn run_trial(parameters: &PbcParameters) -> Result<PbcDiagnosticSample, PbcError> {
    // Sample the support before the independently fresh root seed, matching
    // the supported honest key-generation order.
    let support = random_support(parameters.universe_size, parameters.support_weight);
    let root_seed = fresh_root_seed();
    measure_key_layout(parameters, root_seed, &support)
}

fn main() -> Result<(), String> {
    let (parameters, trials, format) = parse_options()?;
    let mut aggregate = PbcDiagnosticAggregator::new(parameters.clone());
    for _ in 0..trials {
        aggregate.record(run_trial(&parameters).map_err(|error| error.to_string())?);
    }
    let report = aggregate.finish();
    if format == "json" {
        println!("{}", report.to_json().map_err(|error| error.to_string())?);
    } else {
        println!("{}", report.to_text());
    }
    Ok(())
}
