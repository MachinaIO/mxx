use keccak_asm::Keccak256;
use mxx::{
    input_injector::DiamondInjector,
    matrix::dcrt_poly::DCRTPolyMatrix,
    poly::dcrt::params::DCRTPolyParams,
    sampler::{
        hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
        uniform::DCRTPolyUniformSampler,
    },
    utils::bigdecimal_bits_ceil,
};
use std::{
    env,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};
use tracing::info;

type TestInjector = DiamondInjector<
    DCRTPolyMatrix,
    DCRTPolyUniformSampler,
    DCRTPolyHashSampler<Keccak256>,
    DCRTPolyTrapdoorSampler,
>;

const DIAMOND_INJECTOR_HASH_KEY: [u8; 32] = [29u8; 32];
const DIAMOND_INJECTOR_DECODER_COUNT: usize = 1;
const DIAMOND_INJECTOR_TRAPDOOR_SIGMA: f64 = 4.578;
const DIAMOND_INJECTOR_ERROR_SIGMA: f64 = 4.578;
const DEFAULT_RING_DIM: u32 = 1u32 << 16;
const DEFAULT_CRT_BITS: usize = 32;
const DEFAULT_INPUT_COUNT: usize = 32;
const DEFAULT_DIGIT_BITS: u32 = 8;
const DEFAULT_MIN_CRT_DEPTH: usize = 10;
const DEFAULT_MAX_CRT_DEPTH: usize = 64;

#[derive(Debug, Clone)]
struct ErrorPoint {
    q_bits: usize,
    max_error_bits: u64,
}

fn env_or_default_u32(key: &str, default: u32) -> u32 {
    env::var(key)
        .map(|raw| raw.parse::<u32>().unwrap_or_else(|e| panic!("{key} must be a valid u32: {e}")))
        .unwrap_or(default)
}

fn env_or_default_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .map(|raw| {
            raw.parse::<usize>().unwrap_or_else(|e| panic!("{key} must be a valid usize: {e}"))
        })
        .unwrap_or(default)
}

fn output_path_from_env() -> PathBuf {
    env::var("DIAMOND_INJECTOR_ERROR_PLOT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./test_data/diamond_injector_q_bits_vs_max_error.svg"))
}

fn normalize_bits(value: f64, min: f64, max: f64) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        return 0.5;
    }
    (value - min) / (max - min)
}

fn u64_to_usize(value: u64, context: &str) -> usize {
    usize::try_from(value)
        .unwrap_or_else(|_| panic!("{context}={value} must fit into usize for SVG plotting"))
}

fn write_svg_plot(
    points: &[ErrorPoint],
    output_path: &Path,
    ring_dim: u32,
    crt_bits: usize,
    base_bits: u32,
    input_count: usize,
    digit_bits: u32,
) {
    assert!(!points.is_empty(), "error plot requires at least one point");

    let width = 1280i32;
    let height = 720i32;
    let left_margin = 120i32;
    let right = 40i32;
    let top = 80i32;
    let bottom = 100i32;
    let available_plot_width = width - left_margin - right;
    let available_plot_height = height - top - bottom;

    let min_q_bits = points.iter().map(|point| point.q_bits).min().expect("points are non-empty");
    let max_q_bits = points.iter().map(|point| point.q_bits).max().expect("points are non-empty");
    let min_error_bits =
        points.iter().map(|point| point.max_error_bits).min().expect("points are non-empty");
    let max_error_bits =
        points.iter().map(|point| point.max_error_bits).max().expect("points are non-empty");
    let major_tick_bits =
        u64::try_from(crt_bits).expect("crt_bits must fit into u64 for plot tick computation");
    assert!(major_tick_bits > 0, "crt_bits must be positive for plot tick computation");
    let minor_tick_bits = major_tick_bits as f64 / 8.0;

    let x_min_bits = (min_q_bits as u64 / major_tick_bits) * major_tick_bits;
    let x_max_bits = (max_q_bits as u64).div_ceil(major_tick_bits) * major_tick_bits;
    let x_max_bits =
        if x_min_bits == x_max_bits { x_max_bits + major_tick_bits } else { x_max_bits };
    let y_min_bits = (min_error_bits / major_tick_bits) * major_tick_bits;
    let y_max_bits = (max_error_bits.div_ceil(major_tick_bits) + 1) * major_tick_bits;
    let y_max_bits =
        if y_min_bits == y_max_bits { y_max_bits + major_tick_bits } else { y_max_bits };
    let x_major_tick_count = usize::try_from((x_max_bits - x_min_bits) / major_tick_bits)
        .expect("x major tick count must fit into usize");
    let y_major_tick_count = usize::try_from((y_max_bits - y_min_bits) / major_tick_bits)
        .expect("y major tick count must fit into usize");
    let pixels_per_major = (available_plot_width as f64 / x_major_tick_count as f64)
        .min(available_plot_height as f64 / y_major_tick_count as f64);
    let plot_width = (pixels_per_major * x_major_tick_count as f64).round() as i32;
    let plot_height = (pixels_per_major * y_major_tick_count as f64).round() as i32;
    let plot_left = left_margin + (available_plot_width - plot_width) / 2;
    let plot_top = top + (available_plot_height - plot_height) / 2;

    let x_min_bits_f64 = x_min_bits as f64;
    let x_max_bits_f64 = x_max_bits as f64;
    let y_min_bits_f64 = y_min_bits as f64;
    let y_max_bits_f64 = y_max_bits as f64;
    let x_at = |q_bits: usize| -> i32 {
        if x_min_bits == x_max_bits {
            return plot_left + plot_width / 2;
        }
        let ratio = normalize_bits(q_bits as f64, x_min_bits_f64, x_max_bits_f64);
        plot_left + (ratio * plot_width as f64).round() as i32
    };
    let x_at_bits = |q_bits: f64| -> i32 {
        let ratio = normalize_bits(q_bits, x_min_bits_f64, x_max_bits_f64);
        plot_left + (ratio * plot_width as f64).round() as i32
    };
    let y_at = |error_bits: u64| -> i32 {
        let ratio = normalize_bits(error_bits as f64, y_min_bits_f64, y_max_bits_f64);
        plot_top + ((1.0 - ratio) * plot_height as f64).round() as i32
    };
    let y_at_bits = |error_bits: f64| -> i32 {
        let ratio = normalize_bits(error_bits, y_min_bits_f64, y_max_bits_f64);
        plot_top + ((1.0 - ratio) * plot_height as f64).round() as i32
    };

    let mut svg = String::new();
    writeln!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"##
    )
    .expect("svg header should format");
    writeln!(svg, r##"<rect width="100%" height="100%" fill="#ffffff"/>"##)
        .expect("svg background should format");
    writeln!(
        svg,
        r##"<text x="{x}" y="36" font-size="28" font-family="monospace" text-anchor="middle" fill="#111111">DiamondInjector q_bits vs max_error_bits</text>"##,
        x = width / 2
    )
    .expect("svg title should format");
    writeln!(
        svg,
        r##"<text x="{x}" y="60" font-size="16" font-family="monospace" text-anchor="middle" fill="#444444">ring_dim={ring_dim}, crt_bits={crt_bits}, base_bits={base_bits}, input_count={input_count}, digit_bits={digit_bits}</text>"##,
        x = width / 2
    )
    .expect("svg subtitle should format");

    for major_idx in 0..x_major_tick_count {
        let segment_start_bits = x_min_bits + (major_idx as u64) * major_tick_bits;
        for minor_idx in 1..8 {
            let minor_bits = segment_start_bits as f64 + (minor_idx as f64) * minor_tick_bits;
            let x = x_at_bits(minor_bits);
            writeln!(
                svg,
                r##"<line x1="{x}" y1="{axis_y}" x2="{x}" y2="{tick_y}" stroke="#9ca3af" stroke-width="1"/>"##,
                axis_y = plot_top + plot_height,
                tick_y = plot_top + plot_height + 6
            )
            .expect("svg x-axis minor tick should format");
        }
    }

    for major_idx in 0..y_major_tick_count {
        let segment_start_bits = y_min_bits + (major_idx as u64) * major_tick_bits;
        for minor_idx in 1..8 {
            let minor_bits = segment_start_bits as f64 + (minor_idx as f64) * minor_tick_bits;
            let y = y_at_bits(minor_bits);
            writeln!(
                svg,
                r##"<line x1="{tick_x}" y1="{y}" x2="{axis_x}" y2="{y}" stroke="#9ca3af" stroke-width="1"/>"##,
                tick_x = plot_left - 6,
                axis_x = plot_left
            )
            .expect("svg y-axis minor tick should format");
        }
    }

    for major_idx in 0..=y_major_tick_count {
        let tick_value = y_min_bits + (major_idx as u64) * major_tick_bits;
        let y = y_at(tick_value);
        writeln!(
            svg,
            r##"<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>"##,
            x1 = plot_left,
            x2 = plot_left + plot_width
        )
        .expect("svg y-grid should format");
        writeln!(
            svg,
            r##"<text x="{x}" y="{label_y}" font-size="13" font-family="monospace" text-anchor="end" fill="#374151">{label}</text>"##,
            x = plot_left - 12,
            label_y = y + 4,
            label = tick_value
        )
        .expect("svg y-label should format");
    }

    for major_idx in 0..=x_major_tick_count {
        let q_bits = x_min_bits + (major_idx as u64) * major_tick_bits;
        let q_bits = u64_to_usize(q_bits, "x_axis_tick_bits");
        let x = x_at(q_bits);
        writeln!(
            svg,
            r##"<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#f3f4f6" stroke-width="1"/>"##,
            y1 = plot_top,
            y2 = plot_top + plot_height
        )
        .expect("svg x-grid should format");
        writeln!(
            svg,
            r##"<text x="{x}" y="{y}" font-size="13" font-family="monospace" text-anchor="middle" fill="#374151">{q_bits}</text>"##,
            y = plot_top + plot_height + 24
        )
        .expect("svg x-label should format");
    }

    writeln!(
        svg,
        r##"<line x1="{x1}" y1="{axis_y}" x2="{x2}" y2="{axis_y}" stroke="#111111" stroke-width="2"/>"##,
        x1 = plot_left,
        axis_y = plot_top + plot_height,
        x2 = plot_left + plot_width
    )
    .expect("svg x-axis should format");
    writeln!(
        svg,
        r##"<line x1="{axis_x}" y1="{y1}" x2="{axis_x}" y2="{y2}" stroke="#111111" stroke-width="2"/>"##,
        axis_x = plot_left,
        y1 = plot_top,
        y2 = plot_top + plot_height
    )
    .expect("svg y-axis should format");
    writeln!(
        svg,
        r##"<text x="{x}" y="{y}" font-size="16" font-family="monospace" text-anchor="middle" fill="#111111">q_bits = crt_bits * crt_depth</text>"##,
        x = plot_left + plot_width / 2,
        y = plot_top + plot_height + 52
    )
    .expect("svg x-axis title should format");
    writeln!(
        svg,
        r##"<text x="{x}" y="{y}" font-size="16" font-family="monospace" text-anchor="middle" fill="#111111" transform="rotate(-90 {x} {y})">max_error_bits</text>"##,
        x = plot_left - 54,
        y = plot_top + plot_height / 2
    )
    .expect("svg y-axis title should format");

    let guideline_start_bits = std::cmp::max(x_min_bits, y_min_bits);
    let guideline_end_bits = std::cmp::min(x_max_bits, y_max_bits);
    if guideline_start_bits < guideline_end_bits {
        writeln!(
            svg,
            r##"<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111111" stroke-width="2" stroke-dasharray="10 8"/>"##,
            x1 = x_at_bits(guideline_start_bits as f64),
            y1 = y_at_bits(guideline_start_bits as f64),
            x2 = x_at_bits(guideline_end_bits as f64),
            y2 = y_at_bits(guideline_end_bits as f64)
        )
        .expect("svg y=x guideline should format");
    }

    let polyline_points = points
        .iter()
        .map(|point| format!("{},{}", x_at(point.q_bits), y_at(point.max_error_bits)))
        .collect::<Vec<_>>()
        .join(" ");
    writeln!(
        svg,
        r##"<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{polyline_points}"/>"##
    )
    .expect("svg polyline should format");

    for point in points {
        let x = x_at(point.q_bits);
        let y = y_at(point.max_error_bits);
        writeln!(
            svg,
            r##"<circle cx="{x}" cy="{y}" r="5" fill="#1d4ed8" stroke="#ffffff" stroke-width="2"/>"##
        )
        .expect("svg point should format");
    }

    if let Some(first_below_diagonal) =
        points.iter().find(|point| point.max_error_bits < point.q_bits as u64)
    {
        let x = x_at(first_below_diagonal.q_bits);
        let y = y_at(first_below_diagonal.max_error_bits);
        writeln!(
            svg,
            r##"<text x="{x}" y="{y}" font-size="13" font-family="monospace" text-anchor="middle" fill="#1d4ed8">q_bits={q_bits}</text>"##,
            y = y + 18,
            q_bits = first_below_diagonal.q_bits
        )
        .expect("svg diagonal-crossing caption should format");
    }

    svg.push_str("</svg>\n");

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|e| {
            panic!(
                "failed to create DiamondInjector error-plot directory {}: {e}",
                parent.display()
            )
        });
    }
    fs::write(output_path, svg).unwrap_or_else(|e| {
        panic!("failed to write DiamondInjector error plot {}: {e}", output_path.display())
    });
}

#[test]
fn test_diamond_injector_q_bits_vs_max_error_plot_generates_svg() {
    let _ = tracing_subscriber::fmt::try_init();

    let ring_dim = env_or_default_u32("DIAMOND_INJECTOR_RING_DIM", DEFAULT_RING_DIM);
    let crt_bits = env_or_default_usize("DIAMOND_INJECTOR_CRT_BITS", DEFAULT_CRT_BITS);
    let input_count = env_or_default_usize("DIAMOND_INJECTOR_INPUT_COUNT", DEFAULT_INPUT_COUNT);
    let digit_bits = env_or_default_u32("DIAMOND_INJECTOR_DIGIT_BITS", DEFAULT_DIGIT_BITS);
    let min_crt_depth =
        env_or_default_usize("DIAMOND_INJECTOR_MIN_CRT_DEPTH", DEFAULT_MIN_CRT_DEPTH);
    let max_crt_depth =
        env_or_default_usize("DIAMOND_INJECTOR_MAX_CRT_DEPTH", DEFAULT_MAX_CRT_DEPTH);
    assert!(
        min_crt_depth <= max_crt_depth,
        "DIAMOND_INJECTOR_MIN_CRT_DEPTH must be <= DIAMOND_INJECTOR_MAX_CRT_DEPTH"
    );

    let base_bits = u32::try_from(crt_bits.div_ceil(2))
        .expect("ceil(crt_bits / 2) must fit into u32 for DCRTPolyParams");
    let input_base = 1usize
        .checked_shl(digit_bits)
        .unwrap_or_else(|| panic!("DIAMOND_INJECTOR_DIGIT_BITS={} is too large", digit_bits));
    let output_path = output_path_from_env();

    let mut points = Vec::with_capacity(max_crt_depth - min_crt_depth + 1);
    for crt_depth in min_crt_depth..=max_crt_depth {
        let params = DCRTPolyParams::new(ring_dim, crt_depth, crt_bits, base_bits);
        let injector = TestInjector::new(
            params,
            DIAMOND_INJECTOR_HASH_KEY,
            input_count,
            input_base,
            DIAMOND_INJECTOR_DECODER_COUNT,
            DIAMOND_INJECTOR_TRAPDOOR_SIGMA,
            DIAMOND_INJECTOR_ERROR_SIGMA,
            std::env::temp_dir(),
        );
        let simulated = injector.simulate_output_error_bounds();
        let max_input_error = simulated
            .input_errors
            .iter()
            .map(|norm| norm.poly_norm.norm.clone())
            .max()
            .expect("input error list must be non-empty");
        let max_decoder_error = simulated
            .decoder_errors
            .iter()
            .map(|norm| norm.poly_norm.norm.clone())
            .max()
            .expect("decoder error list must be non-empty");
        let max_error = std::cmp::max(max_input_error.clone(), max_decoder_error.clone());
        let max_error_bits = bigdecimal_bits_ceil(&max_error);
        let q_bits =
            crt_bits.checked_mul(crt_depth).expect("q_bits overflow in DiamondInjector plot test");

        info!(
            "diamond injector plot point: crt_depth={crt_depth}, q_bits={q_bits}, max_input_bits={}, max_decoder_bits={}, max_total_bits={}",
            bigdecimal_bits_ceil(&max_input_error),
            bigdecimal_bits_ceil(&max_decoder_error),
            bigdecimal_bits_ceil(&max_error),
        );

        points.push(ErrorPoint { q_bits, max_error_bits });
    }

    write_svg_plot(&points, &output_path, ring_dim, crt_bits, base_bits, input_count, digit_bits);

    let written =
        fs::read_to_string(&output_path).unwrap_or_else(|e| panic!("failed to re-read plot: {e}"));
    assert!(
        written.contains("<svg") &&
            written.contains("DiamondInjector q_bits vs max_error_bits") &&
            written.contains("stroke=\"#9ca3af\"") &&
            !written.contains("error_bits=") &&
            written.contains("stroke-dasharray=\"10 8\""),
        "generated plot should be a readable SVG with the expected title"
    );
    if let Some(first_below_diagonal) =
        points.iter().find(|point| point.max_error_bits < point.q_bits as u64)
    {
        assert!(
            written.contains(&format!("q_bits={}", first_below_diagonal.q_bits)),
            "generated plot should annotate the first point below the y=x guideline"
        );
    } else {
        assert!(
            !written.contains("q_bits="),
            "generated plot should not add a q_bits caption unless a point falls below the y=x guideline"
        );
    }
    info!(
        "diamond injector error plot saved to {} with {} points",
        output_path.display(),
        points.len()
    );
}
