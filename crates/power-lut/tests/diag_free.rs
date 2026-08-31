mod refresh_parameter_search_support;

#[test]
fn diag_free() {
    let config = refresh_parameter_search_support::SearchConfig::reviewed();
    let result = refresh_parameter_search_support::prepare_candidate(
        &config,
        refresh_parameter_search_support::Candidate { crt_depth: 30, log_ring_dimension: 15 },
    );
    match result {
        Ok(prepared) => {
            eprintln!("DIAG prepare ok");
            let checked = refresh_parameter_search_support::check_refresh_bundle(
                &config,
                prepared.candidate,
                &prepared,
            );
            eprintln!("DIAG check result: {checked:?}");
        }
        Err(error) => eprintln!("DIAG prepare error: {error}"),
    }
}
