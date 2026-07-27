pub fn diamond_io_eval_relation_asserts() -> bool {
    std::env::var("MXX_DIAMOND_IO_EVAL_RELATION_ASSERTS")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}
