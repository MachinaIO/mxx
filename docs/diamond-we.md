# Diamond witness encryption

Diamond witness-encryption sources remain under `crates/we/src/diamond/`, but the module is
temporarily not exported. This compiling-shell state is intentional during the perfect-correctness
migration.

The reusable input-injection preprocessing is active in `mxx-gadgets`, and BGG+-specific
operations remain active in `mxx-bgg`. Re-enabling Diamond WE requires an application-specific
deterministic hard-noise recurrence or a verified checker. Its parameter search must then accept a
candidate only when the probability-zero theorem is verified, the checker and generated parameter
validity predicate both return true, and the separate lattice-security estimate passes.

The retired symbolic simulator is not a supported compatibility path.
