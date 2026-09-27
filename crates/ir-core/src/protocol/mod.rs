//! Protocol declarations, frozen specifications, and workflow connection validation.
//!
//! A [`ProtocolDecl`] links several executable graphs as [`ProtocolStage`]s with artifact
//! bindings. Validation checks each binding's name, type, and availability against the producer,
//! parameter agreement across graphs, and reachability. A [`ClosedProtocolBundle`] holds the
//! workflow, an ideal specification, requirement predicates, a comparator, endpoint bindings,
//! operational decoder targets, and input contracts.
//!
//! [`IdealSpec::new`] and [`PurePredicateSpec::new`] reject every sampler kind in every scope, and
//! a predicate must have exactly one Boolean output. The graph is private and read through
//! `graph()`, so these checks cannot be bypassed.

pub mod bundle;
mod declaration;
mod spec;

pub use bundle::*;
pub use declaration::*;
pub use spec::*;
