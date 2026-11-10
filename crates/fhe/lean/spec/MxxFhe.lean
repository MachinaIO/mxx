import MxxFhe.Defs
import MxxFhe.Hybrid
import MxxFhe.Bounded

/-!
# MxxFhe

The implementation-agnostic definitions of public-key homomorphic encryption (`MxxFhe.Defs`): the
syntax, correctness of a circuit, and IND-CPA security
relative to the allowed assumptions of `MxxSecurity`, with the reduction template the statements
use. `MxxFhe.Hybrid` proves the hybrid
argument once. `mxx_fhe::spec::lean` instantiates these definitions with generated computations.
-/
