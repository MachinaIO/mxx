import MxxSecurity.Sampling
import MxxSecurity.Assumptions
import MxxSecurity.Lemmas
import MxxSecurity.Distribution
import MxxSecurity.Tape
import MxxSecurity.TapeBounds
import MxxSecurity.Uniform
import MxxSecurity.Structure
import MxxSecurity.IR.Lemmas
import MxxSecurity.IR.Examples

/-!
# MxxSecurity

Protocol-independent security tools for the mxx DSL: the sampled computations generated graphs run
in (`MxxSecurity.Sampling`), the allowed hardness assumptions (`MxxSecurity.Assumptions`), and
lemmas about the distributions of sampled computations, sampled computations as functions of a
sampling tape with tail bounds over it (`MxxSecurity.Tape`, `MxxSecurity.TapeBounds`), and the IR as Lean data with a functional
semantics and a cost (`MxxSecurity.IR`), so a statement can quantify over IR programs of bounded
cost. Protocol definitions, such as public-key
FHE, live with their protocol (`crates/fhe/lean/spec`).
-/
