import Mxx.Certificate.Identity
import Mxx.Certificate.ProtocolSyntax
import Mxx.Certificate.Derivation
import Mxx.Certificate.OperationalBounds

/-!
The proof-oriented analyzer and symbolic-evaluation imports are retained in their owning source
files but are intentionally not re-exported while gadget decomposition is migrated to explicit,
partial backend layouts.  Those modules still assume a total, one-layout gadget constructor.
The active public certificate surface includes generated IR syntax, protocol declaration syntax,
and the operational derivation and hard-bound checker. `OperationalSemantics` remains in its
owning source while the flat operational facts are migrated; it is intentionally not re-exported
until the later end-to-end correctness-proof milestone. The older recursive symbolic analyzer is
not re-exported.
-/
