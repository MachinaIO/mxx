import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard644
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard645

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound95029
def owner : Owner := ⟨.program ⟨214⟩, ⟨16863⟩⟩
def transferEvent : Nat := 95029
def frameStart : Nat := 94939
def rule : BoundRule := .product (.predecessor 0 95027 .coefficient) (.predecessor 1 95028 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95027 .coefficient)
      LeftAuthority94982.bound (LeftAuthority94982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact94983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94982.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95028 .coefficient)
      LeftAuthority95025.bound (LeftAuthority95025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95025.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95025.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority94982.bound LeftAuthority95025.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94982.bound, LeftAuthority95025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority94982.actual selector witness) * (LeftAuthority95025.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95029

namespace LeftBound95037
def owner : Owner := ⟨.program ⟨214⟩, ⟨16864⟩⟩
def transferEvent : Nat := 95037
def frameStart : Nat := 94939
def rule : BoundRule := .sum [.predecessor 0 95035 .coefficient, .predecessor 1 95036 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95035 .coefficient)
      LeftAuthority95033.bound (LeftAuthority95033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95036 .coefficient)
      LeftBound95029.bound (LeftBound95029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95029.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority95033.bound, LeftBound95029.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95033.bound, LeftBound95029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority95033.actual selector witness, LeftBound95029.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95037

namespace LeftBound95041
def owner : Owner := ⟨.program ⟨214⟩, ⟨25672⟩⟩
def transferEvent : Nat := 95041
def frameStart : Nat := 94939
def rule : BoundRule := .sum [.predecessor 0 95039 .coefficient, .predecessor 1 95040 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95039 .coefficient)
      LeftBound95037.bound (LeftBound95037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95037.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95040 .coefficient)
      LeftBound95018.bound (LeftBound95018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95018.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95037.bound, LeftBound95018.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95037.bound, LeftBound95018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95037.actual selector witness, LeftBound95018.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95041

namespace LeftBound95054
def owner : Owner := ⟨.program ⟨214⟩, ⟨25670⟩⟩
def transferEvent : Nat := 95054
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95052 .coefficient, .predecessor 1 95053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95052 .coefficient)
      LeftBound94899.bound (LeftBound94899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95053 .coefficient)
      LeftBound94882.bound (LeftBound94882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94899.bound, LeftBound94882.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94899.bound, LeftBound94882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94899.actual selector witness, LeftBound94882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95054

namespace LeftBound95057
def owner : Owner := ⟨.program ⟨214⟩, ⟨25670⟩⟩
def transferEvent : Nat := 95057
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 95051 .summary, .result 94889 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95051 .summary)
      LeftBound94901.bound (LeftBound94901.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20168⟩⟩) (rawTerms := some (Proof.Events371.exact95051RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94889 .summary)
      LeftBound94884.bound (LeftBound94884.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25669⟩⟩) (rawTerms := some (Proof.Events370.exact94889RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94884.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94901.bound, LeftBound94884.bound]
def bound : CoeffClass := .finite ⟨352182857248768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94901.bound, LeftBound94884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94901.actual selector witness, LeftBound94884.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95057

namespace LeftBound95061
def owner : Owner := ⟨.program ⟨214⟩, ⟨29786⟩⟩
def transferEvent : Nat := 95061
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95059 .coefficient) (.predecessor 1 95060 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95059 .coefficient)
      LeftBound95054.bound (LeftBound95054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95060 .coefficient)
      LeftAuthority94804.bound (LeftAuthority94804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94804.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95054.bound LeftAuthority94804.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95054.bound, LeftAuthority94804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95054.actual selector witness) * (LeftAuthority94804.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95061

namespace LeftBound95062
def owner : Owner := ⟨.program ⟨214⟩, ⟨29786⟩⟩
def transferEvent : Nat := 95062
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29784⟩⟩]⟩ [⟨.result 94805 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94805 .coefficient)
      LeftAuthority94804.bound (LeftAuthority94804.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29784⟩⟩) (rawTerms := some (Proof.Events370.exact94805RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94804.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority94804.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94804.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95062

namespace LeftBound95063
def owner : Owner := ⟨.program ⟨214⟩, ⟨29786⟩⟩
def transferEvent : Nat := 95063
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95058 .summary) (.transfer 95062) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95058 .summary)
      LeftBound95057.bound (LeftBound95057.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25670⟩⟩) (rawTerms := some (Proof.Events371.exact95058RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95062)
      LeftBound95062.bound (LeftBound95062.actual selector witness) := by
  exact .transfer (LeftBound95062.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95057.bound LeftBound95062.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95057.bound, LeftBound95062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95057.actual selector witness) * (LeftBound95062.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95063

namespace LeftBound95074
def owner : Owner := ⟨.program ⟨214⟩, ⟨22687⟩⟩
def transferEvent : Nat := 95074
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 95072 .coefficient) (.value (.predecessor 1 95073 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95072 .coefficient)
      LeftAuthority95070.bound (LeftAuthority95070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95070.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95073 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority95070.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95070.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95070.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95074

namespace LeftBound95078
def owner : Owner := ⟨.program ⟨214⟩, ⟨22688⟩⟩
def transferEvent : Nat := 95078
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95076 .coefficient) (.predecessor 1 95077 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95076 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95077 .coefficient)
      LeftBound95074.bound (LeftBound95074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95074.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound95074.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound95074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound95074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95078

namespace LeftBound95079
def owner : Owner := ⟨.program ⟨214⟩, ⟨22688⟩⟩
def transferEvent : Nat := 95079
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22685⟩⟩]⟩ [⟨.result 95071 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95071 .coefficient)
      LeftAuthority95070.bound (LeftAuthority95070.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22685⟩⟩) (rawTerms := some (Proof.Events371.exact95071RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95070.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95070.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95070.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95070.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95079

namespace LeftBound95080
def owner : Owner := ⟨.program ⟨214⟩, ⟨22688⟩⟩
def transferEvent : Nat := 95080
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 95079) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95079)
      LeftBound95079.bound (LeftBound95079.actual selector witness) := by
  exact .transfer (LeftBound95079.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound95079.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound95079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound95079.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95080

namespace LeftBound95151
def owner : Owner := ⟨.program ⟨214⟩, ⟨16862⟩⟩
def transferEvent : Nat := 95151
def frameStart : Nat := 95124
def rule : BoundRule := .identity (.predecessor 0 95150 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95150 .coefficient)
      LeftAuthority95148.bound (LeftAuthority95148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95148.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95148.derived selector witness)

def rawBound : CoeffClass := LeftAuthority95148.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority95148.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95151

namespace LeftBound95168
def owner : Owner := ⟨.program ⟨214⟩, ⟨16959⟩⟩
def transferEvent : Nat := 95168
def frameStart : Nat := 95124
def rule : BoundRule := .sum [.predecessor 0 95166 .coefficient, .predecessor 1 95167 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95166 .coefficient)
      LeftBound95151.bound (LeftBound95151.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound95151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95167 .coefficient)
      LeftAuthority95164.bound (LeftAuthority95164.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority95164.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95151.bound, LeftAuthority95164.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95151.bound, LeftAuthority95164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95151.actual selector witness, LeftAuthority95164.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95168

namespace LeftBound95171
def owner : Owner := ⟨.program ⟨214⟩, ⟨16960⟩⟩
def transferEvent : Nat := 95171
def frameStart : Nat := 95124
def rule : BoundRule := .identity (.predecessor 0 95170 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95170 .coefficient)
      LeftBound95168.bound (LeftBound95168.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound95168.derived selector witness)

def rawBound : CoeffClass := LeftBound95168.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound95168.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95171

namespace LeftBound95177
def owner : Owner := ⟨.program ⟨214⟩, ⟨16961⟩⟩
def transferEvent : Nat := 95177
def frameStart : Nat := 95124
def rule : BoundRule := .product (.predecessor 0 95175 .coefficient) (.predecessor 1 95176 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95175 .coefficient)
      LeftAuthority95173.bound (LeftAuthority95173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95176 .coefficient)
      LeftBound95171.bound (LeftBound95171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95171.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority95173.bound LeftBound95171.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95173.bound, LeftBound95171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority95173.actual selector witness) * (LeftBound95171.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95177

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
