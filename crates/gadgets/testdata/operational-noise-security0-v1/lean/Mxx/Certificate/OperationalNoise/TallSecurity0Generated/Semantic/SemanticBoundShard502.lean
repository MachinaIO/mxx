import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard095
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard096
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard501

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound73513
def owner : Owner := ⟨.program ⟨214⟩, ⟨10476⟩⟩
def transferEvent : Nat := 73513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73511 .coefficient, .predecessor 1 73512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73511 .coefficient)
      LeftBound73508.bound (LeftBound73508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73512 .coefficient)
      LeftBound73503.bound (LeftBound73503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73508.bound, LeftBound73503.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73508.bound, LeftBound73503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73508.actual selector witness, LeftBound73503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73513

namespace LeftBound73517
def owner : Owner := ⟨.program ⟨214⟩, ⟨10477⟩⟩
def transferEvent : Nat := 73517
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73515 .coefficient, .predecessor 1 73516 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73515 .coefficient)
      LeftBound73513.bound (LeftBound73513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73516 .coefficient)
      LeftBound14980.bound (LeftBound14980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73513.bound, LeftBound14980.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73513.bound, LeftBound14980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73513.actual selector witness, LeftBound14980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73517

namespace LeftBound73518
def owner : Owner := ⟨.program ⟨214⟩, ⟨10477⟩⟩
def transferEvent : Nat := 73518
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩ [⟨.result 14981 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14981 .coefficient)
      LeftBound14980.bound (LeftBound14980.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨86⟩⟩) (rawTerms := some (Proof.Events058.exact14981RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14980.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14980.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73518

namespace LeftBound73523
def owner : Owner := ⟨.program ⟨214⟩, ⟨10478⟩⟩
def transferEvent : Nat := 73523
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73521 .coefficient) (.predecessor 1 73522 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73521 .coefficient)
      LeftBound73517.bound (LeftBound73517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73522 .coefficient)
      LeftAuthority3479.bound (LeftAuthority3479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound73517.bound LeftAuthority3479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73517.bound, LeftAuthority3479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound73517.actual selector witness) * (LeftAuthority3479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73523

namespace LeftBound73524
def owner : Owner := ⟨.program ⟨214⟩, ⟨10478⟩⟩
def transferEvent : Nat := 73524
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩ [⟨.result 3480 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3480 .coefficient)
      LeftAuthority3479.bound (LeftAuthority3479.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9395⟩⟩) (rawTerms := some (Proof.Events013.exact3480RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3479.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3479.bound []
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3479.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73524

namespace LeftBound73525
def owner : Owner := ⟨.program ⟨214⟩, ⟨10478⟩⟩
def transferEvent : Nat := 73525
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 73520 .summary) (.transfer 73524) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73520 .summary)
      LeftBound73518.bound (LeftBound73518.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10477⟩⟩) (rawTerms := some (Proof.Events287.exact73520RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73518.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 73524)
      LeftBound73524.bound (LeftBound73524.actual selector witness) := by
  exact .transfer (LeftBound73524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound73518.bound LeftBound73524.bound
def bound : CoeffClass := .finite ⟨1664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73518.bound, LeftBound73524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound73518.actual selector witness) * (LeftBound73524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73525

namespace LeftBound73531
def owner : Owner := ⟨.program ⟨214⟩, ⟨9396⟩⟩
def transferEvent : Nat := 73531
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 73529 .coefficient) (.predecessor 1 73530 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73529 .coefficient)
      LeftAuthority3479.bound (LeftAuthority3479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73530 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3479.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3479.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3479.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73531

namespace LeftBound73536
def owner : Owner := ⟨.program ⟨214⟩, ⟨7189⟩⟩
def transferEvent : Nat := 73536
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73534 .coefficient) (.predecessor 1 73535 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73534 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73535 .coefficient)
      LeftBound15029.bound (LeftBound15029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15029.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound15029.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound15029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound15029.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73536

namespace LeftBound73541
def owner : Owner := ⟨.program ⟨214⟩, ⟨9397⟩⟩
def transferEvent : Nat := 73541
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73539 .coefficient, .predecessor 1 73540 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73539 .coefficient)
      LeftBound73536.bound (LeftBound73536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73540 .coefficient)
      LeftBound73531.bound (LeftBound73531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73536.bound, LeftBound73531.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73536.bound, LeftBound73531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73536.actual selector witness, LeftBound73531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73541

namespace LeftBound73545
def owner : Owner := ⟨.program ⟨214⟩, ⟨9398⟩⟩
def transferEvent : Nat := 73545
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73543 .coefficient, .predecessor 1 73544 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73543 .coefficient)
      LeftBound73541.bound (LeftBound73541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73544 .coefficient)
      LeftBound15021.bound (LeftBound15021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73541.bound, LeftBound15021.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73541.bound, LeftBound15021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73541.actual selector witness, LeftBound15021.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73545

namespace LeftBound73546
def owner : Owner := ⟨.program ⟨214⟩, ⟨9398⟩⟩
def transferEvent : Nat := 73546
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨85⟩⟩]⟩ [⟨.result 15022 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15022 .coefficient)
      LeftBound15021.bound (LeftBound15021.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨85⟩⟩) (rawTerms := some (Proof.Events058.exact15022RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound15021.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound15021.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73546

namespace LeftBound73551
def owner : Owner := ⟨.program ⟨214⟩, ⟨9399⟩⟩
def transferEvent : Nat := 73551
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73549 .coefficient) (.predecessor 1 73550 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73549 .coefficient)
      LeftBound73545.bound (LeftBound73545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73550 .coefficient)
      LeftBound15018.bound (LeftBound15018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15018.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73545.bound LeftBound15018.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73545.bound, LeftBound15018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73545.actual selector witness) * (LeftBound15018.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73551

namespace LeftBound73552
def owner : Owner := ⟨.program ⟨214⟩, ⟨9399⟩⟩
def transferEvent : Nat := 73552
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩ [⟨.result 15015 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15015 .coefficient)
      LeftAuthority15014.bound (LeftAuthority15014.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7831⟩⟩) (rawTerms := some (Proof.Events058.exact15015RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15014.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15014.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15014.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73552

namespace LeftBound73553
def owner : Owner := ⟨.program ⟨214⟩, ⟨9399⟩⟩
def transferEvent : Nat := 73553
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 73548 .summary) (.transfer 73552) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73548 .summary)
      LeftBound73546.bound (LeftBound73546.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9398⟩⟩) (rawTerms := some (Proof.Events287.exact73548RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 73552)
      LeftBound73552.bound (LeftBound73552.actual selector witness) := by
  exact .transfer (LeftBound73552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73546.bound LeftBound73552.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73546.bound, LeftBound73552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73546.actual selector witness) * (LeftBound73552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73553

namespace LeftBound73561
def owner : Owner := ⟨.program ⟨214⟩, ⟨10479⟩⟩
def transferEvent : Nat := 73561
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73559 .coefficient, .predecessor 1 73560 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73559 .coefficient)
      LeftBound73551.bound (LeftBound73551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73560 .coefficient)
      LeftBound73523.bound (LeftBound73523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73551.bound, LeftBound73523.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73551.bound, LeftBound73523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73551.actual selector witness, LeftBound73523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73561

namespace LeftBound73563
def owner : Owner := ⟨.program ⟨214⟩, ⟨10479⟩⟩
def transferEvent : Nat := 73563
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73558 .summary, .result 73528 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73558 .summary)
      LeftBound73553.bound (LeftBound73553.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9399⟩⟩) (rawTerms := some (Proof.Events287.exact73558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73553.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73528 .summary)
      LeftBound73525.bound (LeftBound73525.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10478⟩⟩) (rawTerms := some (Proof.Events287.exact73528RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73525.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73553.bound, LeftBound73525.bound]
def bound : CoeffClass := .finite ⟨95422080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73553.bound, LeftBound73525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73553.actual selector witness, LeftBound73525.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73563

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
