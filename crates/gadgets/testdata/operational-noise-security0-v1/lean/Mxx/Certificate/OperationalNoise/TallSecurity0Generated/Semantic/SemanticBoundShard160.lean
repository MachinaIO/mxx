import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard158
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard159

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24573
def owner : Owner := ⟨.program ⟨214⟩, ⟨25238⟩⟩
def transferEvent : Nat := 24573
def frameStart : Nat := 24459
def rule : BoundRule := .sum [.predecessor 0 24571 .coefficient, .predecessor 1 24572 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24571 .coefficient)
      LeftBound24569.bound (LeftBound24569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24572 .coefficient)
      LeftBound24550.bound (LeftBound24550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24569.bound, LeftBound24550.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24569.bound, LeftBound24550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24569.actual selector witness, LeftBound24550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24573

namespace LeftBound24586
def owner : Owner := ⟨.program ⟨214⟩, ⟨25236⟩⟩
def transferEvent : Nat := 24586
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24584 .coefficient, .predecessor 1 24585 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24584 .coefficient)
      LeftBound24407.bound (LeftBound24407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24585 .coefficient)
      LeftBound24390.bound (LeftBound24390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24390.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24407.bound, LeftBound24390.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24407.bound, LeftBound24390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24407.actual selector witness, LeftBound24390.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24586

namespace LeftBound24589
def owner : Owner := ⟨.program ⟨214⟩, ⟨25236⟩⟩
def transferEvent : Nat := 24589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 24583 .summary, .result 24397 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24583 .summary)
      LeftBound24409.bound (LeftBound24409.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19831⟩⟩) (rawTerms := some (Proof.Events096.exact24583RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24397 .summary)
      LeftBound24392.bound (LeftBound24392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25235⟩⟩) (rawTerms := some (Proof.Events095.exact24397RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24392.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24409.bound, LeftBound24392.bound]
def bound : CoeffClass := .finite ⟨352115681275904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24409.bound, LeftBound24392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24409.actual selector witness, LeftBound24392.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24589

namespace LeftBound24593
def owner : Owner := ⟨.program ⟨214⟩, ⟨28775⟩⟩
def transferEvent : Nat := 24593
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24591 .coefficient) (.predecessor 1 24592 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24591 .coefficient)
      LeftBound24586.bound (LeftBound24586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24592 .coefficient)
      LeftAuthority24312.bound (LeftAuthority24312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24312.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24586.bound LeftAuthority24312.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24586.bound, LeftAuthority24312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24586.actual selector witness) * (LeftAuthority24312.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24593

namespace LeftBound24594
def owner : Owner := ⟨.program ⟨214⟩, ⟨28775⟩⟩
def transferEvent : Nat := 24594
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩ [⟨.result 24313 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24313 .coefficient)
      LeftAuthority24312.bound (LeftAuthority24312.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28773⟩⟩) (rawTerms := some (Proof.Events094.exact24313RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24312.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24312.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24312.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24594

namespace LeftBound24595
def owner : Owner := ⟨.program ⟨214⟩, ⟨28775⟩⟩
def transferEvent : Nat := 24595
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 24590 .summary) (.transfer 24594) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24590 .summary)
      LeftBound24589.bound (LeftBound24589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25236⟩⟩) (rawTerms := some (Proof.Events096.exact24590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24594)
      LeftBound24594.bound (LeftBound24594.actual selector witness) := by
  exact .transfer (LeftBound24594.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24589.bound LeftBound24594.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24589.bound, LeftBound24594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24589.actual selector witness) * (LeftBound24594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24595

namespace LeftBound24606
def owner : Owner := ⟨.program ⟨214⟩, ⟨21990⟩⟩
def transferEvent : Nat := 24606
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 24604 .coefficient) (.value (.predecessor 1 24605 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24604 .coefficient)
      LeftAuthority24602.bound (LeftAuthority24602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24605 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority24602.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24602.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24602.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24606

namespace LeftBound24610
def owner : Owner := ⟨.program ⟨214⟩, ⟨21991⟩⟩
def transferEvent : Nat := 24610
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24608 .coefficient) (.predecessor 1 24609 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24608 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24609 .coefficient)
      LeftBound24606.bound (LeftBound24606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24606.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound24606.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound24606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound24606.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24610

namespace LeftBound24611
def owner : Owner := ⟨.program ⟨214⟩, ⟨21991⟩⟩
def transferEvent : Nat := 24611
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩ [⟨.result 24603 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24603 .coefficient)
      LeftAuthority24602.bound (LeftAuthority24602.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21988⟩⟩) (rawTerms := some (Proof.Events096.exact24603RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24602.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24602.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24602.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24611

namespace LeftBound24612
def owner : Owner := ⟨.program ⟨214⟩, ⟨21991⟩⟩
def transferEvent : Nat := 24612
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 24611) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24611)
      LeftBound24611.bound (LeftBound24611.actual selector witness) := by
  exact .transfer (LeftBound24611.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound24611.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound24611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound24611.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24612

namespace LeftBound24707
def owner : Owner := ⟨.program ⟨214⟩, ⟨16394⟩⟩
def transferEvent : Nat := 24707
def frameStart : Nat := 24668
def rule : BoundRule := .identity (.predecessor 0 24706 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24706 .coefficient)
      LeftAuthority24704.bound (LeftAuthority24704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24704.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24704.derived selector witness)

def rawBound : CoeffClass := LeftAuthority24704.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority24704.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24707

namespace LeftBound24724
def owner : Owner := ⟨.program ⟨214⟩, ⟨16433⟩⟩
def transferEvent : Nat := 24724
def frameStart : Nat := 24668
def rule : BoundRule := .sum [.predecessor 0 24722 .coefficient, .predecessor 1 24723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24722 .coefficient)
      LeftBound24707.bound (LeftBound24707.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24723 .coefficient)
      LeftAuthority24720.bound (LeftAuthority24720.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24707.bound, LeftAuthority24720.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24707.bound, LeftAuthority24720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24707.actual selector witness, LeftAuthority24720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24724

namespace LeftBound24727
def owner : Owner := ⟨.program ⟨214⟩, ⟨16434⟩⟩
def transferEvent : Nat := 24727
def frameStart : Nat := 24668
def rule : BoundRule := .identity (.predecessor 0 24726 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24726 .coefficient)
      LeftBound24724.bound (LeftBound24724.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24724.derived selector witness)

def rawBound : CoeffClass := LeftBound24724.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound24724.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24727

namespace LeftBound24733
def owner : Owner := ⟨.program ⟨214⟩, ⟨16435⟩⟩
def transferEvent : Nat := 24733
def frameStart : Nat := 24668
def rule : BoundRule := .product (.predecessor 0 24731 .coefficient) (.predecessor 1 24732 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24731 .coefficient)
      LeftAuthority24729.bound (LeftAuthority24729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24732 .coefficient)
      LeftBound24727.bound (LeftBound24727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24727.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority24729.bound LeftBound24727.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24729.bound, LeftBound24727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority24729.actual selector witness) * (LeftBound24727.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24733

namespace LeftBound24741
def owner : Owner := ⟨.program ⟨214⟩, ⟨16436⟩⟩
def transferEvent : Nat := 24741
def frameStart : Nat := 24668
def rule : BoundRule := .sum [.predecessor 0 24739 .coefficient, .predecessor 1 24740 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24739 .coefficient)
      LeftAuthority24737.bound (LeftAuthority24737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24737.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24737.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24740 .coefficient)
      LeftBound24733.bound (LeftBound24733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24733.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority24737.bound, LeftBound24733.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24737.bound, LeftBound24733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority24737.actual selector witness, LeftBound24733.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24741

namespace LeftBound24745
def owner : Owner := ⟨.program ⟨214⟩, ⟨28774⟩⟩
def transferEvent : Nat := 24745
def frameStart : Nat := 24668
def rule : BoundRule := .product (.predecessor 0 24743 .coefficient) (.predecessor 1 24744 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24743 .coefficient)
      LeftBound24741.bound (LeftBound24741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24744 .coefficient)
      LeftAuthority24718.bound (LeftAuthority24718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24741.bound LeftAuthority24718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24741.bound, LeftAuthority24718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24741.actual selector witness) * (LeftAuthority24718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24745

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
