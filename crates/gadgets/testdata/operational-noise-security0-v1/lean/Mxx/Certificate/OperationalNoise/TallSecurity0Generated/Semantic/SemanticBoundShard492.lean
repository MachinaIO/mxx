import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard491

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72115
def owner : Owner := ⟨.program ⟨214⟩, ⟨12162⟩⟩
def transferEvent : Nat := 72115
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72113 .coefficient, .predecessor 1 72114 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72113 .coefficient)
      LeftBound72105.bound (LeftBound72105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72114 .coefficient)
      LeftBound72077.bound (LeftBound72077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72105.bound, LeftBound72077.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72105.bound, LeftBound72077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72105.actual selector witness, LeftBound72077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72115

namespace LeftBound72117
def owner : Owner := ⟨.program ⟨214⟩, ⟨12162⟩⟩
def transferEvent : Nat := 72117
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 72112 .summary, .result 72082 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72112 .summary)
      LeftBound72107.bound (LeftBound72107.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12161⟩⟩) (rawTerms := some (Proof.Events281.exact72112RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72082 .summary)
      LeftBound72079.bound (LeftBound72079.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12157⟩⟩) (rawTerms := some (Proof.Events281.exact72082RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72107.bound, LeftBound72079.bound]
def bound : CoeffClass := .finite ⟨95425408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72107.bound, LeftBound72079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72107.actual selector witness, LeftBound72079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72117

namespace LeftBound72121
def owner : Owner := ⟨.program ⟨214⟩, ⟨25292⟩⟩
def transferEvent : Nat := 72121
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72119 .coefficient) (.predecessor 1 72120 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72119 .coefficient)
      LeftBound72115.bound (LeftBound72115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72115.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72120 .coefficient)
      LeftAuthority72053.bound (LeftAuthority72053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72053.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72053.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72115.bound LeftAuthority72053.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72115.bound, LeftAuthority72053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72115.actual selector witness) * (LeftAuthority72053.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72121

namespace LeftBound72122
def owner : Owner := ⟨.program ⟨214⟩, ⟨25292⟩⟩
def transferEvent : Nat := 72122
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩ [⟨.result 72054 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72054 .coefficient)
      LeftAuthority72053.bound (LeftAuthority72053.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25291⟩⟩) (rawTerms := some (Proof.Events281.exact72054RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72053.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72053.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72053.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72053.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72122

namespace LeftBound72123
def owner : Owner := ⟨.program ⟨214⟩, ⟨25292⟩⟩
def transferEvent : Nat := 72123
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72118 .summary) (.transfer 72122) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72118 .summary)
      LeftBound72117.bound (LeftBound72117.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12162⟩⟩) (rawTerms := some (Proof.Events281.exact72118RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72122)
      LeftBound72122.bound (LeftBound72122.actual selector witness) := by
  exact .transfer (LeftBound72122.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72117.bound LeftBound72122.bound
def bound : CoeffClass := .finite ⟨350212774166528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72117.bound, LeftBound72122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72117.actual selector witness) * (LeftBound72122.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72123

namespace LeftBound72134
def owner : Owner := ⟨.program ⟨214⟩, ⟨19238⟩⟩
def transferEvent : Nat := 72134
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 72132 .coefficient) (.value (.predecessor 1 72133 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72132 .coefficient)
      LeftAuthority72130.bound (LeftAuthority72130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72133 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority72130.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72130.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72130.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72134

namespace LeftBound72138
def owner : Owner := ⟨.program ⟨214⟩, ⟨19239⟩⟩
def transferEvent : Nat := 72138
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72136 .coefficient) (.predecessor 1 72137 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72136 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72137 .coefficient)
      LeftBound72134.bound (LeftBound72134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72134.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound72134.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound72134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound72134.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72138

namespace LeftBound72139
def owner : Owner := ⟨.program ⟨214⟩, ⟨19239⟩⟩
def transferEvent : Nat := 72139
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19236⟩⟩]⟩ [⟨.result 72131 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72131 .coefficient)
      LeftAuthority72130.bound (LeftAuthority72130.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19236⟩⟩) (rawTerms := some (Proof.Events281.exact72131RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72130.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72130.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72130.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72139

namespace LeftBound72140
def owner : Owner := ⟨.program ⟨214⟩, ⟨19239⟩⟩
def transferEvent : Nat := 72140
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 72139) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72139)
      LeftBound72139.bound (LeftBound72139.actual selector witness) := by
  exact .transfer (LeftBound72139.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound72139.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound72139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound72139.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72140

namespace LeftBound72219
def owner : Owner := ⟨.program ⟨214⟩, ⟨12155⟩⟩
def transferEvent : Nat := 72219
def frameStart : Nat := 72190
def rule : BoundRule := .product (.predecessor 0 72217 .coefficient) (.predecessor 1 72218 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72217 .coefficient)
      LeftAuthority72215.bound (LeftAuthority72215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72215.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72218 .coefficient)
      LeftAuthority72212.bound (LeftAuthority72212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72212.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72212.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority72215.bound LeftAuthority72212.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72215.bound, LeftAuthority72212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority72215.actual selector witness) * (LeftAuthority72212.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72219

namespace LeftBound72223
def owner : Owner := ⟨.program ⟨214⟩, ⟨12156⟩⟩
def transferEvent : Nat := 72223
def frameStart : Nat := 72190
def rule : BoundRule := .identity (.predecessor 0 72222 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72222 .coefficient)
      LeftBound72219.bound (LeftBound72219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72219.derived selector witness)

def rawBound : CoeffClass := LeftBound72219.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound72219.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72223

namespace LeftBound72240
def owner : Owner := ⟨.program ⟨214⟩, ⟨12266⟩⟩
def transferEvent : Nat := 72240
def frameStart : Nat := 72190
def rule : BoundRule := .sum [.predecessor 0 72238 .coefficient, .predecessor 1 72239 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72238 .coefficient)
      LeftBound72223.bound (LeftBound72223.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72239 .coefficient)
      LeftAuthority72236.bound (LeftAuthority72236.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority72236.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72223.bound, LeftAuthority72236.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72223.bound, LeftAuthority72236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72223.actual selector witness, LeftAuthority72236.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72240

namespace LeftBound72243
def owner : Owner := ⟨.program ⟨214⟩, ⟨12267⟩⟩
def transferEvent : Nat := 72243
def frameStart : Nat := 72190
def rule : BoundRule := .identity (.predecessor 0 72242 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72242 .coefficient)
      LeftBound72240.bound (LeftBound72240.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72240.derived selector witness)

def rawBound : CoeffClass := LeftBound72240.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound72240.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72243

namespace LeftBound72249
def owner : Owner := ⟨.program ⟨214⟩, ⟨12268⟩⟩
def transferEvent : Nat := 72249
def frameStart : Nat := 72190
def rule : BoundRule := .product (.predecessor 0 72247 .coefficient) (.predecessor 1 72248 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72247 .coefficient)
      LeftAuthority72245.bound (LeftAuthority72245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72245.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72248 .coefficient)
      LeftBound72243.bound (LeftBound72243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72243.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority72245.bound LeftBound72243.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72245.bound, LeftBound72243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority72245.actual selector witness) * (LeftBound72243.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72249

namespace LeftBound72265
def owner : Owner := ⟨.program ⟨214⟩, ⟨7841⟩⟩
def transferEvent : Nat := 72265
def frameStart : Nat := 72190
def rule : BoundRule := .scale (.predecessor 0 72263 .coefficient) (.value (.predecessor 1 72264 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72263 .coefficient)
      LeftAuthority72261.bound (LeftAuthority72261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72261.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72264 .coefficient)
      LeftAuthority72252.bound (LeftAuthority72252.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority72252.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority72261.bound LeftAuthority72252.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72261.bound, LeftAuthority72252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72261.actual selector witness) * (LeftAuthority72252.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72265

namespace LeftBound72268
def owner : Owner := ⟨.program ⟨214⟩, ⟨6792⟩⟩
def transferEvent : Nat := 72268
def frameStart : Nat := 72190
def rule : BoundRule := .identity (.predecessor 0 72267 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72267 .coefficient)
      LeftAuthority72255.bound (LeftAuthority72255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72255.derived selector witness)

def rawBound : CoeffClass := LeftAuthority72255.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority72255.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72268

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
