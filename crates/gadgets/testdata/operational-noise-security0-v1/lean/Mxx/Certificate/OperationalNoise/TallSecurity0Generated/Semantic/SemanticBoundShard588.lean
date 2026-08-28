import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard080
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard587

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound85977
def owner : Owner := ⟨.program ⟨214⟩, ⟨21115⟩⟩
def transferEvent : Nat := 85977
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩ [⟨.result 85969 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85969 .coefficient)
      LeftAuthority85968.bound (LeftAuthority85968.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21112⟩⟩) (rawTerms := some (Proof.Events335.exact85969RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85968.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority85968.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85968.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85977

namespace LeftBound85978
def owner : Owner := ⟨.program ⟨214⟩, ⟨21115⟩⟩
def transferEvent : Nat := 85978
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 85977) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 85977)
      LeftBound85977.bound (LeftBound85977.actual selector witness) := by
  exact .transfer (LeftBound85977.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound85977.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound85977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound85977.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85978

namespace LeftBound86073
def owner : Owner := ⟨.program ⟨214⟩, ⟨15703⟩⟩
def transferEvent : Nat := 86073
def frameStart : Nat := 86034
def rule : BoundRule := .identity (.predecessor 0 86072 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86072 .coefficient)
      LeftAuthority86070.bound (LeftAuthority86070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86070.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86070.derived selector witness)

def rawBound : CoeffClass := LeftAuthority86070.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority86070.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86073

namespace LeftBound86090
def owner : Owner := ⟨.program ⟨214⟩, ⟨15777⟩⟩
def transferEvent : Nat := 86090
def frameStart : Nat := 86034
def rule : BoundRule := .sum [.predecessor 0 86088 .coefficient, .predecessor 1 86089 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86088 .coefficient)
      LeftBound86073.bound (LeftBound86073.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound86073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86089 .coefficient)
      LeftAuthority86086.bound (LeftAuthority86086.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority86086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86073.bound, LeftAuthority86086.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86073.bound, LeftAuthority86086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86073.actual selector witness, LeftAuthority86086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86090

namespace LeftBound86093
def owner : Owner := ⟨.program ⟨214⟩, ⟨15778⟩⟩
def transferEvent : Nat := 86093
def frameStart : Nat := 86034
def rule : BoundRule := .identity (.predecessor 0 86092 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86092 .coefficient)
      LeftBound86090.bound (LeftBound86090.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound86090.derived selector witness)

def rawBound : CoeffClass := LeftBound86090.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound86090.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86093

namespace LeftBound86099
def owner : Owner := ⟨.program ⟨214⟩, ⟨15779⟩⟩
def transferEvent : Nat := 86099
def frameStart : Nat := 86034
def rule : BoundRule := .product (.predecessor 0 86097 .coefficient) (.predecessor 1 86098 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86097 .coefficient)
      LeftAuthority86095.bound (LeftAuthority86095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86098 .coefficient)
      LeftBound86093.bound (LeftBound86093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority86095.bound LeftBound86093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86095.bound, LeftBound86093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority86095.actual selector witness) * (LeftBound86093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86099

namespace LeftBound86107
def owner : Owner := ⟨.program ⟨214⟩, ⟨15780⟩⟩
def transferEvent : Nat := 86107
def frameStart : Nat := 86034
def rule : BoundRule := .sum [.predecessor 0 86105 .coefficient, .predecessor 1 86106 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86105 .coefficient)
      LeftAuthority86103.bound (LeftAuthority86103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86106 .coefficient)
      LeftBound86099.bound (LeftBound86099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86099.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority86103.bound, LeftBound86099.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86103.bound, LeftBound86099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority86103.actual selector witness, LeftBound86099.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86107

namespace LeftBound86111
def owner : Owner := ⟨.program ⟨214⟩, ⟨27433⟩⟩
def transferEvent : Nat := 86111
def frameStart : Nat := 86034
def rule : BoundRule := .product (.predecessor 0 86109 .coefficient) (.predecessor 1 86110 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86109 .coefficient)
      LeftBound86107.bound (LeftBound86107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86110 .coefficient)
      LeftAuthority86084.bound (LeftAuthority86084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86084.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86084.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86107.bound LeftAuthority86084.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86107.bound, LeftAuthority86084.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86107.actual selector witness) * (LeftAuthority86084.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86111

namespace LeftBound86122
def owner : Owner := ⟨.program ⟨214⟩, ⟨15749⟩⟩
def transferEvent : Nat := 86122
def frameStart : Nat := 86034
def rule : BoundRule := .product (.predecessor 0 86120 .coefficient) (.predecessor 1 86121 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86120 .coefficient)
      LeftAuthority86095.bound (LeftAuthority86095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86121 .coefficient)
      LeftAuthority86118.bound (LeftAuthority86118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86118.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority86095.bound LeftAuthority86118.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86095.bound, LeftAuthority86118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority86095.actual selector witness) * (LeftAuthority86118.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86122

namespace LeftBound86130
def owner : Owner := ⟨.program ⟨214⟩, ⟨15750⟩⟩
def transferEvent : Nat := 86130
def frameStart : Nat := 86034
def rule : BoundRule := .sum [.predecessor 0 86128 .coefficient, .predecessor 1 86129 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86128 .coefficient)
      LeftAuthority86126.bound (LeftAuthority86126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86126.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86129 .coefficient)
      LeftBound86122.bound (LeftBound86122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86122.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority86126.bound, LeftBound86122.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86126.bound, LeftBound86122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority86126.actual selector witness, LeftBound86122.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86130

namespace LeftBound86134
def owner : Owner := ⟨.program ⟨214⟩, ⟨27437⟩⟩
def transferEvent : Nat := 86134
def frameStart : Nat := 86034
def rule : BoundRule := .sum [.predecessor 0 86132 .coefficient, .predecessor 1 86133 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86132 .coefficient)
      LeftBound86130.bound (LeftBound86130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86133 .coefficient)
      LeftBound86111.bound (LeftBound86111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86111.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86111.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86130.bound, LeftBound86111.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86130.bound, LeftBound86111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86130.actual selector witness, LeftBound86111.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86134

namespace LeftBound86147
def owner : Owner := ⟨.program ⟨214⟩, ⟨27435⟩⟩
def transferEvent : Nat := 86147
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86145 .coefficient, .predecessor 1 86146 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86145 .coefficient)
      LeftBound85976.bound (LeftBound85976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86146 .coefficient)
      LeftBound85959.bound (LeftBound85959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85959.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85976.bound, LeftBound85959.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85976.bound, LeftBound85959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85976.actual selector witness, LeftBound85959.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86147

namespace LeftBound86150
def owner : Owner := ⟨.program ⟨214⟩, ⟨27435⟩⟩
def transferEvent : Nat := 86150
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 86144 .summary, .result 85966 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86144 .summary)
      LeftBound85978.bound (LeftBound85978.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21115⟩⟩) (rawTerms := some (Proof.Events336.exact86144RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85966 .summary)
      LeftBound85961.bound (LeftBound85961.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27434⟩⟩) (rawTerms := some (Proof.Events335.exact85966RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85978.bound, LeftBound85961.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85978.bound, LeftBound85961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85978.actual selector witness, LeftBound85961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86150

namespace LeftBound86174
def owner : Owner := ⟨.program ⟨214⟩, ⟨11218⟩⟩
def transferEvent : Nat := 86174
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 86172 .coefficient) (.predecessor 1 86173 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86172 .coefficient)
      LeftAuthority4126.bound (LeftAuthority4126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4126.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86173 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4126.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4126.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4126.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86174

namespace LeftBound86179
def owner : Owner := ⟨.program ⟨214⟩, ⟨7232⟩⟩
def transferEvent : Nat := 86179
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86177 .coefficient) (.predecessor 1 86178 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86177 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86178 .coefficient)
      LeftBound12984.bound (LeftBound12984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12984.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound12984.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound12984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound12984.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86179

namespace LeftBound86184
def owner : Owner := ⟨.program ⟨214⟩, ⟨11219⟩⟩
def transferEvent : Nat := 86184
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86182 .coefficient, .predecessor 1 86183 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86182 .coefficient)
      LeftBound86179.bound (LeftBound86179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86183 .coefficient)
      LeftBound86174.bound (LeftBound86174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86179.bound, LeftBound86174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86179.bound, LeftBound86174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86179.actual selector witness, LeftBound86174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86184

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
