import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard580

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound85133
def owner : Owner := ⟨.program ⟨214⟩, ⟨16016⟩⟩
def transferEvent : Nat := 85133
def frameStart : Nat := 85074
def rule : BoundRule := .identity (.predecessor 0 85132 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85132 .coefficient)
      LeftBound85130.bound (LeftBound85130.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound85130.derived selector witness)

def rawBound : CoeffClass := LeftBound85130.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound85130.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85133

namespace LeftBound85139
def owner : Owner := ⟨.program ⟨214⟩, ⟨16017⟩⟩
def transferEvent : Nat := 85139
def frameStart : Nat := 85074
def rule : BoundRule := .product (.predecessor 0 85137 .coefficient) (.predecessor 1 85138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85137 .coefficient)
      LeftAuthority85135.bound (LeftAuthority85135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85135.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85138 .coefficient)
      LeftBound85133.bound (LeftBound85133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85133.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority85135.bound LeftBound85133.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85135.bound, LeftBound85133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority85135.actual selector witness) * (LeftBound85133.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85139

namespace LeftBound85147
def owner : Owner := ⟨.program ⟨214⟩, ⟨16018⟩⟩
def transferEvent : Nat := 85147
def frameStart : Nat := 85074
def rule : BoundRule := .sum [.predecessor 0 85145 .coefficient, .predecessor 1 85146 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85145 .coefficient)
      LeftAuthority85143.bound (LeftAuthority85143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85143.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85146 .coefficient)
      LeftBound85139.bound (LeftBound85139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority85143.bound, LeftBound85139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85143.bound, LeftBound85139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority85143.actual selector witness, LeftBound85139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85147

namespace LeftBound85151
def owner : Owner := ⟨.program ⟨214⟩, ⟨27867⟩⟩
def transferEvent : Nat := 85151
def frameStart : Nat := 85074
def rule : BoundRule := .product (.predecessor 0 85149 .coefficient) (.predecessor 1 85150 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85149 .coefficient)
      LeftBound85147.bound (LeftBound85147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85150 .coefficient)
      LeftAuthority85124.bound (LeftAuthority85124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85124.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85147.bound LeftAuthority85124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85147.bound, LeftAuthority85124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85147.actual selector witness) * (LeftAuthority85124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85151

namespace LeftBound85162
def owner : Owner := ⟨.program ⟨214⟩, ⟨15987⟩⟩
def transferEvent : Nat := 85162
def frameStart : Nat := 85074
def rule : BoundRule := .product (.predecessor 0 85160 .coefficient) (.predecessor 1 85161 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85160 .coefficient)
      LeftAuthority85135.bound (LeftAuthority85135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85135.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85161 .coefficient)
      LeftAuthority85158.bound (LeftAuthority85158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85158.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85158.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority85135.bound LeftAuthority85158.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85135.bound, LeftAuthority85158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority85135.actual selector witness) * (LeftAuthority85158.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85162

namespace LeftBound85170
def owner : Owner := ⟨.program ⟨214⟩, ⟨15988⟩⟩
def transferEvent : Nat := 85170
def frameStart : Nat := 85074
def rule : BoundRule := .sum [.predecessor 0 85168 .coefficient, .predecessor 1 85169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85168 .coefficient)
      LeftAuthority85166.bound (LeftAuthority85166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85169 .coefficient)
      LeftBound85162.bound (LeftBound85162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85162.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority85166.bound, LeftBound85162.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85166.bound, LeftBound85162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority85166.actual selector witness, LeftBound85162.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85170

namespace LeftBound85174
def owner : Owner := ⟨.program ⟨214⟩, ⟨27871⟩⟩
def transferEvent : Nat := 85174
def frameStart : Nat := 85074
def rule : BoundRule := .sum [.predecessor 0 85172 .coefficient, .predecessor 1 85173 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85172 .coefficient)
      LeftBound85170.bound (LeftBound85170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85173 .coefficient)
      LeftBound85151.bound (LeftBound85151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85170.bound, LeftBound85151.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85170.bound, LeftBound85151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85170.actual selector witness, LeftBound85151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85174

namespace LeftBound85187
def owner : Owner := ⟨.program ⟨214⟩, ⟨27869⟩⟩
def transferEvent : Nat := 85187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85185 .coefficient, .predecessor 1 85186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85185 .coefficient)
      LeftBound85016.bound (LeftBound85016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85186 .coefficient)
      LeftBound84999.bound (LeftBound84999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85016.bound, LeftBound84999.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85016.bound, LeftBound84999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85016.actual selector witness, LeftBound84999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85187

namespace LeftBound85190
def owner : Owner := ⟨.program ⟨214⟩, ⟨27869⟩⟩
def transferEvent : Nat := 85190
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 85184 .summary, .result 85006 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85184 .summary)
      LeftBound85018.bound (LeftBound85018.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21403⟩⟩) (rawTerms := some (Proof.Events332.exact85184RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85006 .summary)
      LeftBound85001.bound (LeftBound85001.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27868⟩⟩) (rawTerms := some (Proof.Events332.exact85006RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85001.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85018.bound, LeftBound85001.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85018.bound, LeftBound85001.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85018.actual selector witness, LeftBound85001.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85190

namespace LeftBound85214
def owner : Owner := ⟨.program ⟨214⟩, ⟨11386⟩⟩
def transferEvent : Nat := 85214
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 85212 .coefficient) (.predecessor 1 85213 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85212 .coefficient)
      LeftAuthority4080.bound (LeftAuthority4080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4080.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4080.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85213 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4080.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4080.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4080.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound85214

namespace LeftBound85219
def owner : Owner := ⟨.program ⟨214⟩, ⟨7234⟩⟩
def transferEvent : Nat := 85219
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85217 .coefficient) (.predecessor 1 85218 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85217 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85218 .coefficient)
      LeftBound11982.bound (LeftBound11982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11982.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound11982.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound11982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound11982.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85219

namespace LeftBound85224
def owner : Owner := ⟨.program ⟨214⟩, ⟨11387⟩⟩
def transferEvent : Nat := 85224
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85222 .coefficient, .predecessor 1 85223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85222 .coefficient)
      LeftBound85219.bound (LeftBound85219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85223 .coefficient)
      LeftBound85214.bound (LeftBound85214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85214.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85219.bound, LeftBound85214.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85219.bound, LeftBound85214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85219.actual selector witness, LeftBound85214.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85224

namespace LeftBound85228
def owner : Owner := ⟨.program ⟨214⟩, ⟨11388⟩⟩
def transferEvent : Nat := 85228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85226 .coefficient, .predecessor 1 85227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85226 .coefficient)
      LeftBound85224.bound (LeftBound85224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85227 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85224.bound, LeftBound11974.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85224.bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85224.actual selector witness, LeftBound11974.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85228

namespace LeftBound85229
def owner : Owner := ⟨.program ⟨214⟩, ⟨11388⟩⟩
def transferEvent : Nat := 85229
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩ [⟨.result 11975 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11975 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨92⟩⟩) (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11974.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11974.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85229

namespace LeftBound85234
def owner : Owner := ⟨.program ⟨214⟩, ⟨13993⟩⟩
def transferEvent : Nat := 85234
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85232 .coefficient) (.predecessor 1 85233 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85232 .coefficient)
      LeftBound85228.bound (LeftBound85228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85233 .coefficient)
      LeftAuthority4083.bound (LeftAuthority4083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4083.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound85228.bound LeftAuthority4083.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85228.bound, LeftAuthority4083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound85228.actual selector witness) * (LeftAuthority4083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85234

namespace LeftBound85235
def owner : Owner := ⟨.program ⟨214⟩, ⟨13993⟩⟩
def transferEvent : Nat := 85235
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13990⟩⟩], []⟩ [⟨.result 4084 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4084 .coefficient)
      LeftAuthority4083.bound (LeftAuthority4083.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13990⟩⟩) (rawTerms := some (Proof.Events015.exact4084RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4083.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4083.bound []
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4083.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85235

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
