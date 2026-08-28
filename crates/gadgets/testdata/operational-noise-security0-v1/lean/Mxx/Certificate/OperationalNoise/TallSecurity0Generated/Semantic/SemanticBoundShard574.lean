import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard573

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84202
def owner : Owner := ⟨.program ⟨214⟩, ⟨18351⟩⟩
def transferEvent : Nat := 84202
def frameStart : Nat := 84114
def rule : BoundRule := .product (.predecessor 0 84200 .coefficient) (.predecessor 1 84201 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84200 .coefficient)
      LeftAuthority84175.bound (LeftAuthority84175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84201 .coefficient)
      LeftAuthority84198.bound (LeftAuthority84198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84198.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority84175.bound LeftAuthority84198.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84175.bound, LeftAuthority84198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority84175.actual selector witness) * (LeftAuthority84198.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84202

namespace LeftBound84210
def owner : Owner := ⟨.program ⟨214⟩, ⟨18352⟩⟩
def transferEvent : Nat := 84210
def frameStart : Nat := 84114
def rule : BoundRule := .sum [.predecessor 0 84208 .coefficient, .predecessor 1 84209 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84208 .coefficient)
      LeftAuthority84206.bound (LeftAuthority84206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84206.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84209 .coefficient)
      LeftBound84202.bound (LeftBound84202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84206.bound, LeftBound84202.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84206.bound, LeftBound84202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority84206.actual selector witness, LeftBound84202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84210

namespace LeftBound84214
def owner : Owner := ⟨.program ⟨214⟩, ⟨28305⟩⟩
def transferEvent : Nat := 84214
def frameStart : Nat := 84114
def rule : BoundRule := .sum [.predecessor 0 84212 .coefficient, .predecessor 1 84213 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84212 .coefficient)
      LeftBound84210.bound (LeftBound84210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84213 .coefficient)
      LeftBound84191.bound (LeftBound84191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84210.bound, LeftBound84191.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84210.bound, LeftBound84191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84210.actual selector witness, LeftBound84191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84214

namespace LeftBound84227
def owner : Owner := ⟨.program ⟨214⟩, ⟨28303⟩⟩
def transferEvent : Nat := 84227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84225 .coefficient, .predecessor 1 84226 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84225 .coefficient)
      LeftBound84056.bound (LeftBound84056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84226 .coefficient)
      LeftBound84039.bound (LeftBound84039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84039.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84056.bound, LeftBound84039.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84056.bound, LeftBound84039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84056.actual selector witness, LeftBound84039.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84227

namespace LeftBound84230
def owner : Owner := ⟨.program ⟨214⟩, ⟨28303⟩⟩
def transferEvent : Nat := 84230
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84224 .summary, .result 84046 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84224 .summary)
      LeftBound84058.bound (LeftBound84058.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21691⟩⟩) (rawTerms := some (Proof.Events329.exact84224RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84046 .summary)
      LeftBound84041.bound (LeftBound84041.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28302⟩⟩) (rawTerms := some (Proof.Events328.exact84046RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84058.bound, LeftBound84041.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84058.bound, LeftBound84041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84058.actual selector witness, LeftBound84041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84230

namespace LeftBound84254
def owner : Owner := ⟨.program ⟨214⟩, ⟨11554⟩⟩
def transferEvent : Nat := 84254
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 84252 .coefficient) (.predecessor 1 84253 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84252 .coefficient)
      LeftAuthority4034.bound (LeftAuthority4034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4034.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84253 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4034.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4034.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4034.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84254

namespace LeftBound84259
def owner : Owner := ⟨.program ⟨214⟩, ⟨7236⟩⟩
def transferEvent : Nat := 84259
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84257 .coefficient) (.predecessor 1 84258 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84257 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84258 .coefficient)
      LeftBound10980.bound (LeftBound10980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound10980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound10980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound10980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84259

namespace LeftBound84264
def owner : Owner := ⟨.program ⟨214⟩, ⟨11555⟩⟩
def transferEvent : Nat := 84264
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84262 .coefficient, .predecessor 1 84263 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84262 .coefficient)
      LeftBound84259.bound (LeftBound84259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84263 .coefficient)
      LeftBound84254.bound (LeftBound84254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84254.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84254.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84259.bound, LeftBound84254.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84259.bound, LeftBound84254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84259.actual selector witness, LeftBound84254.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84264

namespace LeftBound84268
def owner : Owner := ⟨.program ⟨214⟩, ⟨11556⟩⟩
def transferEvent : Nat := 84268
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84266 .coefficient, .predecessor 1 84267 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84266 .coefficient)
      LeftBound84264.bound (LeftBound84264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84267 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84264.bound, LeftBound10972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84264.bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84264.actual selector witness, LeftBound10972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84268

namespace LeftBound84269
def owner : Owner := ⟨.program ⟨214⟩, ⟨11556⟩⟩
def transferEvent : Nat := 84269
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩ [⟨.result 10973 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10973 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨94⟩⟩) (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10972.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10972.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84269

namespace LeftBound84274
def owner : Owner := ⟨.program ⟨214⟩, ⟨14427⟩⟩
def transferEvent : Nat := 84274
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84272 .coefficient) (.predecessor 1 84273 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84272 .coefficient)
      LeftBound84268.bound (LeftBound84268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84273 .coefficient)
      LeftAuthority4037.bound (LeftAuthority4037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4037.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4037.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound84268.bound LeftAuthority4037.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84268.bound, LeftAuthority4037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound84268.actual selector witness) * (LeftAuthority4037.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84274

namespace LeftBound84275
def owner : Owner := ⟨.program ⟨214⟩, ⟨14427⟩⟩
def transferEvent : Nat := 84275
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩ [⟨.result 4038 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4038 .coefficient)
      LeftAuthority4037.bound (LeftAuthority4037.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14424⟩⟩) (rawTerms := some (Proof.Events015.exact4038RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4037.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4037.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4037.bound []
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4037.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84275

namespace LeftBound84276
def owner : Owner := ⟨.program ⟨214⟩, ⟨14427⟩⟩
def transferEvent : Nat := 84276
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84271 .summary) (.transfer 84275) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84271 .summary)
      LeftBound84269.bound (LeftBound84269.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11556⟩⟩) (rawTerms := some (Proof.Events329.exact84271RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84275)
      LeftBound84275.bound (LeftBound84275.actual selector witness) := by
  exact .transfer (LeftBound84275.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound84269.bound LeftBound84275.bound
def bound : CoeffClass := .finite ⟨18304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84269.bound, LeftBound84275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound84269.actual selector witness) * (LeftBound84275.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84276

namespace LeftBound84282
def owner : Owner := ⟨.program ⟨214⟩, ⟨14428⟩⟩
def transferEvent : Nat := 84282
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 84280 .coefficient) (.predecessor 1 84281 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84280 .coefficient)
      LeftAuthority4037.bound (LeftAuthority4037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4037.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84281 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4037.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4037.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4037.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84282

namespace LeftBound84287
def owner : Owner := ⟨.program ⟨214⟩, ⟨7217⟩⟩
def transferEvent : Nat := 84287
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84285 .coefficient) (.predecessor 1 84286 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84285 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84286 .coefficient)
      LeftBound11021.bound (LeftBound11021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound11021.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound11021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound11021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84287

namespace LeftBound84292
def owner : Owner := ⟨.program ⟨214⟩, ⟨14429⟩⟩
def transferEvent : Nat := 84292
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84290 .coefficient, .predecessor 1 84291 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84290 .coefficient)
      LeftBound84287.bound (LeftBound84287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84291 .coefficient)
      LeftBound84282.bound (LeftBound84282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84287.bound, LeftBound84282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84287.bound, LeftBound84282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84287.actual selector witness, LeftBound84282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84292

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
