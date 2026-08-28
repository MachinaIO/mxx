import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard574

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84296
def owner : Owner := ⟨.program ⟨214⟩, ⟨14430⟩⟩
def transferEvent : Nat := 84296
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84294 .coefficient, .predecessor 1 84295 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84294 .coefficient)
      LeftBound84292.bound (LeftBound84292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84292.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84292.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84295 .coefficient)
      LeftBound11013.bound (LeftBound11013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84292.bound, LeftBound11013.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84292.bound, LeftBound11013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84292.actual selector witness, LeftBound11013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84296

namespace LeftBound84297
def owner : Owner := ⟨.program ⟨214⟩, ⟨14430⟩⟩
def transferEvent : Nat := 84297
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨75⟩⟩]⟩ [⟨.result 11014 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11014 .coefficient)
      LeftBound11013.bound (LeftBound11013.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨75⟩⟩) (rawTerms := some (Proof.Events043.exact11014RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11013.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11013.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84297

namespace LeftBound84302
def owner : Owner := ⟨.program ⟨214⟩, ⟨14431⟩⟩
def transferEvent : Nat := 84302
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84300 .coefficient) (.predecessor 1 84301 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84300 .coefficient)
      LeftBound84296.bound (LeftBound84296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84296.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84301 .coefficient)
      LeftBound11010.bound (LeftBound11010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11010.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84296.bound LeftBound11010.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84296.bound, LeftBound11010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84296.actual selector witness) * (LeftBound11010.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84302

namespace LeftBound84303
def owner : Owner := ⟨.program ⟨214⟩, ⟨14431⟩⟩
def transferEvent : Nat := 84303
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩ [⟨.result 11007 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11007 .coefficient)
      LeftAuthority11006.bound (LeftAuthority11006.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7855⟩⟩) (rawTerms := some (Proof.Events042.exact11007RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11006.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11006.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11006.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84303

namespace LeftBound84304
def owner : Owner := ⟨.program ⟨214⟩, ⟨14431⟩⟩
def transferEvent : Nat := 84304
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84299 .summary) (.transfer 84303) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84299 .summary)
      LeftBound84297.bound (LeftBound84297.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14430⟩⟩) (rawTerms := some (Proof.Events329.exact84299RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84303)
      LeftBound84303.bound (LeftBound84303.actual selector witness) := by
  exact .transfer (LeftBound84303.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84297.bound LeftBound84303.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84297.bound, LeftBound84303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84297.actual selector witness) * (LeftBound84303.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84304

namespace LeftBound84312
def owner : Owner := ⟨.program ⟨214⟩, ⟨14432⟩⟩
def transferEvent : Nat := 84312
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84310 .coefficient, .predecessor 1 84311 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84310 .coefficient)
      LeftBound84302.bound (LeftBound84302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84311 .coefficient)
      LeftBound84274.bound (LeftBound84274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84302.bound, LeftBound84274.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84302.bound, LeftBound84274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84302.actual selector witness, LeftBound84274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84312

namespace LeftBound84314
def owner : Owner := ⟨.program ⟨214⟩, ⟨14432⟩⟩
def transferEvent : Nat := 84314
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84309 .summary, .result 84279 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84309 .summary)
      LeftBound84304.bound (LeftBound84304.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14431⟩⟩) (rawTerms := some (Proof.Events329.exact84309RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84304.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84279 .summary)
      LeftBound84276.bound (LeftBound84276.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14427⟩⟩) (rawTerms := some (Proof.Events329.exact84279RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84276.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84304.bound, LeftBound84276.bound]
def bound : CoeffClass := .finite ⟨95438720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84304.bound, LeftBound84276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84304.actual selector witness, LeftBound84276.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84314

namespace LeftBound84318
def owner : Owner := ⟨.program ⟨214⟩, ⟨26144⟩⟩
def transferEvent : Nat := 84318
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84316 .coefficient) (.predecessor 1 84317 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84316 .coefficient)
      LeftBound84312.bound (LeftBound84312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84312.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84317 .coefficient)
      LeftAuthority84250.bound (LeftAuthority84250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84250.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84250.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84312.bound LeftAuthority84250.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84312.bound, LeftAuthority84250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84312.actual selector witness) * (LeftAuthority84250.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84318

namespace LeftBound84319
def owner : Owner := ⟨.program ⟨214⟩, ⟨26144⟩⟩
def transferEvent : Nat := 84319
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩ [⟨.result 84251 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84251 .coefficient)
      LeftAuthority84250.bound (LeftAuthority84250.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26143⟩⟩) (rawTerms := some (Proof.Events329.exact84251RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84250.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84250.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84250.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84250.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84319

namespace LeftBound84320
def owner : Owner := ⟨.program ⟨214⟩, ⟨26144⟩⟩
def transferEvent : Nat := 84320
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84315 .summary) (.transfer 84319) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84315 .summary)
      LeftBound84314.bound (LeftBound84314.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14432⟩⟩) (rawTerms := some (Proof.Events329.exact84315RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84319)
      LeftBound84319.bound (LeftBound84319.actual selector witness) := by
  exact .transfer (LeftBound84319.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84314.bound LeftBound84319.bound
def bound : CoeffClass := .finite ⟨350261629419520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84314.bound, LeftBound84319.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84314.actual selector witness) * (LeftBound84319.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84320

namespace LeftBound84331
def owner : Owner := ⟨.program ⟨214⟩, ⟨19602⟩⟩
def transferEvent : Nat := 84331
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 84329 .coefficient) (.value (.predecessor 1 84330 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84329 .coefficient)
      LeftAuthority84327.bound (LeftAuthority84327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84330 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority84327.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84327.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84327.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84331

namespace LeftBound84335
def owner : Owner := ⟨.program ⟨214⟩, ⟨19603⟩⟩
def transferEvent : Nat := 84335
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84333 .coefficient) (.predecessor 1 84334 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84333 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84334 .coefficient)
      LeftBound84331.bound (LeftBound84331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84331.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound84331.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound84331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound84331.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84335

namespace LeftBound84336
def owner : Owner := ⟨.program ⟨214⟩, ⟨19603⟩⟩
def transferEvent : Nat := 84336
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩ [⟨.result 84328 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84328 .coefficient)
      LeftAuthority84327.bound (LeftAuthority84327.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19600⟩⟩) (rawTerms := some (Proof.Events329.exact84328RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84327.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84327.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84327.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84336

namespace LeftBound84337
def owner : Owner := ⟨.program ⟨214⟩, ⟨19603⟩⟩
def transferEvent : Nat := 84337
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 84336) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84336)
      LeftBound84336.bound (LeftBound84336.actual selector witness) := by
  exact .transfer (LeftBound84336.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound84336.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound84336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound84336.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84337

namespace LeftBound84416
def owner : Owner := ⟨.program ⟨214⟩, ⟨14425⟩⟩
def transferEvent : Nat := 84416
def frameStart : Nat := 84387
def rule : BoundRule := .product (.predecessor 0 84414 .coefficient) (.predecessor 1 84415 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84414 .coefficient)
      LeftAuthority84412.bound (LeftAuthority84412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84412.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84415 .coefficient)
      LeftAuthority84409.bound (LeftAuthority84409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84409.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84409.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority84412.bound LeftAuthority84409.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84412.bound, LeftAuthority84409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority84412.actual selector witness) * (LeftAuthority84409.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84416

namespace LeftBound84420
def owner : Owner := ⟨.program ⟨214⟩, ⟨14426⟩⟩
def transferEvent : Nat := 84420
def frameStart : Nat := 84387
def rule : BoundRule := .identity (.predecessor 0 84419 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84419 .coefficient)
      LeftBound84416.bound (LeftBound84416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84416.derived selector witness)

def rawBound : CoeffClass := LeftBound84416.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound84416.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84420

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
