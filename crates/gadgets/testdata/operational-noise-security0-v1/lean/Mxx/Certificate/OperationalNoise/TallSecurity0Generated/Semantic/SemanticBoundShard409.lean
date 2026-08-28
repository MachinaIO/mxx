import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard341
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard406
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard407
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard408

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound60777
def owner : Owner := ⟨.program ⟨214⟩, ⟨18654⟩⟩
def transferEvent : Nat := 60777
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60775 .coefficient, .predecessor 1 60776 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60775 .coefficient)
      LeftBound60773.bound (LeftBound60773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60776 .coefficient)
      LeftBound60633.bound (LeftBound60633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60633.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60773.bound, LeftBound60633.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60773.bound, LeftBound60633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60773.actual selector witness, LeftBound60633.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60777

namespace LeftBound60781
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def transferEvent : Nat := 60781
def frameStart : Nat := 60103
def rule : BoundRule := .product (.predecessor 0 60779 .coefficient) (.predecessor 1 60780 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60779 .coefficient)
      LeftBound60777.bound (LeftBound60777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60780 .coefficient)
      LeftAuthority60618.bound (LeftAuthority60618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound60777.bound LeftAuthority60618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60777.bound, LeftAuthority60618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound60777.actual selector witness) * (LeftAuthority60618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60781

namespace LeftBound60860
def owner : Owner := ⟨.program ⟨214⟩, ⟨18501⟩⟩
def transferEvent : Nat := 60860
def frameStart : Nat := 60103
def rule : BoundRule := .product (.predecessor 0 60858 .coefficient) (.predecessor 1 60859 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60858 .coefficient)
      LeftAuthority60629.bound (LeftAuthority60629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60629.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60859 .coefficient)
      LeftAuthority60856.bound (LeftAuthority60856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60856.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority60629.bound LeftAuthority60856.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority60629.bound, LeftAuthority60856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority60629.actual selector witness) * (LeftAuthority60856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60860

namespace LeftBound60868
def owner : Owner := ⟨.program ⟨214⟩, ⟨18502⟩⟩
def transferEvent : Nat := 60868
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60866 .coefficient, .predecessor 1 60867 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60866 .coefficient)
      LeftAuthority60864.bound (LeftAuthority60864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60867 .coefficient)
      LeftBound60860.bound (LeftBound60860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60860.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority60864.bound, LeftBound60860.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority60864.bound, LeftBound60860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority60864.actual selector witness, LeftBound60860.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60868

namespace LeftBound60872
def owner : Owner := ⟨.program ⟨214⟩, ⟨18686⟩⟩
def transferEvent : Nat := 60872
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60870 .coefficient, .predecessor 1 60871 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60870 .coefficient)
      LeftBound60868.bound (LeftBound60868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60871 .coefficient)
      LeftBound60781.bound (LeftBound60781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60781.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60868.bound, LeftBound60781.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60868.bound, LeftBound60781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60868.actual selector witness, LeftBound60781.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60872

namespace LeftBound60919
def owner : Owner := ⟨.program ⟨214⟩, ⟨30145⟩⟩
def transferEvent : Nat := 60919
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60917 .coefficient, .predecessor 1 60918 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60917 .coefficient)
      LeftBound59510.bound (LeftBound59510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60918 .coefficient)
      LeftBound59425.bound (LeftBound59425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59510.bound, LeftBound59425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59510.bound, LeftBound59425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59510.actual selector witness, LeftBound59425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60919

namespace LeftBound60956
def owner : Owner := ⟨.program ⟨214⟩, ⟨30145⟩⟩
def transferEvent : Nat := 60956
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60916 .summary, .result 59500 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 60916 .summary)
      LeftBound59512.bound (LeftBound59512.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18566⟩⟩) (rawTerms := some (Proof.Events237.exact60916RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59500 .summary)
      LeftBound59427.bound (LeftBound59427.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30144⟩⟩) (rawTerms := some (Proof.Events232.exact59500RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59427.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59512.bound, LeftBound59427.bound]
def bound : CoeffClass := .finite ⟨85361036953731455419885957120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59512.bound, LeftBound59427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59512.actual selector witness, LeftBound59427.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60956

namespace LeftBound60960
def owner : Owner := ⟨.program ⟨214⟩, ⟨30146⟩⟩
def transferEvent : Nat := 60960
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 60958 .coefficient) (.predecessor 1 60959 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60958 .coefficient)
      LeftBound60919.bound (LeftBound60919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact60957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60959 .coefficient)
      LeftBound5498.bound (LeftBound5498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound60919.bound LeftBound5498.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60919.bound, LeftBound5498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound60919.actual selector witness) * (LeftBound5498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60960

namespace LeftBound60961
def owner : Owner := ⟨.program ⟨214⟩, ⟨30146⟩⟩
def transferEvent : Nat := 60961
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩ [⟨.result 5495 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5495 .coefficient)
      LeftAuthority5494.bound (LeftAuthority5494.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6651⟩⟩) (rawTerms := some (Proof.Events021.exact5495RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5494.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5494.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5494.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound60961

namespace LeftBound60962
def owner : Owner := ⟨.program ⟨214⟩, ⟨30146⟩⟩
def transferEvent : Nat := 60962
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 60957 .summary) (.transfer 60961) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 60957 .summary)
      LeftBound60956.bound (LeftBound60956.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30145⟩⟩) (rawTerms := some (Proof.Events238.exact60957RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 60961)
      LeftBound60961.bound (LeftBound60961.actual selector witness) := by
  exact .transfer (LeftBound60961.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound60956.bound LeftBound60961.bound
def bound : CoeffClass := .finite ⟨313276371396785701094268180805713920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60956.bound, LeftBound60961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound60956.actual selector witness) * (LeftBound60961.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60962

namespace LeftBound60977
def owner : Owner := ⟨.program ⟨214⟩, ⟨30134⟩⟩
def transferEvent : Nat := 60977
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 60975 .coefficient) (.predecessor 1 60976 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60975 .coefficient)
      LeftBound50944.bound (LeftBound50944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact50948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60976 .coefficient)
      LeftAuthority60973.bound (LeftAuthority60973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact60974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60973.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50944.bound LeftAuthority60973.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50944.bound, LeftAuthority60973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50944.actual selector witness) * (LeftAuthority60973.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60977

namespace LeftBound60978
def owner : Owner := ⟨.program ⟨214⟩, ⟨30134⟩⟩
def transferEvent : Nat := 60978
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30132⟩⟩]⟩ [⟨.result 60974 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 60974 .coefficient)
      LeftAuthority60973.bound (LeftAuthority60973.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30132⟩⟩) (rawTerms := some (Proof.Events238.exact60974RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60973.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority60973.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority60973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority60973.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound60978

namespace LeftBound60979
def owner : Owner := ⟨.program ⟨214⟩, ⟨30134⟩⟩
def transferEvent : Nat := 60979
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50948 .summary) (.transfer 60978) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50948 .summary)
      LeftBound50947.bound (LeftBound50947.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25765⟩⟩) (rawTerms := some (Proof.Events199.exact50948RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 60978)
      LeftBound60978.bound (LeftBound60978.actual selector witness) := by
  exact .transfer (LeftBound60978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50947.bound LeftBound60978.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50947.bound, LeftBound60978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50947.actual selector witness) * (LeftBound60978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60979

namespace LeftBound60990
def owner : Owner := ⟨.program ⟨214⟩, ⟨22774⟩⟩
def transferEvent : Nat := 60990
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 60988 .coefficient) (.value (.predecessor 1 60989 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60988 .coefficient)
      LeftAuthority60986.bound (LeftAuthority60986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact60987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60989 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority60986.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority60986.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority60986.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound60990

namespace LeftBound60994
def owner : Owner := ⟨.program ⟨214⟩, ⟨22775⟩⟩
def transferEvent : Nat := 60994
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 60992 .coefficient) (.predecessor 1 60993 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60992 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60993 .coefficient)
      LeftBound60990.bound (LeftBound60990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact60991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60990.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound60990.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound60990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound60990.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60994

namespace LeftBound60995
def owner : Owner := ⟨.program ⟨214⟩, ⟨22775⟩⟩
def transferEvent : Nat := 60995
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22772⟩⟩]⟩ [⟨.result 60987 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 60987 .coefficient)
      LeftAuthority60986.bound (LeftAuthority60986.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22772⟩⟩) (rawTerms := some (Proof.Events238.exact60987RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60986.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority60986.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority60986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority60986.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound60995

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
