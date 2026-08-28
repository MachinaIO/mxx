import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard557

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound82046
def owner : Owner := ⟨.program ⟨214⟩, ⟨12664⟩⟩
def transferEvent : Nat := 82046
def frameStart : Nat := 81987
def rule : BoundRule := .product (.predecessor 0 82044 .coefficient) (.predecessor 1 82045 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82044 .coefficient)
      LeftAuthority82042.bound (LeftAuthority82042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82042.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82045 .coefficient)
      LeftBound82040.bound (LeftBound82040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority82042.bound LeftBound82040.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82042.bound, LeftBound82040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority82042.actual selector witness) * (LeftBound82040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82046

namespace LeftBound82060
def owner : Owner := ⟨.program ⟨214⟩, ⟨7871⟩⟩
def transferEvent : Nat := 82060
def frameStart : Nat := 81987
def rule : BoundRule := .scale (.predecessor 0 82058 .coefficient) (.value (.predecessor 1 82059 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82058 .coefficient)
      LeftAuthority82056.bound (LeftAuthority82056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82059 .coefficient)
      LeftAuthority81990.bound (LeftAuthority81990.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority81990.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority82056.bound LeftAuthority81990.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82056.bound, LeftAuthority81990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82056.actual selector witness) * (LeftAuthority81990.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82060

namespace LeftBound82063
def owner : Owner := ⟨.program ⟨214⟩, ⟨6766⟩⟩
def transferEvent : Nat := 82063
def frameStart : Nat := 81987
def rule : BoundRule := .identity (.predecessor 0 82062 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82062 .coefficient)
      LeftAuthority82050.bound (LeftAuthority82050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82050.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82050.derived selector witness)

def rawBound : CoeffClass := LeftAuthority82050.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority82050.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82063

namespace LeftBound82067
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def transferEvent : Nat := 82067
def frameStart : Nat := 81987
def rule : BoundRule := .product (.predecessor 0 82065 .coefficient) (.predecessor 1 82066 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82065 .coefficient)
      LeftBound82063.bound (LeftBound82063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82066 .coefficient)
      LeftBound82060.bound (LeftBound82060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82060.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82060.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82063.bound LeftBound82060.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82063.bound, LeftBound82060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82063.actual selector witness) * (LeftBound82060.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82067

namespace LeftBound82072
def owner : Owner := ⟨.program ⟨214⟩, ⟨12665⟩⟩
def transferEvent : Nat := 82072
def frameStart : Nat := 81987
def rule : BoundRule := .sum [.predecessor 0 82070 .coefficient, .predecessor 1 82071 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82070 .coefficient)
      LeftBound82067.bound (LeftBound82067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82071 .coefficient)
      LeftBound82046.bound (LeftBound82046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82046.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82046.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82067.bound, LeftBound82046.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82067.bound, LeftBound82046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82067.actual selector witness, LeftBound82046.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82072

namespace LeftBound82076
def owner : Owner := ⟨.program ⟨214⟩, ⟨25453⟩⟩
def transferEvent : Nat := 82076
def frameStart : Nat := 81987
def rule : BoundRule := .product (.predecessor 0 82074 .coefficient) (.predecessor 1 82075 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82074 .coefficient)
      LeftBound82072.bound (LeftBound82072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82075 .coefficient)
      LeftAuthority82031.bound (LeftAuthority82031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82031.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82072.bound LeftAuthority82031.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82072.bound, LeftAuthority82031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82072.actual selector witness) * (LeftAuthority82031.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82076

namespace LeftBound82087
def owner : Owner := ⟨.program ⟨214⟩, ⟨16551⟩⟩
def transferEvent : Nat := 82087
def frameStart : Nat := 81987
def rule : BoundRule := .product (.predecessor 0 82085 .coefficient) (.predecessor 1 82086 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82085 .coefficient)
      LeftAuthority82042.bound (LeftAuthority82042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82042.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82086 .coefficient)
      LeftAuthority82083.bound (LeftAuthority82083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82083.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority82042.bound LeftAuthority82083.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82042.bound, LeftAuthority82083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority82042.actual selector witness) * (LeftAuthority82083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82087

namespace LeftBound82095
def owner : Owner := ⟨.program ⟨214⟩, ⟨16552⟩⟩
def transferEvent : Nat := 82095
def frameStart : Nat := 81987
def rule : BoundRule := .sum [.predecessor 0 82093 .coefficient, .predecessor 1 82094 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82093 .coefficient)
      LeftAuthority82091.bound (LeftAuthority82091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82091.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82094 .coefficient)
      LeftBound82087.bound (LeftBound82087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority82091.bound, LeftBound82087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82091.bound, LeftBound82087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority82091.actual selector witness, LeftBound82087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82095

namespace LeftBound82099
def owner : Owner := ⟨.program ⟨214⟩, ⟨25454⟩⟩
def transferEvent : Nat := 82099
def frameStart : Nat := 81987
def rule : BoundRule := .sum [.predecessor 0 82097 .coefficient, .predecessor 1 82098 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82097 .coefficient)
      LeftBound82095.bound (LeftBound82095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82098 .coefficient)
      LeftBound82076.bound (LeftBound82076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82076.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82076.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82095.bound, LeftBound82076.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82095.bound, LeftBound82076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82095.actual selector witness, LeftBound82076.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82099

namespace LeftBound82112
def owner : Owner := ⟨.program ⟨214⟩, ⟨25452⟩⟩
def transferEvent : Nat := 82112
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82110 .coefficient, .predecessor 1 82111 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82110 .coefficient)
      LeftBound81935.bound (LeftBound81935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82111 .coefficient)
      LeftBound81918.bound (LeftBound81918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact81925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81918.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81935.bound, LeftBound81918.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81935.bound, LeftBound81918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81935.actual selector witness, LeftBound81918.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82112

namespace LeftBound82115
def owner : Owner := ⟨.program ⟨214⟩, ⟨25452⟩⟩
def transferEvent : Nat := 82115
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 82109 .summary, .result 81925 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82109 .summary)
      LeftBound81937.bound (LeftBound81937.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19963⟩⟩) (rawTerms := some (Proof.Events320.exact82109RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81925 .summary)
      LeftBound81920.bound (LeftBound81920.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25451⟩⟩) (rawTerms := some (Proof.Events320.exact81925RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81920.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81937.bound, LeftBound81920.bound]
def bound : CoeffClass := .finite ⟨352134001995776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81937.bound, LeftBound81920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81937.actual selector witness, LeftBound81920.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82115

namespace LeftBound82119
def owner : Owner := ⟨.program ⟨214⟩, ⟨29170⟩⟩
def transferEvent : Nat := 82119
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82117 .coefficient) (.predecessor 1 82118 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82117 .coefficient)
      LeftBound82112.bound (LeftBound82112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82118 .coefficient)
      LeftAuthority81840.bound (LeftAuthority81840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81840.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81840.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82112.bound LeftAuthority81840.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82112.bound, LeftAuthority81840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82112.actual selector witness) * (LeftAuthority81840.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82119

namespace LeftBound82120
def owner : Owner := ⟨.program ⟨214⟩, ⟨29170⟩⟩
def transferEvent : Nat := 82120
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩ [⟨.result 81841 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81841 .coefficient)
      LeftAuthority81840.bound (LeftAuthority81840.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29168⟩⟩) (rawTerms := some (Proof.Events319.exact81841RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81840.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81840.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority81840.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81840.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82120

namespace LeftBound82121
def owner : Owner := ⟨.program ⟨214⟩, ⟨29170⟩⟩
def transferEvent : Nat := 82121
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82116 .summary) (.transfer 82120) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82116 .summary)
      LeftBound82115.bound (LeftBound82115.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25452⟩⟩) (rawTerms := some (Proof.Events320.exact82116RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82115.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82120)
      LeftBound82120.bound (LeftBound82120.actual selector witness) := by
  exact .transfer (LeftBound82120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82115.bound LeftBound82120.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82115.bound, LeftBound82120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82115.actual selector witness) * (LeftBound82120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82121

namespace LeftBound82132
def owner : Owner := ⟨.program ⟨214⟩, ⟨22266⟩⟩
def transferEvent : Nat := 82132
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 82130 .coefficient) (.value (.predecessor 1 82131 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82130 .coefficient)
      LeftAuthority82128.bound (LeftAuthority82128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82131 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority82128.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82128.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82128.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82132

namespace LeftBound82136
def owner : Owner := ⟨.program ⟨214⟩, ⟨22267⟩⟩
def transferEvent : Nat := 82136
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82134 .coefficient) (.predecessor 1 82135 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82134 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82135 .coefficient)
      LeftBound82132.bound (LeftBound82132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82132.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound82132.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound82132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound82132.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82136

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
