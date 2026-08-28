import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard586

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound85886
def owner : Owner := ⟨.program ⟨214⟩, ⟨13882⟩⟩
def transferEvent : Nat := 85886
def frameStart : Nat := 85827
def rule : BoundRule := .product (.predecessor 0 85884 .coefficient) (.predecessor 1 85885 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85884 .coefficient)
      LeftAuthority85882.bound (LeftAuthority85882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85882.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85882.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85885 .coefficient)
      LeftBound85880.bound (LeftBound85880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85880.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority85882.bound LeftBound85880.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85882.bound, LeftBound85880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority85882.actual selector witness) * (LeftBound85880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85886

namespace LeftBound85900
def owner : Owner := ⟨.program ⟨214⟩, ⟨7847⟩⟩
def transferEvent : Nat := 85900
def frameStart : Nat := 85827
def rule : BoundRule := .scale (.predecessor 0 85898 .coefficient) (.value (.predecessor 1 85899 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85898 .coefficient)
      LeftAuthority85896.bound (LeftAuthority85896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85899 .coefficient)
      LeftAuthority85830.bound (LeftAuthority85830.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority85830.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority85896.bound LeftAuthority85830.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85896.bound, LeftAuthority85830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85896.actual selector witness) * (LeftAuthority85830.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound85900

namespace LeftBound85903
def owner : Owner := ⟨.program ⟨214⟩, ⟨6794⟩⟩
def transferEvent : Nat := 85903
def frameStart : Nat := 85827
def rule : BoundRule := .identity (.predecessor 0 85902 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85902 .coefficient)
      LeftAuthority85890.bound (LeftAuthority85890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85890.derived selector witness)

def rawBound : CoeffClass := LeftAuthority85890.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority85890.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85903

namespace LeftBound85907
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def transferEvent : Nat := 85907
def frameStart : Nat := 85827
def rule : BoundRule := .product (.predecessor 0 85905 .coefficient) (.predecessor 1 85906 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85905 .coefficient)
      LeftBound85903.bound (LeftBound85903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85903.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85903.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85906 .coefficient)
      LeftBound85900.bound (LeftBound85900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85900.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85903.bound LeftBound85900.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85903.bound, LeftBound85900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85903.actual selector witness) * (LeftBound85900.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85907

namespace LeftBound85912
def owner : Owner := ⟨.program ⟨214⟩, ⟨13883⟩⟩
def transferEvent : Nat := 85912
def frameStart : Nat := 85827
def rule : BoundRule := .sum [.predecessor 0 85910 .coefficient, .predecessor 1 85911 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85910 .coefficient)
      LeftBound85907.bound (LeftBound85907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85911 .coefficient)
      LeftBound85886.bound (LeftBound85886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85886.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85886.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85907.bound, LeftBound85886.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85907.bound, LeftBound85886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85907.actual selector witness, LeftBound85886.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85912

namespace LeftBound85916
def owner : Owner := ⟨.program ⟨214⟩, ⟨25915⟩⟩
def transferEvent : Nat := 85916
def frameStart : Nat := 85827
def rule : BoundRule := .product (.predecessor 0 85914 .coefficient) (.predecessor 1 85915 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85914 .coefficient)
      LeftBound85912.bound (LeftBound85912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85915 .coefficient)
      LeftAuthority85871.bound (LeftAuthority85871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85871.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85912.bound LeftAuthority85871.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85912.bound, LeftAuthority85871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85912.actual selector witness) * (LeftAuthority85871.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85916

namespace LeftBound85927
def owner : Owner := ⟨.program ⟨214⟩, ⟨15704⟩⟩
def transferEvent : Nat := 85927
def frameStart : Nat := 85827
def rule : BoundRule := .product (.predecessor 0 85925 .coefficient) (.predecessor 1 85926 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85925 .coefficient)
      LeftAuthority85882.bound (LeftAuthority85882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85882.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85882.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85926 .coefficient)
      LeftAuthority85923.bound (LeftAuthority85923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85923.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority85882.bound LeftAuthority85923.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85882.bound, LeftAuthority85923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority85882.actual selector witness) * (LeftAuthority85923.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85927

namespace LeftBound85935
def owner : Owner := ⟨.program ⟨214⟩, ⟨15705⟩⟩
def transferEvent : Nat := 85935
def frameStart : Nat := 85827
def rule : BoundRule := .sum [.predecessor 0 85933 .coefficient, .predecessor 1 85934 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85933 .coefficient)
      LeftAuthority85931.bound (LeftAuthority85931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85934 .coefficient)
      LeftBound85927.bound (LeftBound85927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority85931.bound, LeftBound85927.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85931.bound, LeftBound85927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority85931.actual selector witness, LeftBound85927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85935

namespace LeftBound85939
def owner : Owner := ⟨.program ⟨214⟩, ⟨25916⟩⟩
def transferEvent : Nat := 85939
def frameStart : Nat := 85827
def rule : BoundRule := .sum [.predecessor 0 85937 .coefficient, .predecessor 1 85938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85937 .coefficient)
      LeftBound85935.bound (LeftBound85935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85938 .coefficient)
      LeftBound85916.bound (LeftBound85916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85935.bound, LeftBound85916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85935.bound, LeftBound85916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85935.actual selector witness, LeftBound85916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85939

namespace LeftBound85952
def owner : Owner := ⟨.program ⟨214⟩, ⟨25914⟩⟩
def transferEvent : Nat := 85952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85950 .coefficient, .predecessor 1 85951 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85950 .coefficient)
      LeftBound85775.bound (LeftBound85775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85951 .coefficient)
      LeftBound85758.bound (LeftBound85758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85758.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85775.bound, LeftBound85758.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85775.bound, LeftBound85758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85775.actual selector witness, LeftBound85758.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85952

namespace LeftBound85955
def owner : Owner := ⟨.program ⟨214⟩, ⟨25914⟩⟩
def transferEvent : Nat := 85955
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 85949 .summary, .result 85765 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85949 .summary)
      LeftBound85777.bound (LeftBound85777.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19387⟩⟩) (rawTerms := some (Proof.Events335.exact85949RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85765 .summary)
      LeftBound85760.bound (LeftBound85760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25913⟩⟩) (rawTerms := some (Proof.Events335.exact85765RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85760.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85777.bound, LeftBound85760.bound]
def bound : CoeffClass := .finite ⟨352042398396416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85777.bound, LeftBound85760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85777.actual selector witness, LeftBound85760.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85955

namespace LeftBound85959
def owner : Owner := ⟨.program ⟨214⟩, ⟨27434⟩⟩
def transferEvent : Nat := 85959
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85957 .coefficient) (.predecessor 1 85958 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85957 .coefficient)
      LeftBound85952.bound (LeftBound85952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85958 .coefficient)
      LeftAuthority85680.bound (LeftAuthority85680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85680.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85680.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85952.bound LeftAuthority85680.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85952.bound, LeftAuthority85680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85952.actual selector witness) * (LeftAuthority85680.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85959

namespace LeftBound85960
def owner : Owner := ⟨.program ⟨214⟩, ⟨27434⟩⟩
def transferEvent : Nat := 85960
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩ [⟨.result 85681 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85681 .coefficient)
      LeftAuthority85680.bound (LeftAuthority85680.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27432⟩⟩) (rawTerms := some (Proof.Events334.exact85681RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85680.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85680.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority85680.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85680.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85960

namespace LeftBound85961
def owner : Owner := ⟨.program ⟨214⟩, ⟨27434⟩⟩
def transferEvent : Nat := 85961
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 85956 .summary) (.transfer 85960) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85956 .summary)
      LeftBound85955.bound (LeftBound85955.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25914⟩⟩) (rawTerms := some (Proof.Events335.exact85956RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 85960)
      LeftBound85960.bound (LeftBound85960.actual selector witness) := by
  exact .transfer (LeftBound85960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85955.bound LeftBound85960.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85955.bound, LeftBound85960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85955.actual selector witness) * (LeftBound85960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85961

namespace LeftBound85972
def owner : Owner := ⟨.program ⟨214⟩, ⟨21114⟩⟩
def transferEvent : Nat := 85972
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 85970 .coefficient) (.value (.predecessor 1 85971 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85970 .coefficient)
      LeftAuthority85968.bound (LeftAuthority85968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85968.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85971 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority85968.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85968.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85968.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound85972

namespace LeftBound85976
def owner : Owner := ⟨.program ⟨214⟩, ⟨21115⟩⟩
def transferEvent : Nat := 85976
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85974 .coefficient) (.predecessor 1 85975 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85974 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85975 .coefficient)
      LeftBound85972.bound (LeftBound85972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound85972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound85972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound85972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85976

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
