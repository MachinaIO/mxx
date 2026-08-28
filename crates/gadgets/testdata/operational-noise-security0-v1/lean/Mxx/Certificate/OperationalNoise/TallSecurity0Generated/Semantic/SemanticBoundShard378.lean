import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard377

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55885
def owner : Owner := ⟨.program ⟨214⟩, ⟨15945⟩⟩
def transferEvent : Nat := 55885
def frameStart : Nat := 55846
def rule : BoundRule := .identity (.predecessor 0 55884 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55884 .coefficient)
      LeftAuthority55882.bound (LeftAuthority55882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55882.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55882.derived selector witness)

def rawBound : CoeffClass := LeftAuthority55882.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority55882.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55885

namespace LeftBound55902
def owner : Owner := ⟨.program ⟨214⟩, ⟨16019⟩⟩
def transferEvent : Nat := 55902
def frameStart : Nat := 55846
def rule : BoundRule := .sum [.predecessor 0 55900 .coefficient, .predecessor 1 55901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55900 .coefficient)
      LeftBound55885.bound (LeftBound55885.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55901 .coefficient)
      LeftAuthority55898.bound (LeftAuthority55898.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority55898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55885.bound, LeftAuthority55898.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55885.bound, LeftAuthority55898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55885.actual selector witness, LeftAuthority55898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55902

namespace LeftBound55905
def owner : Owner := ⟨.program ⟨214⟩, ⟨16020⟩⟩
def transferEvent : Nat := 55905
def frameStart : Nat := 55846
def rule : BoundRule := .identity (.predecessor 0 55904 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55904 .coefficient)
      LeftBound55902.bound (LeftBound55902.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55902.derived selector witness)

def rawBound : CoeffClass := LeftBound55902.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound55902.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55905

namespace LeftBound55911
def owner : Owner := ⟨.program ⟨214⟩, ⟨16021⟩⟩
def transferEvent : Nat := 55911
def frameStart : Nat := 55846
def rule : BoundRule := .product (.predecessor 0 55909 .coefficient) (.predecessor 1 55910 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55909 .coefficient)
      LeftAuthority55907.bound (LeftAuthority55907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55907.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55910 .coefficient)
      LeftBound55905.bound (LeftBound55905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55905.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority55907.bound LeftBound55905.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55907.bound, LeftBound55905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority55907.actual selector witness) * (LeftBound55905.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55911

namespace LeftBound55919
def owner : Owner := ⟨.program ⟨214⟩, ⟨16022⟩⟩
def transferEvent : Nat := 55919
def frameStart : Nat := 55846
def rule : BoundRule := .sum [.predecessor 0 55917 .coefficient, .predecessor 1 55918 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55917 .coefficient)
      LeftAuthority55915.bound (LeftAuthority55915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55918 .coefficient)
      LeftBound55911.bound (LeftBound55911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55911.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55911.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55915.bound, LeftBound55911.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55915.bound, LeftBound55911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority55915.actual selector witness, LeftBound55911.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55919

namespace LeftBound55923
def owner : Owner := ⟨.program ⟨214⟩, ⟨27880⟩⟩
def transferEvent : Nat := 55923
def frameStart : Nat := 55846
def rule : BoundRule := .product (.predecessor 0 55921 .coefficient) (.predecessor 1 55922 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55921 .coefficient)
      LeftBound55919.bound (LeftBound55919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55922 .coefficient)
      LeftAuthority55896.bound (LeftAuthority55896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55896.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55919.bound LeftAuthority55896.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55919.bound, LeftAuthority55896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55919.actual selector witness) * (LeftAuthority55896.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55923

namespace LeftBound55934
def owner : Owner := ⟨.program ⟨214⟩, ⟨15990⟩⟩
def transferEvent : Nat := 55934
def frameStart : Nat := 55846
def rule : BoundRule := .product (.predecessor 0 55932 .coefficient) (.predecessor 1 55933 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55932 .coefficient)
      LeftAuthority55907.bound (LeftAuthority55907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55907.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55933 .coefficient)
      LeftAuthority55930.bound (LeftAuthority55930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55930.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55930.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority55907.bound LeftAuthority55930.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55907.bound, LeftAuthority55930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority55907.actual selector witness) * (LeftAuthority55930.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55934

namespace LeftBound55942
def owner : Owner := ⟨.program ⟨214⟩, ⟨15991⟩⟩
def transferEvent : Nat := 55942
def frameStart : Nat := 55846
def rule : BoundRule := .sum [.predecessor 0 55940 .coefficient, .predecessor 1 55941 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55940 .coefficient)
      LeftAuthority55938.bound (LeftAuthority55938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55938.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55941 .coefficient)
      LeftBound55934.bound (LeftBound55934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55934.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55938.bound, LeftBound55934.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55938.bound, LeftBound55934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority55938.actual selector witness, LeftBound55934.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55942

namespace LeftBound55946
def owner : Owner := ⟨.program ⟨214⟩, ⟨27884⟩⟩
def transferEvent : Nat := 55946
def frameStart : Nat := 55846
def rule : BoundRule := .sum [.predecessor 0 55944 .coefficient, .predecessor 1 55945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55944 .coefficient)
      LeftBound55942.bound (LeftBound55942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55945 .coefficient)
      LeftBound55923.bound (LeftBound55923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55942.bound, LeftBound55923.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55942.bound, LeftBound55923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55942.actual selector witness, LeftBound55923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55946

namespace LeftBound55959
def owner : Owner := ⟨.program ⟨214⟩, ⟨27882⟩⟩
def transferEvent : Nat := 55959
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55957 .coefficient, .predecessor 1 55958 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55957 .coefficient)
      LeftBound55788.bound (LeftBound55788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55788.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55788.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55958 .coefficient)
      LeftBound55771.bound (LeftBound55771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55771.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55788.bound, LeftBound55771.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55788.bound, LeftBound55771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55788.actual selector witness, LeftBound55771.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55959

namespace LeftBound55962
def owner : Owner := ⟨.program ⟨214⟩, ⟨27882⟩⟩
def transferEvent : Nat := 55962
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55956 .summary, .result 55778 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55956 .summary)
      LeftBound55790.bound (LeftBound55790.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21407⟩⟩) (rawTerms := some (Proof.Events218.exact55956RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55790.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55778 .summary)
      LeftBound55773.bound (LeftBound55773.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27881⟩⟩) (rawTerms := some (Proof.Events217.exact55778RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55773.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55790.bound, LeftBound55773.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55790.bound, LeftBound55773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55790.actual selector witness, LeftBound55773.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55962

namespace LeftBound55986
def owner : Owner := ⟨.program ⟨214⟩, ⟨11390⟩⟩
def transferEvent : Nat := 55986
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 55984 .coefficient) (.predecessor 1 55985 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55984 .coefficient)
      LeftAuthority2590.bound (LeftAuthority2590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2590.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55985 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2590.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2590.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2590.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55986

namespace LeftBound55991
def owner : Owner := ⟨.program ⟨214⟩, ⟨7272⟩⟩
def transferEvent : Nat := 55991
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55989 .coefficient) (.predecessor 1 55990 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55989 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55990 .coefficient)
      LeftBound11982.bound (LeftBound11982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11982.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound11982.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound11982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound11982.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55991

namespace LeftBound55996
def owner : Owner := ⟨.program ⟨214⟩, ⟨11391⟩⟩
def transferEvent : Nat := 55996
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55994 .coefficient, .predecessor 1 55995 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55994 .coefficient)
      LeftBound55991.bound (LeftBound55991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55995 .coefficient)
      LeftBound55986.bound (LeftBound55986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55986.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55991.bound, LeftBound55986.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55991.bound, LeftBound55986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55991.actual selector witness, LeftBound55986.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55996

namespace LeftBound56000
def owner : Owner := ⟨.program ⟨214⟩, ⟨11392⟩⟩
def transferEvent : Nat := 56000
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55998 .coefficient, .predecessor 1 55999 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55998 .coefficient)
      LeftBound55996.bound (LeftBound55996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55999 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55996.bound, LeftBound11974.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55996.bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55996.actual selector witness, LeftBound11974.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56000

namespace LeftBound56001
def owner : Owner := ⟨.program ⟨214⟩, ⟨11392⟩⟩
def transferEvent : Nat := 56001
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
end LeftBound56001

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
