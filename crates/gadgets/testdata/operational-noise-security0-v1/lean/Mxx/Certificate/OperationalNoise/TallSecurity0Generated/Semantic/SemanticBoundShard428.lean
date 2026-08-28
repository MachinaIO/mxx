import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard391
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard392

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63945
def owner : Owner := ⟨.program ⟨214⟩, ⟨27006⟩⟩
def transferEvent : Nat := 63945
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63943 .coefficient) (.predecessor 1 63944 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63943 .coefficient)
      LeftBound57692.bound (LeftBound57692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63944 .coefficient)
      LeftAuthority63941.bound (LeftAuthority63941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63941.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63941.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57692.bound LeftAuthority63941.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57692.bound, LeftAuthority63941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57692.actual selector witness) * (LeftAuthority63941.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63945

namespace LeftBound63946
def owner : Owner := ⟨.program ⟨214⟩, ⟨27006⟩⟩
def transferEvent : Nat := 63946
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27004⟩⟩]⟩ [⟨.result 63942 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63942 .coefficient)
      LeftAuthority63941.bound (LeftAuthority63941.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27004⟩⟩) (rawTerms := some (Proof.Events249.exact63942RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63941.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63941.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63941.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63941.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63946

namespace LeftBound63947
def owner : Owner := ⟨.program ⟨214⟩, ⟨27006⟩⟩
def transferEvent : Nat := 63947
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 57696 .summary) (.transfer 63946) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57696 .summary)
      LeftBound57695.bound (LeftBound57695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25303⟩⟩) (rawTerms := some (Proof.Events225.exact57696RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63946)
      LeftBound63946.bound (LeftBound63946.actual selector witness) := by
  exact .transfer (LeftBound63946.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57695.bound LeftBound63946.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57695.bound, LeftBound63946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57695.actual selector witness) * (LeftBound63946.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63947

namespace LeftBound63958
def owner : Owner := ⟨.program ⟨214⟩, ⟨20758⟩⟩
def transferEvent : Nat := 63958
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 63956 .coefficient) (.value (.predecessor 1 63957 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63956 .coefficient)
      LeftAuthority63954.bound (LeftAuthority63954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63954.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63957 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority63954.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63954.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63954.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound63958

namespace LeftBound63962
def owner : Owner := ⟨.program ⟨214⟩, ⟨20759⟩⟩
def transferEvent : Nat := 63962
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63960 .coefficient) (.predecessor 1 63961 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63960 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63961 .coefficient)
      LeftBound63958.bound (LeftBound63958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63958.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound63958.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound63958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound63958.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63962

namespace LeftBound63963
def owner : Owner := ⟨.program ⟨214⟩, ⟨20759⟩⟩
def transferEvent : Nat := 63963
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20756⟩⟩]⟩ [⟨.result 63955 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63955 .coefficient)
      LeftAuthority63954.bound (LeftAuthority63954.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20756⟩⟩) (rawTerms := some (Proof.Events249.exact63955RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63954.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63954.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63954.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63954.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63963

namespace LeftBound63964
def owner : Owner := ⟨.program ⟨214⟩, ⟨20759⟩⟩
def transferEvent : Nat := 63964
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 63963) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63963)
      LeftBound63963.bound (LeftBound63963.actual selector witness) := by
  exact .transfer (LeftBound63963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound63963.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound63963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound63963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63964

namespace LeftBound64059
def owner : Owner := ⟨.program ⟨214⟩, ⟨15427⟩⟩
def transferEvent : Nat := 64059
def frameStart : Nat := 64020
def rule : BoundRule := .identity (.predecessor 0 64058 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64058 .coefficient)
      LeftAuthority64056.bound (LeftAuthority64056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64056.derived selector witness)

def rawBound : CoeffClass := LeftAuthority64056.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority64056.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64059

namespace LeftBound64076
def owner : Owner := ⟨.program ⟨214⟩, ⟨15466⟩⟩
def transferEvent : Nat := 64076
def frameStart : Nat := 64020
def rule : BoundRule := .sum [.predecessor 0 64074 .coefficient, .predecessor 1 64075 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64074 .coefficient)
      LeftBound64059.bound (LeftBound64059.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64075 .coefficient)
      LeftAuthority64072.bound (LeftAuthority64072.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority64072.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64059.bound, LeftAuthority64072.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64059.bound, LeftAuthority64072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64059.actual selector witness, LeftAuthority64072.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64076

namespace LeftBound64079
def owner : Owner := ⟨.program ⟨214⟩, ⟨15467⟩⟩
def transferEvent : Nat := 64079
def frameStart : Nat := 64020
def rule : BoundRule := .identity (.predecessor 0 64078 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64078 .coefficient)
      LeftBound64076.bound (LeftBound64076.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64076.derived selector witness)

def rawBound : CoeffClass := LeftBound64076.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound64076.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64079

namespace LeftBound64085
def owner : Owner := ⟨.program ⟨214⟩, ⟨15468⟩⟩
def transferEvent : Nat := 64085
def frameStart : Nat := 64020
def rule : BoundRule := .product (.predecessor 0 64083 .coefficient) (.predecessor 1 64084 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64083 .coefficient)
      LeftAuthority64081.bound (LeftAuthority64081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64081.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64084 .coefficient)
      LeftBound64079.bound (LeftBound64079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64079.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority64081.bound LeftBound64079.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64081.bound, LeftBound64079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority64081.actual selector witness) * (LeftBound64079.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64085

namespace LeftBound64093
def owner : Owner := ⟨.program ⟨214⟩, ⟨15469⟩⟩
def transferEvent : Nat := 64093
def frameStart : Nat := 64020
def rule : BoundRule := .sum [.predecessor 0 64091 .coefficient, .predecessor 1 64092 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64091 .coefficient)
      LeftAuthority64089.bound (LeftAuthority64089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64089.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64089.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64092 .coefficient)
      LeftBound64085.bound (LeftBound64085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64089.bound, LeftBound64085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64089.bound, LeftBound64085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority64089.actual selector witness, LeftBound64085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64093

namespace LeftBound64097
def owner : Owner := ⟨.program ⟨214⟩, ⟨27005⟩⟩
def transferEvent : Nat := 64097
def frameStart : Nat := 64020
def rule : BoundRule := .product (.predecessor 0 64095 .coefficient) (.predecessor 1 64096 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64095 .coefficient)
      LeftBound64093.bound (LeftBound64093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64093.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64096 .coefficient)
      LeftAuthority64070.bound (LeftAuthority64070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64070.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64070.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64093.bound LeftAuthority64070.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64093.bound, LeftAuthority64070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64093.actual selector witness) * (LeftAuthority64070.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64097

namespace LeftBound64108
def owner : Owner := ⟨.program ⟨214⟩, ⟨15524⟩⟩
def transferEvent : Nat := 64108
def frameStart : Nat := 64020
def rule : BoundRule := .product (.predecessor 0 64106 .coefficient) (.predecessor 1 64107 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64106 .coefficient)
      LeftAuthority64081.bound (LeftAuthority64081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64081.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64107 .coefficient)
      LeftAuthority64104.bound (LeftAuthority64104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64104.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64104.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority64081.bound LeftAuthority64104.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64081.bound, LeftAuthority64104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority64081.actual selector witness) * (LeftAuthority64104.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64108

namespace LeftBound64116
def owner : Owner := ⟨.program ⟨214⟩, ⟨15525⟩⟩
def transferEvent : Nat := 64116
def frameStart : Nat := 64020
def rule : BoundRule := .sum [.predecessor 0 64114 .coefficient, .predecessor 1 64115 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64114 .coefficient)
      LeftAuthority64112.bound (LeftAuthority64112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64112.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64115 .coefficient)
      LeftBound64108.bound (LeftBound64108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64108.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64112.bound, LeftBound64108.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64112.bound, LeftBound64108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority64112.actual selector witness, LeftBound64108.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64116

namespace LeftBound64120
def owner : Owner := ⟨.program ⟨214⟩, ⟨27010⟩⟩
def transferEvent : Nat := 64120
def frameStart : Nat := 64020
def rule : BoundRule := .sum [.predecessor 0 64118 .coefficient, .predecessor 1 64119 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64118 .coefficient)
      LeftBound64116.bound (LeftBound64116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64119 .coefficient)
      LeftBound64097.bound (LeftBound64097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64097.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64097.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64116.bound, LeftBound64097.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64116.bound, LeftBound64097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64116.actual selector witness, LeftBound64097.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64120

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
