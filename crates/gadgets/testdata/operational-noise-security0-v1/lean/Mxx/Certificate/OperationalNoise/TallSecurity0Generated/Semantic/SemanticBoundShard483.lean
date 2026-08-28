import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard482

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound70897
def owner : Owner := ⟨.program ⟨214⟩, ⟨21255⟩⟩
def transferEvent : Nat := 70897
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 70896) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70896)
      LeftBound70896.bound (LeftBound70896.actual selector witness) := by
  exact .transfer (LeftBound70896.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound70896.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound70896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound70896.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70897

namespace LeftBound70992
def owner : Owner := ⟨.program ⟨214⟩, ⟨15818⟩⟩
def transferEvent : Nat := 70992
def frameStart : Nat := 70953
def rule : BoundRule := .identity (.predecessor 0 70991 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70991 .coefficient)
      LeftAuthority70989.bound (LeftAuthority70989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact70990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70989.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70989.derived selector witness)

def rawBound : CoeffClass := LeftAuthority70989.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority70989.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70992

namespace LeftBound71009
def owner : Owner := ⟨.program ⟨214⟩, ⟨15892⟩⟩
def transferEvent : Nat := 71009
def frameStart : Nat := 70953
def rule : BoundRule := .sum [.predecessor 0 71007 .coefficient, .predecessor 1 71008 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71007 .coefficient)
      LeftBound70992.bound (LeftBound70992.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71008 .coefficient)
      LeftAuthority71005.bound (LeftAuthority71005.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority71005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70992.bound, LeftAuthority71005.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70992.bound, LeftAuthority71005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70992.actual selector witness, LeftAuthority71005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71009

namespace LeftBound71012
def owner : Owner := ⟨.program ⟨214⟩, ⟨15893⟩⟩
def transferEvent : Nat := 71012
def frameStart : Nat := 70953
def rule : BoundRule := .identity (.predecessor 0 71011 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71011 .coefficient)
      LeftBound71009.bound (LeftBound71009.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71009.derived selector witness)

def rawBound : CoeffClass := LeftBound71009.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound71009.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71012

namespace LeftBound71018
def owner : Owner := ⟨.program ⟨214⟩, ⟨15894⟩⟩
def transferEvent : Nat := 71018
def frameStart : Nat := 70953
def rule : BoundRule := .product (.predecessor 0 71016 .coefficient) (.predecessor 1 71017 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71016 .coefficient)
      LeftAuthority71014.bound (LeftAuthority71014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71017 .coefficient)
      LeftBound71012.bound (LeftBound71012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority71014.bound LeftBound71012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71014.bound, LeftBound71012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority71014.actual selector witness) * (LeftBound71012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71018

namespace LeftBound71026
def owner : Owner := ⟨.program ⟨214⟩, ⟨15895⟩⟩
def transferEvent : Nat := 71026
def frameStart : Nat := 70953
def rule : BoundRule := .sum [.predecessor 0 71024 .coefficient, .predecessor 1 71025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71024 .coefficient)
      LeftAuthority71022.bound (LeftAuthority71022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71025 .coefficient)
      LeftBound71018.bound (LeftBound71018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71018.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority71022.bound, LeftBound71018.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71022.bound, LeftBound71018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority71022.actual selector witness, LeftBound71018.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71026

namespace LeftBound71030
def owner : Owner := ⟨.program ⟨214⟩, ⟨27637⟩⟩
def transferEvent : Nat := 71030
def frameStart : Nat := 70953
def rule : BoundRule := .product (.predecessor 0 71028 .coefficient) (.predecessor 1 71029 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71028 .coefficient)
      LeftBound71026.bound (LeftBound71026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71029 .coefficient)
      LeftAuthority71003.bound (LeftAuthority71003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71003.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71026.bound LeftAuthority71003.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71026.bound, LeftAuthority71003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71026.actual selector witness) * (LeftAuthority71003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71030

namespace LeftBound71041
def owner : Owner := ⟨.program ⟨214⟩, ⟨15865⟩⟩
def transferEvent : Nat := 71041
def frameStart : Nat := 70953
def rule : BoundRule := .product (.predecessor 0 71039 .coefficient) (.predecessor 1 71040 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71039 .coefficient)
      LeftAuthority71014.bound (LeftAuthority71014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71040 .coefficient)
      LeftAuthority71037.bound (LeftAuthority71037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71037.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71037.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority71014.bound LeftAuthority71037.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71014.bound, LeftAuthority71037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority71014.actual selector witness) * (LeftAuthority71037.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71041

namespace LeftBound71049
def owner : Owner := ⟨.program ⟨214⟩, ⟨15866⟩⟩
def transferEvent : Nat := 71049
def frameStart : Nat := 70953
def rule : BoundRule := .sum [.predecessor 0 71047 .coefficient, .predecessor 1 71048 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71047 .coefficient)
      LeftAuthority71045.bound (LeftAuthority71045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71045.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71048 .coefficient)
      LeftBound71041.bound (LeftBound71041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71041.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority71045.bound, LeftBound71041.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71045.bound, LeftBound71041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority71045.actual selector witness, LeftBound71041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71049

namespace LeftBound71053
def owner : Owner := ⟨.program ⟨214⟩, ⟨27641⟩⟩
def transferEvent : Nat := 71053
def frameStart : Nat := 70953
def rule : BoundRule := .sum [.predecessor 0 71051 .coefficient, .predecessor 1 71052 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71051 .coefficient)
      LeftBound71049.bound (LeftBound71049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71049.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71052 .coefficient)
      LeftBound71030.bound (LeftBound71030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71030.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71049.bound, LeftBound71030.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71049.bound, LeftBound71030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71049.actual selector witness, LeftBound71030.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71053

namespace LeftBound71066
def owner : Owner := ⟨.program ⟨214⟩, ⟨27639⟩⟩
def transferEvent : Nat := 71066
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71064 .coefficient, .predecessor 1 71065 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71064 .coefficient)
      LeftBound70895.bound (LeftBound70895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71065 .coefficient)
      LeftBound70878.bound (LeftBound70878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70895.bound, LeftBound70878.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70895.bound, LeftBound70878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70895.actual selector witness, LeftBound70878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71066

namespace LeftBound71069
def owner : Owner := ⟨.program ⟨214⟩, ⟨27639⟩⟩
def transferEvent : Nat := 71069
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 71063 .summary, .result 70885 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71063 .summary)
      LeftBound70897.bound (LeftBound70897.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21255⟩⟩) (rawTerms := some (Proof.Events277.exact71063RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70885 .summary)
      LeftBound70880.bound (LeftBound70880.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27638⟩⟩) (rawTerms := some (Proof.Events276.exact70885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70897.bound, LeftBound70880.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70897.bound, LeftBound70880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70897.actual selector witness, LeftBound70880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71069

namespace LeftBound71093
def owner : Owner := ⟨.program ⟨214⟩, ⟨11298⟩⟩
def transferEvent : Nat := 71093
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 71091 .coefficient) (.predecessor 1 71092 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71091 .coefficient)
      LeftAuthority3361.bound (LeftAuthority3361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3361.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71092 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3361.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3361.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3361.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71093

namespace LeftBound71098
def owner : Owner := ⟨.program ⟨214⟩, ⟨7195⟩⟩
def transferEvent : Nat := 71098
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71096 .coefficient) (.predecessor 1 71097 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71096 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71097 .coefficient)
      LeftBound12483.bound (LeftBound12483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound12483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound12483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound12483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71098

namespace LeftBound71103
def owner : Owner := ⟨.program ⟨214⟩, ⟨11299⟩⟩
def transferEvent : Nat := 71103
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71101 .coefficient, .predecessor 1 71102 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71101 .coefficient)
      LeftBound71098.bound (LeftBound71098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71098.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71102 .coefficient)
      LeftBound71093.bound (LeftBound71093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71093.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71098.bound, LeftBound71093.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71098.bound, LeftBound71093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71098.actual selector witness, LeftBound71093.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71103

namespace LeftBound71107
def owner : Owner := ⟨.program ⟨214⟩, ⟨11300⟩⟩
def transferEvent : Nat := 71107
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71105 .coefficient, .predecessor 1 71106 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71105 .coefficient)
      LeftBound71103.bound (LeftBound71103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71106 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71103.bound, LeftBound12475.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71103.bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71103.actual selector witness, LeftBound12475.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71107

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
