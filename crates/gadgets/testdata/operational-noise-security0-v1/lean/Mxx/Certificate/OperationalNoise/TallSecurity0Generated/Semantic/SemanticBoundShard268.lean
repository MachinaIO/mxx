import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard267

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40081
def owner : Owner := ⟨.program ⟨214⟩, ⟨14661⟩⟩
def transferEvent : Nat := 40081
def frameStart : Nat := 40048
def rule : BoundRule := .identity (.predecessor 0 40080 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40080 .coefficient)
      LeftBound40077.bound (LeftBound40077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40077.derived selector witness)

def rawBound : CoeffClass := LeftBound40077.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound40077.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40081

namespace LeftBound40098
def owner : Owner := ⟨.program ⟨214⟩, ⟨14756⟩⟩
def transferEvent : Nat := 40098
def frameStart : Nat := 40048
def rule : BoundRule := .sum [.predecessor 0 40096 .coefficient, .predecessor 1 40097 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40096 .coefficient)
      LeftBound40081.bound (LeftBound40081.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound40081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40097 .coefficient)
      LeftAuthority40094.bound (LeftAuthority40094.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority40094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40081.bound, LeftAuthority40094.bound]
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40081.bound, LeftAuthority40094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40081.actual selector witness, LeftAuthority40094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40098

namespace LeftBound40101
def owner : Owner := ⟨.program ⟨214⟩, ⟨14757⟩⟩
def transferEvent : Nat := 40101
def frameStart : Nat := 40048
def rule : BoundRule := .identity (.predecessor 0 40100 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40100 .coefficient)
      LeftBound40098.bound (LeftBound40098.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound40098.derived selector witness)

def rawBound : CoeffClass := LeftBound40098.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound40098.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40101

namespace LeftBound40107
def owner : Owner := ⟨.program ⟨214⟩, ⟨14758⟩⟩
def transferEvent : Nat := 40107
def frameStart : Nat := 40048
def rule : BoundRule := .product (.predecessor 0 40105 .coefficient) (.predecessor 1 40106 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40105 .coefficient)
      LeftAuthority40103.bound (LeftAuthority40103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40106 .coefficient)
      LeftBound40101.bound (LeftBound40101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40101.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority40103.bound LeftBound40101.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40103.bound, LeftBound40101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority40103.actual selector witness) * (LeftBound40101.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40107

namespace LeftBound40123
def owner : Owner := ⟨.program ⟨214⟩, ⟨7859⟩⟩
def transferEvent : Nat := 40123
def frameStart : Nat := 40048
def rule : BoundRule := .scale (.predecessor 0 40121 .coefficient) (.value (.predecessor 1 40122 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40121 .coefficient)
      LeftAuthority40119.bound (LeftAuthority40119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40122 .coefficient)
      LeftAuthority40110.bound (LeftAuthority40110.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority40110.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority40119.bound LeftAuthority40110.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40119.bound, LeftAuthority40110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40119.actual selector witness) * (LeftAuthority40110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40123

namespace LeftBound40126
def owner : Owner := ⟨.program ⟨214⟩, ⟨6762⟩⟩
def transferEvent : Nat := 40126
def frameStart : Nat := 40048
def rule : BoundRule := .identity (.predecessor 0 40125 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40125 .coefficient)
      LeftAuthority40113.bound (LeftAuthority40113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40113.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40113.derived selector witness)

def rawBound : CoeffClass := LeftAuthority40113.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority40113.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40126

namespace LeftBound40130
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def transferEvent : Nat := 40130
def frameStart : Nat := 40048
def rule : BoundRule := .product (.predecessor 0 40128 .coefficient) (.predecessor 1 40129 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40128 .coefficient)
      LeftBound40126.bound (LeftBound40126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40129 .coefficient)
      LeftBound40123.bound (LeftBound40123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40126.bound LeftBound40123.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40126.bound, LeftBound40123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40126.actual selector witness) * (LeftBound40123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40130

namespace LeftBound40135
def owner : Owner := ⟨.program ⟨214⟩, ⟨14759⟩⟩
def transferEvent : Nat := 40135
def frameStart : Nat := 40048
def rule : BoundRule := .sum [.predecessor 0 40133 .coefficient, .predecessor 1 40134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40133 .coefficient)
      LeftBound40130.bound (LeftBound40130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40134 .coefficient)
      LeftBound40107.bound (LeftBound40107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40107.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40130.bound, LeftBound40107.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40130.bound, LeftBound40107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40130.actual selector witness, LeftBound40107.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40135

namespace LeftBound40139
def owner : Owner := ⟨.program ⟨214⟩, ⟨26233⟩⟩
def transferEvent : Nat := 40139
def frameStart : Nat := 40048
def rule : BoundRule := .product (.predecessor 0 40137 .coefficient) (.predecessor 1 40138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40137 .coefficient)
      LeftBound40135.bound (LeftBound40135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40138 .coefficient)
      LeftAuthority40092.bound (LeftAuthority40092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40135.bound LeftAuthority40092.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40135.bound, LeftAuthority40092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40135.actual selector witness) * (LeftAuthority40092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40139

namespace LeftBound40150
def owner : Owner := ⟨.program ⟨214⟩, ⟨16188⟩⟩
def transferEvent : Nat := 40150
def frameStart : Nat := 40048
def rule : BoundRule := .product (.predecessor 0 40148 .coefficient) (.predecessor 1 40149 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40148 .coefficient)
      LeftAuthority40103.bound (LeftAuthority40103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40149 .coefficient)
      LeftAuthority40146.bound (LeftAuthority40146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40146.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority40103.bound LeftAuthority40146.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40103.bound, LeftAuthority40146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority40103.actual selector witness) * (LeftAuthority40146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40150

namespace LeftBound40158
def owner : Owner := ⟨.program ⟨214⟩, ⟨16189⟩⟩
def transferEvent : Nat := 40158
def frameStart : Nat := 40048
def rule : BoundRule := .sum [.predecessor 0 40156 .coefficient, .predecessor 1 40157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40156 .coefficient)
      LeftAuthority40154.bound (LeftAuthority40154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40157 .coefficient)
      LeftBound40150.bound (LeftBound40150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority40154.bound, LeftBound40150.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40154.bound, LeftBound40150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority40154.actual selector witness, LeftBound40150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40158

namespace LeftBound40162
def owner : Owner := ⟨.program ⟨214⟩, ⟨26234⟩⟩
def transferEvent : Nat := 40162
def frameStart : Nat := 40048
def rule : BoundRule := .sum [.predecessor 0 40160 .coefficient, .predecessor 1 40161 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40160 .coefficient)
      LeftBound40158.bound (LeftBound40158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40161 .coefficient)
      LeftBound40139.bound (LeftBound40139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40158.bound, LeftBound40139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40158.bound, LeftBound40139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40158.actual selector witness, LeftBound40139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40162

namespace LeftBound40175
def owner : Owner := ⟨.program ⟨214⟩, ⟨26232⟩⟩
def transferEvent : Nat := 40175
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40173 .coefficient, .predecessor 1 40174 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40173 .coefficient)
      LeftBound39996.bound (LeftBound39996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40174 .coefficient)
      LeftBound39979.bound (LeftBound39979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact39986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39996.bound, LeftBound39979.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39996.bound, LeftBound39979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39996.actual selector witness, LeftBound39979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40175

namespace LeftBound40178
def owner : Owner := ⟨.program ⟨214⟩, ⟨26232⟩⟩
def transferEvent : Nat := 40178
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40172 .summary, .result 39986 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40172 .summary)
      LeftBound39998.bound (LeftBound39998.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19683⟩⟩) (rawTerms := some (Proof.Events156.exact40172RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39986 .summary)
      LeftBound39981.bound (LeftBound39981.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26231⟩⟩) (rawTerms := some (Proof.Events156.exact39986RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39981.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39998.bound, LeftBound39981.bound]
def bound : CoeffClass := .finite ⟨352091253649408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39998.bound, LeftBound39981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39998.actual selector witness, LeftBound39981.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40178

namespace LeftBound40182
def owner : Owner := ⟨.program ⟨214⟩, ⟨28328⟩⟩
def transferEvent : Nat := 40182
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40180 .coefficient) (.predecessor 1 40181 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40180 .coefficient)
      LeftBound40175.bound (LeftBound40175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40181 .coefficient)
      LeftAuthority39901.bound (LeftAuthority39901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39901.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40175.bound LeftAuthority39901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40175.bound, LeftAuthority39901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40175.actual selector witness) * (LeftAuthority39901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40182

namespace LeftBound40183
def owner : Owner := ⟨.program ⟨214⟩, ⟨28328⟩⟩
def transferEvent : Nat := 40183
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩ [⟨.result 39902 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39902 .coefficient)
      LeftAuthority39901.bound (LeftAuthority39901.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28326⟩⟩) (rawTerms := some (Proof.Events155.exact39902RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39901.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority39901.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39901.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40183

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
