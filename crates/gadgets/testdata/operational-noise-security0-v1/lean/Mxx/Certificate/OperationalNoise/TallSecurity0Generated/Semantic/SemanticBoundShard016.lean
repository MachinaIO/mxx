import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard001
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard015

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound5203
def owner : Owner := ⟨.program ⟨214⟩, ⟨14875⟩⟩
def transferEvent : Nat := 5203
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5201 .coefficient, .predecessor 1 5202 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5201 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5202 .coefficient)
      LeftBound5198.bound (LeftBound5198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5198.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound5198.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644362, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound5198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound726.actual selector witness, LeftBound5198.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5203

namespace LeftBound5207
def owner : Owner := ⟨.program ⟨214⟩, ⟨15036⟩⟩
def transferEvent : Nat := 5207
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5205 .coefficient, .predecessor 1 5206 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5205 .coefficient)
      LeftBound5203.bound (LeftBound5203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5206 .coefficient)
      LeftBound5190.bound (LeftBound5190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5190.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5190.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5203.bound, LeftBound5190.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5203.bound, LeftBound5190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5203.actual selector witness, LeftBound5190.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5207

namespace LeftBound5211
def owner : Owner := ⟨.program ⟨214⟩, ⟨15197⟩⟩
def transferEvent : Nat := 5211
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5209 .coefficient, .predecessor 1 5210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5209 .coefficient)
      LeftBound5207.bound (LeftBound5207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5210 .coefficient)
      LeftBound5182.bound (LeftBound5182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5182.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5207.bound, LeftBound5182.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5207.bound, LeftBound5182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5207.actual selector witness, LeftBound5182.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5211

namespace LeftBound5215
def owner : Owner := ⟨.program ⟨214⟩, ⟨15505⟩⟩
def transferEvent : Nat := 5215
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5213 .coefficient, .predecessor 1 5214 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5213 .coefficient)
      LeftBound5211.bound (LeftBound5211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5214 .coefficient)
      LeftBound5174.bound (LeftBound5174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5211.bound, LeftBound5174.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5211.bound, LeftBound5174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5211.actual selector witness, LeftBound5174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5215

namespace LeftBound5219
def owner : Owner := ⟨.program ⟨214⟩, ⟨17794⟩⟩
def transferEvent : Nat := 5219
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5217 .coefficient, .predecessor 1 5218 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5217 .coefficient)
      LeftBound5215.bound (LeftBound5215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5218 .coefficient)
      LeftBound5166.bound (LeftBound5166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5215.bound, LeftBound5166.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5215.bound, LeftBound5166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5215.actual selector witness, LeftBound5166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5219

namespace LeftBound5223
def owner : Owner := ⟨.program ⟨214⟩, ⟨17795⟩⟩
def transferEvent : Nat := 5223
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5221 .coefficient, .predecessor 1 5222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5221 .coefficient)
      LeftBound5219.bound (LeftBound5219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5222 .coefficient)
      LeftBound5158.bound (LeftBound5158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5219.bound, LeftBound5158.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5219.bound, LeftBound5158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5219.actual selector witness, LeftBound5158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5223

namespace LeftBound5227
def owner : Owner := ⟨.program ⟨214⟩, ⟨17796⟩⟩
def transferEvent : Nat := 5227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5225 .coefficient, .predecessor 1 5226 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5225 .coefficient)
      LeftBound5223.bound (LeftBound5223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5226 .coefficient)
      LeftBound5150.bound (LeftBound5150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5223.bound, LeftBound5150.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5223.bound, LeftBound5150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5223.actual selector witness, LeftBound5150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5227

namespace LeftBound5231
def owner : Owner := ⟨.program ⟨214⟩, ⟨17797⟩⟩
def transferEvent : Nat := 5231
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5229 .coefficient, .predecessor 1 5230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5229 .coefficient)
      LeftBound5227.bound (LeftBound5227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5230 .coefficient)
      LeftBound5142.bound (LeftBound5142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5227.bound, LeftBound5142.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5227.bound, LeftBound5142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5227.actual selector witness, LeftBound5142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5231

namespace LeftBound5235
def owner : Owner := ⟨.program ⟨214⟩, ⟨18018⟩⟩
def transferEvent : Nat := 5235
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5233 .coefficient, .predecessor 1 5234 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5233 .coefficient)
      LeftBound5231.bound (LeftBound5231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5234 .coefficient)
      LeftBound5134.bound (LeftBound5134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5231.bound, LeftBound5134.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5231.bound, LeftBound5134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5231.actual selector witness, LeftBound5134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5235

namespace LeftBound5239
def owner : Owner := ⟨.program ⟨214⟩, ⟨18019⟩⟩
def transferEvent : Nat := 5239
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5237 .coefficient, .predecessor 1 5238 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5237 .coefficient)
      LeftBound5235.bound (LeftBound5235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5238 .coefficient)
      LeftBound5126.bound (LeftBound5126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5235.bound, LeftBound5126.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5235.bound, LeftBound5126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5235.actual selector witness, LeftBound5126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5239

namespace LeftBound5243
def owner : Owner := ⟨.program ⟨214⟩, ⟨18020⟩⟩
def transferEvent : Nat := 5243
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5241 .coefficient, .predecessor 1 5242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5241 .coefficient)
      LeftBound5239.bound (LeftBound5239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5242 .coefficient)
      LeftBound5118.bound (LeftBound5118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5239.bound, LeftBound5118.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5239.bound, LeftBound5118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5239.actual selector witness, LeftBound5118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5243

namespace LeftBound5247
def owner : Owner := ⟨.program ⟨214⟩, ⟨18794⟩⟩
def transferEvent : Nat := 5247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5245 .coefficient, .predecessor 1 5246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5245 .coefficient)
      LeftBound5243.bound (LeftBound5243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5246 .coefficient)
      LeftBound5110.bound (LeftBound5110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5243.bound, LeftBound5110.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5243.bound, LeftBound5110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5243.actual selector witness, LeftBound5110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5247

namespace LeftBound5251
def owner : Owner := ⟨.program ⟨214⟩, ⟨18795⟩⟩
def transferEvent : Nat := 5251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5249 .coefficient, .predecessor 1 5250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5249 .coefficient)
      LeftBound5247.bound (LeftBound5247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5250 .coefficient)
      LeftBound5102.bound (LeftBound5102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5247.bound, LeftBound5102.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5247.bound, LeftBound5102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5247.actual selector witness, LeftBound5102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5251

namespace LeftBound5255
def owner : Owner := ⟨.program ⟨214⟩, ⟨18796⟩⟩
def transferEvent : Nat := 5255
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5253 .coefficient, .predecessor 1 5254 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5253 .coefficient)
      LeftBound5251.bound (LeftBound5251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5254 .coefficient)
      LeftBound5094.bound (LeftBound5094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5251.bound, LeftBound5094.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5251.bound, LeftBound5094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5251.actual selector witness, LeftBound5094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5255

namespace LeftBound5259
def owner : Owner := ⟨.program ⟨214⟩, ⟨18797⟩⟩
def transferEvent : Nat := 5259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5257 .coefficient, .predecessor 1 5258 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5257 .coefficient)
      LeftBound5255.bound (LeftBound5255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5258 .coefficient)
      LeftBound5086.bound (LeftBound5086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5255.bound, LeftBound5086.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5255.bound, LeftBound5086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5255.actual selector witness, LeftBound5086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5259

namespace LeftBound5263
def owner : Owner := ⟨.program ⟨214⟩, ⟨18798⟩⟩
def transferEvent : Nat := 5263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5261 .coefficient, .predecessor 1 5262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5261 .coefficient)
      LeftBound5259.bound (LeftBound5259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5262 .coefficient)
      LeftBound5078.bound (LeftBound5078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5259.bound, LeftBound5078.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5259.bound, LeftBound5078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5259.actual selector witness, LeftBound5078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5263

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
