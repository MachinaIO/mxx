import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard453

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound67041
def owner : Owner := ⟨.program ⟨214⟩, ⟨22407⟩⟩
def transferEvent : Nat := 67041
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 67040) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67040)
      LeftBound67040.bound (LeftBound67040.actual selector witness) := by
  exact .transfer (LeftBound67040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound67040.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound67040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound67040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67041

namespace LeftBound67136
def owner : Owner := ⟨.program ⟨214⟩, ⟨16630⟩⟩
def transferEvent : Nat := 67136
def frameStart : Nat := 67097
def rule : BoundRule := .identity (.predecessor 0 67135 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67135 .coefficient)
      LeftAuthority67133.bound (LeftAuthority67133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67133.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67133.derived selector witness)

def rawBound : CoeffClass := LeftAuthority67133.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority67133.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67136

namespace LeftBound67153
def owner : Owner := ⟨.program ⟨214⟩, ⟨16704⟩⟩
def transferEvent : Nat := 67153
def frameStart : Nat := 67097
def rule : BoundRule := .sum [.predecessor 0 67151 .coefficient, .predecessor 1 67152 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67151 .coefficient)
      LeftBound67136.bound (LeftBound67136.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound67136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67152 .coefficient)
      LeftAuthority67149.bound (LeftAuthority67149.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority67149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67136.bound, LeftAuthority67149.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67136.bound, LeftAuthority67149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67136.actual selector witness, LeftAuthority67149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67153

namespace LeftBound67156
def owner : Owner := ⟨.program ⟨214⟩, ⟨16705⟩⟩
def transferEvent : Nat := 67156
def frameStart : Nat := 67097
def rule : BoundRule := .identity (.predecessor 0 67155 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67155 .coefficient)
      LeftBound67153.bound (LeftBound67153.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound67153.derived selector witness)

def rawBound : CoeffClass := LeftBound67153.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound67153.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67156

namespace LeftBound67162
def owner : Owner := ⟨.program ⟨214⟩, ⟨16706⟩⟩
def transferEvent : Nat := 67162
def frameStart : Nat := 67097
def rule : BoundRule := .product (.predecessor 0 67160 .coefficient) (.predecessor 1 67161 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67160 .coefficient)
      LeftAuthority67158.bound (LeftAuthority67158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67158.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67161 .coefficient)
      LeftBound67156.bound (LeftBound67156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67156.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority67158.bound LeftBound67156.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67158.bound, LeftBound67156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority67158.actual selector witness) * (LeftBound67156.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67162

namespace LeftBound67170
def owner : Owner := ⟨.program ⟨214⟩, ⟨16707⟩⟩
def transferEvent : Nat := 67170
def frameStart : Nat := 67097
def rule : BoundRule := .sum [.predecessor 0 67168 .coefficient, .predecessor 1 67169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67168 .coefficient)
      LeftAuthority67166.bound (LeftAuthority67166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67169 .coefficient)
      LeftBound67162.bound (LeftBound67162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67162.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority67166.bound, LeftBound67162.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67166.bound, LeftBound67162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority67166.actual selector witness, LeftBound67162.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67170

namespace LeftBound67174
def owner : Owner := ⟨.program ⟨214⟩, ⟨29373⟩⟩
def transferEvent : Nat := 67174
def frameStart : Nat := 67097
def rule : BoundRule := .product (.predecessor 0 67172 .coefficient) (.predecessor 1 67173 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67172 .coefficient)
      LeftBound67170.bound (LeftBound67170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67173 .coefficient)
      LeftAuthority67147.bound (LeftAuthority67147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67147.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67147.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67170.bound LeftAuthority67147.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67170.bound, LeftAuthority67147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67170.actual selector witness) * (LeftAuthority67147.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67174

namespace LeftBound67185
def owner : Owner := ⟨.program ⟨214⟩, ⟨16677⟩⟩
def transferEvent : Nat := 67185
def frameStart : Nat := 67097
def rule : BoundRule := .product (.predecessor 0 67183 .coefficient) (.predecessor 1 67184 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67183 .coefficient)
      LeftAuthority67158.bound (LeftAuthority67158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67158.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67184 .coefficient)
      LeftAuthority67181.bound (LeftAuthority67181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67181.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67181.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority67158.bound LeftAuthority67181.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67158.bound, LeftAuthority67181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority67158.actual selector witness) * (LeftAuthority67181.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67185

namespace LeftBound67193
def owner : Owner := ⟨.program ⟨214⟩, ⟨16678⟩⟩
def transferEvent : Nat := 67193
def frameStart : Nat := 67097
def rule : BoundRule := .sum [.predecessor 0 67191 .coefficient, .predecessor 1 67192 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67191 .coefficient)
      LeftAuthority67189.bound (LeftAuthority67189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67192 .coefficient)
      LeftBound67185.bound (LeftBound67185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67185.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67185.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority67189.bound, LeftBound67185.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67189.bound, LeftBound67185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority67189.actual selector witness, LeftBound67185.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67193

namespace LeftBound67197
def owner : Owner := ⟨.program ⟨214⟩, ⟨29377⟩⟩
def transferEvent : Nat := 67197
def frameStart : Nat := 67097
def rule : BoundRule := .sum [.predecessor 0 67195 .coefficient, .predecessor 1 67196 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67195 .coefficient)
      LeftBound67193.bound (LeftBound67193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67196 .coefficient)
      LeftBound67174.bound (LeftBound67174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67193.bound, LeftBound67174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67193.bound, LeftBound67174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67193.actual selector witness, LeftBound67174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67197

namespace LeftBound67210
def owner : Owner := ⟨.program ⟨214⟩, ⟨29375⟩⟩
def transferEvent : Nat := 67210
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67208 .coefficient, .predecessor 1 67209 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67208 .coefficient)
      LeftBound67039.bound (LeftBound67039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67039.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67209 .coefficient)
      LeftBound67022.bound (LeftBound67022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact67029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67022.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67039.bound, LeftBound67022.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67039.bound, LeftBound67022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67039.actual selector witness, LeftBound67022.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67210

namespace LeftBound67213
def owner : Owner := ⟨.program ⟨214⟩, ⟨29375⟩⟩
def transferEvent : Nat := 67213
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 67207 .summary, .result 67029 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67207 .summary)
      LeftBound67041.bound (LeftBound67041.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22407⟩⟩) (rawTerms := some (Proof.Events262.exact67207RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67029 .summary)
      LeftBound67024.bound (LeftBound67024.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29374⟩⟩) (rawTerms := some (Proof.Events261.exact67029RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67024.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67041.bound, LeftBound67024.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67041.bound, LeftBound67024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67041.actual selector witness, LeftBound67024.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67213

namespace LeftBound67237
def owner : Owner := ⟨.program ⟨214⟩, ⟨12561⟩⟩
def transferEvent : Nat := 67237
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 67235 .coefficient) (.predecessor 1 67236 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67235 .coefficient)
      LeftAuthority3177.bound (LeftAuthority3177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3177.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67236 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3177.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3177.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3177.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67237

namespace LeftBound67242
def owner : Owner := ⟨.program ⟨214⟩, ⟨7204⟩⟩
def transferEvent : Nat := 67242
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67240 .coefficient) (.predecessor 1 67241 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67240 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67241 .coefficient)
      LeftBound8475.bound (LeftBound8475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound8475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound8475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound8475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67242

namespace LeftBound67247
def owner : Owner := ⟨.program ⟨214⟩, ⟨12562⟩⟩
def transferEvent : Nat := 67247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67245 .coefficient, .predecessor 1 67246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67245 .coefficient)
      LeftBound67242.bound (LeftBound67242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67246 .coefficient)
      LeftBound67237.bound (LeftBound67237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67242.bound, LeftBound67237.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67242.bound, LeftBound67237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67242.actual selector witness, LeftBound67237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67247

namespace LeftBound67251
def owner : Owner := ⟨.program ⟨214⟩, ⟨12563⟩⟩
def transferEvent : Nat := 67251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67249 .coefficient, .predecessor 1 67250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67249 .coefficient)
      LeftBound67247.bound (LeftBound67247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67250 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67247.bound, LeftBound8467.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67247.bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67247.actual selector witness, LeftBound8467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67251

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
