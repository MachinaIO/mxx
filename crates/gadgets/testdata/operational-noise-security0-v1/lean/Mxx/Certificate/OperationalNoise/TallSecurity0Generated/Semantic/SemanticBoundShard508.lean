import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard507

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound74137
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def transferEvent : Nat := 74137
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 74136) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 74136)
      LeftBound74136.bound (LeftBound74136.actual selector witness) := by
  exact .transfer (LeftBound74136.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound74136.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound74136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound74136.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74137

namespace LeftBound75165
def owner : Owner := ⟨.program ⟨214⟩, ⟨15307⟩⟩
def transferEvent : Nat := 75165
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75163 .coefficient, .predecessor 1 75164 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75163 .coefficient)
      LeftAuthority75161.bound (LeftAuthority75161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75161.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75164 .coefficient)
      LeftAuthority75138.bound (LeftAuthority75138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75138.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority75161.bound, LeftAuthority75138.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75161.bound, LeftAuthority75138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority75161.actual selector witness, LeftAuthority75138.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75165

namespace LeftBound75169
def owner : Owner := ⟨.program ⟨214⟩, ⟨15363⟩⟩
def transferEvent : Nat := 75169
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75167 .coefficient, .predecessor 1 75168 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75167 .coefficient)
      LeftBound75165.bound (LeftBound75165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75168 .coefficient)
      LeftAuthority75115.bound (LeftAuthority75115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75115.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75165.bound, LeftAuthority75115.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75165.bound, LeftAuthority75115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75165.actual selector witness, LeftAuthority75115.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75169

namespace LeftBound75173
def owner : Owner := ⟨.program ⟨214⟩, ⟨17319⟩⟩
def transferEvent : Nat := 75173
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75171 .coefficient, .predecessor 1 75172 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75171 .coefficient)
      LeftBound75169.bound (LeftBound75169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75169.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75169.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75172 .coefficient)
      LeftAuthority75092.bound (LeftAuthority75092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75092.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75169.bound, LeftAuthority75092.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75169.bound, LeftAuthority75092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75169.actual selector witness, LeftAuthority75092.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75173

namespace LeftBound75177
def owner : Owner := ⟨.program ⟨214⟩, ⟨17320⟩⟩
def transferEvent : Nat := 75177
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75175 .coefficient, .predecessor 1 75176 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75175 .coefficient)
      LeftBound75173.bound (LeftBound75173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75176 .coefficient)
      LeftAuthority75069.bound (LeftAuthority75069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75069.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75069.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75173.bound, LeftAuthority75069.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75173.bound, LeftAuthority75069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75173.actual selector witness, LeftAuthority75069.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75177

namespace LeftBound75181
def owner : Owner := ⟨.program ⟨214⟩, ⟨17321⟩⟩
def transferEvent : Nat := 75181
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75179 .coefficient, .predecessor 1 75180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75179 .coefficient)
      LeftBound75177.bound (LeftBound75177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75180 .coefficient)
      LeftAuthority75046.bound (LeftAuthority75046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75046.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75046.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75177.bound, LeftAuthority75046.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75177.bound, LeftAuthority75046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75177.actual selector witness, LeftAuthority75046.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75181

namespace LeftBound75185
def owner : Owner := ⟨.program ⟨214⟩, ⟨17322⟩⟩
def transferEvent : Nat := 75185
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75183 .coefficient, .predecessor 1 75184 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75183 .coefficient)
      LeftBound75181.bound (LeftBound75181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75184 .coefficient)
      LeftAuthority75023.bound (LeftAuthority75023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75023.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75181.bound, LeftAuthority75023.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75181.bound, LeftAuthority75023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75181.actual selector witness, LeftAuthority75023.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75185

namespace LeftBound75189
def owner : Owner := ⟨.program ⟨214⟩, ⟨17323⟩⟩
def transferEvent : Nat := 75189
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75187 .coefficient, .predecessor 1 75188 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75187 .coefficient)
      LeftBound75185.bound (LeftBound75185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75185.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75185.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75188 .coefficient)
      LeftAuthority75000.bound (LeftAuthority75000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact75001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75000.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75000.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75185.bound, LeftAuthority75000.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75185.bound, LeftAuthority75000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75185.actual selector witness, LeftAuthority75000.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75189

namespace LeftBound75193
def owner : Owner := ⟨.program ⟨214⟩, ⟨17324⟩⟩
def transferEvent : Nat := 75193
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75191 .coefficient, .predecessor 1 75192 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75191 .coefficient)
      LeftBound75189.bound (LeftBound75189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75192 .coefficient)
      LeftAuthority74977.bound (LeftAuthority74977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75189.bound, LeftAuthority74977.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75189.bound, LeftAuthority74977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75189.actual selector witness, LeftAuthority74977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75193

namespace LeftBound75197
def owner : Owner := ⟨.program ⟨214⟩, ⟨18328⟩⟩
def transferEvent : Nat := 75197
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75195 .coefficient, .predecessor 1 75196 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75195 .coefficient)
      LeftBound75193.bound (LeftBound75193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75196 .coefficient)
      LeftAuthority74954.bound (LeftAuthority74954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74954.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75193.bound, LeftAuthority74954.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75193.bound, LeftAuthority74954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75193.actual selector witness, LeftAuthority74954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75197

namespace LeftBound75201
def owner : Owner := ⟨.program ⟨214⟩, ⟨18329⟩⟩
def transferEvent : Nat := 75201
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75199 .coefficient, .predecessor 1 75200 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75199 .coefficient)
      LeftBound75197.bound (LeftBound75197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75200 .coefficient)
      LeftAuthority74931.bound (LeftAuthority74931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74931.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75197.bound, LeftAuthority74931.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75197.bound, LeftAuthority74931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75197.actual selector witness, LeftAuthority74931.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75201

namespace LeftBound75205
def owner : Owner := ⟨.program ⟨214⟩, ⟨18330⟩⟩
def transferEvent : Nat := 75205
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75203 .coefficient, .predecessor 1 75204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75203 .coefficient)
      LeftBound75201.bound (LeftBound75201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75204 .coefficient)
      LeftAuthority74908.bound (LeftAuthority74908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75201.bound, LeftAuthority74908.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75201.bound, LeftAuthority74908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75201.actual selector witness, LeftAuthority74908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75205

namespace LeftBound75209
def owner : Owner := ⟨.program ⟨214⟩, ⟨18331⟩⟩
def transferEvent : Nat := 75209
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75207 .coefficient, .predecessor 1 75208 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75207 .coefficient)
      LeftBound75205.bound (LeftBound75205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75205.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75208 .coefficient)
      LeftAuthority74885.bound (LeftAuthority74885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74885.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74885.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75205.bound, LeftAuthority74885.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75205.bound, LeftAuthority74885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75205.actual selector witness, LeftAuthority74885.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75209

namespace LeftBound75213
def owner : Owner := ⟨.program ⟨214⟩, ⟨18332⟩⟩
def transferEvent : Nat := 75213
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75211 .coefficient, .predecessor 1 75212 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75211 .coefficient)
      LeftBound75209.bound (LeftBound75209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75212 .coefficient)
      LeftAuthority74862.bound (LeftAuthority74862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75209.bound, LeftAuthority74862.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75209.bound, LeftAuthority74862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75209.actual selector witness, LeftAuthority74862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75213

namespace LeftBound75217
def owner : Owner := ⟨.program ⟨214⟩, ⟨18333⟩⟩
def transferEvent : Nat := 75217
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75215 .coefficient, .predecessor 1 75216 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75215 .coefficient)
      LeftBound75213.bound (LeftBound75213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75216 .coefficient)
      LeftAuthority74839.bound (LeftAuthority74839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74839.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75213.bound, LeftAuthority74839.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75213.bound, LeftAuthority74839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75213.actual selector witness, LeftAuthority74839.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75217

namespace LeftBound75221
def owner : Owner := ⟨.program ⟨214⟩, ⟨18334⟩⟩
def transferEvent : Nat := 75221
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75219 .coefficient, .predecessor 1 75220 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75219 .coefficient)
      LeftBound75217.bound (LeftBound75217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75217.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75220 .coefficient)
      LeftAuthority74816.bound (LeftAuthority74816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events292.exact74817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74816.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75217.bound, LeftAuthority74816.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75217.bound, LeftAuthority74816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75217.actual selector witness, LeftAuthority74816.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75221

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
