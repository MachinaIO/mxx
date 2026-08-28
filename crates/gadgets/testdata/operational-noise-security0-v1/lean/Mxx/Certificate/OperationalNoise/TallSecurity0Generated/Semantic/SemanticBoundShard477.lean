import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard476

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound70157
def owner : Owner := ⟨.program ⟨214⟩, ⟨14202⟩⟩
def transferEvent : Nat := 70157
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 70155 .coefficient) (.predecessor 1 70156 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70155 .coefficient)
      LeftAuthority3318.bound (LeftAuthority3318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70156 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3318.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3318.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3318.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70157

namespace LeftBound70162
def owner : Owner := ⟨.program ⟨214⟩, ⟨7177⟩⟩
def transferEvent : Nat := 70162
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70160 .coefficient) (.predecessor 1 70161 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70160 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70161 .coefficient)
      LeftBound11522.bound (LeftBound11522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound11522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound11522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound11522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70162

namespace LeftBound70167
def owner : Owner := ⟨.program ⟨214⟩, ⟨14203⟩⟩
def transferEvent : Nat := 70167
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70165 .coefficient, .predecessor 1 70166 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70165 .coefficient)
      LeftBound70162.bound (LeftBound70162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70166 .coefficient)
      LeftBound70157.bound (LeftBound70157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70162.bound, LeftBound70157.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70162.bound, LeftBound70157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70162.actual selector witness, LeftBound70157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70167

namespace LeftBound70171
def owner : Owner := ⟨.program ⟨214⟩, ⟨14204⟩⟩
def transferEvent : Nat := 70171
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70169 .coefficient, .predecessor 1 70170 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70169 .coefficient)
      LeftBound70167.bound (LeftBound70167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70167.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70170 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70167.bound, LeftBound11514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70167.bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70167.actual selector witness, LeftBound11514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70171

namespace LeftBound70172
def owner : Owner := ⟨.program ⟨214⟩, ⟨14204⟩⟩
def transferEvent : Nat := 70172
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩ [⟨.result 11515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11515 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨73⟩⟩) (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11514.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70172

namespace LeftBound70177
def owner : Owner := ⟨.program ⟨214⟩, ⟨14205⟩⟩
def transferEvent : Nat := 70177
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70175 .coefficient) (.predecessor 1 70176 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70175 .coefficient)
      LeftBound70171.bound (LeftBound70171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70176 .coefficient)
      LeftBound11511.bound (LeftBound11511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70171.bound LeftBound11511.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70171.bound, LeftBound11511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70171.actual selector witness) * (LeftBound11511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70177

namespace LeftBound70178
def owner : Owner := ⟨.program ⟨214⟩, ⟨14205⟩⟩
def transferEvent : Nat := 70178
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩ [⟨.result 11508 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11508 .coefficient)
      LeftAuthority11507.bound (LeftAuthority11507.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7852⟩⟩) (rawTerms := some (Proof.Events044.exact11508RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11507.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11507.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11507.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70178

namespace LeftBound70179
def owner : Owner := ⟨.program ⟨214⟩, ⟨14205⟩⟩
def transferEvent : Nat := 70179
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70174 .summary) (.transfer 70178) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70174 .summary)
      LeftBound70172.bound (LeftBound70172.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14204⟩⟩) (rawTerms := some (Proof.Events274.exact70174RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70178)
      LeftBound70178.bound (LeftBound70178.actual selector witness) := by
  exact .transfer (LeftBound70178.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70172.bound LeftBound70178.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70172.bound, LeftBound70178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70172.actual selector witness) * (LeftBound70178.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70179

namespace LeftBound70187
def owner : Owner := ⟨.program ⟨214⟩, ⟨14206⟩⟩
def transferEvent : Nat := 70187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70185 .coefficient, .predecessor 1 70186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70185 .coefficient)
      LeftBound70177.bound (LeftBound70177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70186 .coefficient)
      LeftBound70149.bound (LeftBound70149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70177.bound, LeftBound70149.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70177.bound, LeftBound70149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70177.actual selector witness, LeftBound70149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70187

namespace LeftBound70189
def owner : Owner := ⟨.program ⟨214⟩, ⟨14206⟩⟩
def transferEvent : Nat := 70189
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70184 .summary, .result 70154 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70184 .summary)
      LeftBound70179.bound (LeftBound70179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14205⟩⟩) (rawTerms := some (Proof.Events274.exact70184RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70154 .summary)
      LeftBound70151.bound (LeftBound70151.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14201⟩⟩) (rawTerms := some (Proof.Events274.exact70154RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70179.bound, LeftBound70151.bound]
def bound : CoeffClass := .finite ⟨95435392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70179.bound, LeftBound70151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70179.actual selector witness, LeftBound70151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70189

namespace LeftBound70193
def owner : Owner := ⟨.program ⟨214⟩, ⟨26062⟩⟩
def transferEvent : Nat := 70193
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70191 .coefficient) (.predecessor 1 70192 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70191 .coefficient)
      LeftBound70187.bound (LeftBound70187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70192 .coefficient)
      LeftAuthority70125.bound (LeftAuthority70125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70125.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70125.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70187.bound LeftAuthority70125.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70187.bound, LeftAuthority70125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70187.actual selector witness) * (LeftAuthority70125.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70193

namespace LeftBound70194
def owner : Owner := ⟨.program ⟨214⟩, ⟨26062⟩⟩
def transferEvent : Nat := 70194
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩ [⟨.result 70126 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70126 .coefficient)
      LeftAuthority70125.bound (LeftAuthority70125.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26061⟩⟩) (rawTerms := some (Proof.Events273.exact70126RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70125.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70125.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority70125.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70125.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70194

namespace LeftBound70195
def owner : Owner := ⟨.program ⟨214⟩, ⟨26062⟩⟩
def transferEvent : Nat := 70195
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70190 .summary) (.transfer 70194) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70190 .summary)
      LeftBound70189.bound (LeftBound70189.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14206⟩⟩) (rawTerms := some (Proof.Events274.exact70190RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70194)
      LeftBound70194.bound (LeftBound70194.actual selector witness) := by
  exact .transfer (LeftBound70194.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70189.bound LeftBound70194.bound
def bound : CoeffClass := .finite ⟨350249415606272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70189.bound, LeftBound70194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70189.actual selector witness) * (LeftBound70194.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70195

namespace LeftBound70206
def owner : Owner := ⟨.program ⟨214⟩, ⟨19526⟩⟩
def transferEvent : Nat := 70206
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 70204 .coefficient) (.value (.predecessor 1 70205 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70204 .coefficient)
      LeftAuthority70202.bound (LeftAuthority70202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70205 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority70202.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70202.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70202.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70206

namespace LeftBound70210
def owner : Owner := ⟨.program ⟨214⟩, ⟨19527⟩⟩
def transferEvent : Nat := 70210
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70208 .coefficient) (.predecessor 1 70209 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70208 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70209 .coefficient)
      LeftBound70206.bound (LeftBound70206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70206.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound70206.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound70206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound70206.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70210

namespace LeftBound70211
def owner : Owner := ⟨.program ⟨214⟩, ⟨19527⟩⟩
def transferEvent : Nat := 70211
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩ [⟨.result 70203 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70203 .coefficient)
      LeftAuthority70202.bound (LeftAuthority70202.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19524⟩⟩) (rawTerms := some (Proof.Events274.exact70203RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70202.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority70202.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70202.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70211

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
