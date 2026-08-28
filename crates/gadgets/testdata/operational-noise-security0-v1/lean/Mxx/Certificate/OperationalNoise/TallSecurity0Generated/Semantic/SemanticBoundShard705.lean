import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard687
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard690
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard694
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard698
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard701
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard704

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound102129
def owner : Owner := ⟨.program ⟨214⟩, ⟨14827⟩⟩
def transferEvent : Nat := 102129
def frameStart : Nat := 102068
def rule : BoundRule := .sum [.predecessor 0 102127 .coefficient, .predecessor 1 102128 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102127 .coefficient)
      LeftAuthority102125.bound (LeftAuthority102125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102125.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102128 .coefficient)
      LeftBound102121.bound (LeftBound102121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority102125.bound, LeftBound102121.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102125.bound, LeftBound102121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority102125.actual selector witness, LeftBound102121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102129

namespace LeftBound102133
def owner : Owner := ⟨.program ⟨214⟩, ⟨26327⟩⟩
def transferEvent : Nat := 102133
def frameStart : Nat := 102068
def rule : BoundRule := .product (.predecessor 0 102131 .coefficient) (.predecessor 1 102132 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102131 .coefficient)
      LeftBound102129.bound (LeftBound102129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102132 .coefficient)
      LeftAuthority102106.bound (LeftAuthority102106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102106.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound102129.bound LeftAuthority102106.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102129.bound, LeftAuthority102106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound102129.actual selector witness) * (LeftAuthority102106.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102133

namespace LeftBound102144
def owner : Owner := ⟨.program ⟨214⟩, ⟨15259⟩⟩
def transferEvent : Nat := 102144
def frameStart : Nat := 102068
def rule : BoundRule := .product (.predecessor 0 102142 .coefficient) (.predecessor 1 102143 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102142 .coefficient)
      LeftAuthority102117.bound (LeftAuthority102117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102117.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102143 .coefficient)
      LeftAuthority102140.bound (LeftAuthority102140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102140.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102140.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority102117.bound LeftAuthority102140.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102117.bound, LeftAuthority102140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority102117.actual selector witness) * (LeftAuthority102140.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102144

namespace LeftBound102152
def owner : Owner := ⟨.program ⟨214⟩, ⟨15260⟩⟩
def transferEvent : Nat := 102152
def frameStart : Nat := 102068
def rule : BoundRule := .sum [.predecessor 0 102150 .coefficient, .predecessor 1 102151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102150 .coefficient)
      LeftAuthority102148.bound (LeftAuthority102148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102148.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102151 .coefficient)
      LeftBound102144.bound (LeftBound102144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102144.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102144.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority102148.bound, LeftBound102144.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102148.bound, LeftBound102144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority102148.actual selector witness, LeftBound102144.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102152

namespace LeftBound102156
def owner : Owner := ⟨.program ⟨214⟩, ⟨26330⟩⟩
def transferEvent : Nat := 102156
def frameStart : Nat := 102068
def rule : BoundRule := .sum [.predecessor 0 102154 .coefficient, .predecessor 1 102155 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102154 .coefficient)
      LeftBound102152.bound (LeftBound102152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102155 .coefficient)
      LeftBound102133.bound (LeftBound102133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102133.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102152.bound, LeftBound102133.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102152.bound, LeftBound102133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102152.actual selector witness, LeftBound102133.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102156

namespace LeftBound102169
def owner : Owner := ⟨.program ⟨214⟩, ⟨26329⟩⟩
def transferEvent : Nat := 102169
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102167 .coefficient, .predecessor 1 102168 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102167 .coefficient)
      LeftBound102022.bound (LeftBound102022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102168 .coefficient)
      LeftBound102005.bound (LeftBound102005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102022.bound, LeftBound102005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102022.bound, LeftBound102005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102022.actual selector witness, LeftBound102005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102169

namespace LeftBound102172
def owner : Owner := ⟨.program ⟨214⟩, ⟨26329⟩⟩
def transferEvent : Nat := 102172
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102166 .summary, .result 102012 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102166 .summary)
      LeftBound102024.bound (LeftBound102024.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20384⟩⟩) (rawTerms := some (Proof.Events399.exact102166RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102012 .summary)
      LeftBound102007.bound (LeftBound102007.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26328⟩⟩) (rawTerms := some (Proof.Events398.exact102012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102024.bound, LeftBound102007.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102024.bound, LeftBound102007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102024.actual selector witness, LeftBound102007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102172

namespace LeftBound102176
def owner : Owner := ⟨.program ⟨214⟩, ⟨26533⟩⟩
def transferEvent : Nat := 102176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102174 .coefficient, .predecessor 1 102175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102174 .coefficient)
      LeftBound102169.bound (LeftBound102169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102169.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102169.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102175 .coefficient)
      LeftBound101735.bound (LeftBound101735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102169.bound, LeftBound101735.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102169.bound, LeftBound101735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102169.actual selector witness, LeftBound101735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102176

namespace LeftBound102177
def owner : Owner := ⟨.program ⟨214⟩, ⟨26533⟩⟩
def transferEvent : Nat := 102177
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102173 .summary, .result 101739 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102173 .summary)
      LeftBound102172.bound (LeftBound102172.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26329⟩⟩) (rawTerms := some (Proof.Events399.exact102173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101739 .summary)
      LeftBound101738.bound (LeftBound101738.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26532⟩⟩) (rawTerms := some (Proof.Events397.exact101739RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101738.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102172.bound, LeftBound101738.bound]
def bound : CoeffClass := .finite ⟨2583789554981353578496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102172.bound, LeftBound101738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102172.actual selector witness, LeftBound101738.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102177

namespace LeftBound102181
def owner : Owner := ⟨.program ⟨214⟩, ⟨26750⟩⟩
def transferEvent : Nat := 102181
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102179 .coefficient, .predecessor 1 102180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102179 .coefficient)
      LeftBound102176.bound (LeftBound102176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102180 .coefficient)
      LeftBound101301.bound (LeftBound101301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101301.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102176.bound, LeftBound101301.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102176.bound, LeftBound101301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102176.actual selector witness, LeftBound101301.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102181

namespace LeftBound102182
def owner : Owner := ⟨.program ⟨214⟩, ⟨26750⟩⟩
def transferEvent : Nat := 102182
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102178 .summary, .result 101305 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102178 .summary)
      LeftBound102177.bound (LeftBound102177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26533⟩⟩) (rawTerms := some (Proof.Events399.exact102178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101305 .summary)
      LeftBound101304.bound (LeftBound101304.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26749⟩⟩) (rawTerms := some (Proof.Events395.exact101305RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101304.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102177.bound, LeftBound101304.bound]
def bound : CoeffClass := .finite ⟨3875701141805795807232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102177.bound, LeftBound101304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102177.actual selector witness, LeftBound101304.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102182

namespace LeftBound102186
def owner : Owner := ⟨.program ⟨214⟩, ⟨26967⟩⟩
def transferEvent : Nat := 102186
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102184 .coefficient, .predecessor 1 102185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102184 .coefficient)
      LeftBound102181.bound (LeftBound102181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102185 .coefficient)
      LeftBound100867.bound (LeftBound100867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100867.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102181.bound, LeftBound100867.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102181.bound, LeftBound100867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102181.actual selector witness, LeftBound100867.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102186

namespace LeftBound102187
def owner : Owner := ⟨.program ⟨214⟩, ⟨26967⟩⟩
def transferEvent : Nat := 102187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102183 .summary, .result 100871 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102183 .summary)
      LeftBound102182.bound (LeftBound102182.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26750⟩⟩) (rawTerms := some (Proof.Events399.exact102183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100871 .summary)
      LeftBound100870.bound (LeftBound100870.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26966⟩⟩) (rawTerms := some (Proof.Events394.exact100871RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100870.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102182.bound, LeftBound100870.bound]
def bound : CoeffClass := .finite ⟨5167635141075258621952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102182.bound, LeftBound100870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102182.actual selector witness, LeftBound100870.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102187

namespace LeftBound102191
def owner : Owner := ⟨.program ⟨214⟩, ⟨27184⟩⟩
def transferEvent : Nat := 102191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102189 .coefficient, .predecessor 1 102190 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102189 .coefficient)
      LeftBound102186.bound (LeftBound102186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102190 .coefficient)
      LeftBound100433.bound (LeftBound100433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100433.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102186.bound, LeftBound100433.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102186.bound, LeftBound100433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102186.actual selector witness, LeftBound100433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102191

namespace LeftBound102192
def owner : Owner := ⟨.program ⟨214⟩, ⟨27184⟩⟩
def transferEvent : Nat := 102192
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102188 .summary, .result 100437 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102188 .summary)
      LeftBound102187.bound (LeftBound102187.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26967⟩⟩) (rawTerms := some (Proof.Events399.exact102188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100437 .summary)
      LeftBound100436.bound (LeftBound100436.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27183⟩⟩) (rawTerms := some (Proof.Events392.exact100437RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102187.bound, LeftBound100436.bound]
def bound : CoeffClass := .finite ⟨6459613965234762608640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102187.bound, LeftBound100436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102187.actual selector witness, LeftBound100436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102192

namespace LeftBound102196
def owner : Owner := ⟨.program ⟨214⟩, ⟨27401⟩⟩
def transferEvent : Nat := 102196
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102194 .coefficient, .predecessor 1 102195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102194 .coefficient)
      LeftBound102191.bound (LeftBound102191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102195 .coefficient)
      LeftBound99999.bound (LeftBound99999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102191.bound, LeftBound99999.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102191.bound, LeftBound99999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102191.actual selector witness, LeftBound99999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102196

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
