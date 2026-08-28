import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard097

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound15211
def owner : Owner := ⟨.program ⟨214⟩, ⟨6771⟩⟩
def transferEvent : Nat := 15211
def frameStart : Nat := 15133
def rule : BoundRule := .identity (.predecessor 0 15210 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15210 .coefficient)
      LeftAuthority15198.bound (LeftAuthority15198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15198.derived selector witness)

def rawBound : CoeffClass := LeftAuthority15198.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority15198.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound15211

namespace LeftBound15215
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def transferEvent : Nat := 15215
def frameStart : Nat := 15133
def rule : BoundRule := .product (.predecessor 0 15213 .coefficient) (.predecessor 1 15214 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15213 .coefficient)
      LeftBound15211.bound (LeftBound15211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15214 .coefficient)
      LeftBound15208.bound (LeftBound15208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15208.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15211.bound LeftBound15208.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15211.bound, LeftBound15208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15211.actual selector witness) * (LeftBound15208.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15215

namespace LeftBound15220
def owner : Owner := ⟨.program ⟨214⟩, ⟨10595⟩⟩
def transferEvent : Nat := 15220
def frameStart : Nat := 15133
def rule : BoundRule := .sum [.predecessor 0 15218 .coefficient, .predecessor 1 15219 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15218 .coefficient)
      LeftBound15215.bound (LeftBound15215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15219 .coefficient)
      LeftBound15192.bound (LeftBound15192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15192.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15215.bound, LeftBound15192.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15215.bound, LeftBound15192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15215.actual selector witness, LeftBound15192.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15220

namespace LeftBound15224
def owner : Owner := ⟨.program ⟨214⟩, ⟨24934⟩⟩
def transferEvent : Nat := 15224
def frameStart : Nat := 15133
def rule : BoundRule := .product (.predecessor 0 15222 .coefficient) (.predecessor 1 15223 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15222 .coefficient)
      LeftBound15220.bound (LeftBound15220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15223 .coefficient)
      LeftAuthority15177.bound (LeftAuthority15177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15177.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15220.bound LeftAuthority15177.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15220.bound, LeftAuthority15177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15220.actual selector witness) * (LeftAuthority15177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15224

namespace LeftBound15235
def owner : Owner := ⟨.program ⟨214⟩, ⟨14810⟩⟩
def transferEvent : Nat := 15235
def frameStart : Nat := 15133
def rule : BoundRule := .product (.predecessor 0 15233 .coefficient) (.predecessor 1 15234 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15233 .coefficient)
      LeftAuthority15188.bound (LeftAuthority15188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15234 .coefficient)
      LeftAuthority15231.bound (LeftAuthority15231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15231.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority15188.bound LeftAuthority15231.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15188.bound, LeftAuthority15231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority15188.actual selector witness) * (LeftAuthority15231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15235

namespace LeftBound15243
def owner : Owner := ⟨.program ⟨214⟩, ⟨14811⟩⟩
def transferEvent : Nat := 15243
def frameStart : Nat := 15133
def rule : BoundRule := .sum [.predecessor 0 15241 .coefficient, .predecessor 1 15242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15241 .coefficient)
      LeftAuthority15239.bound (LeftAuthority15239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15242 .coefficient)
      LeftBound15235.bound (LeftBound15235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15235.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority15239.bound, LeftBound15235.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15239.bound, LeftBound15235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority15239.actual selector witness, LeftBound15235.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15243

namespace LeftBound15247
def owner : Owner := ⟨.program ⟨214⟩, ⟨24935⟩⟩
def transferEvent : Nat := 15247
def frameStart : Nat := 15133
def rule : BoundRule := .sum [.predecessor 0 15245 .coefficient, .predecessor 1 15246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15245 .coefficient)
      LeftBound15243.bound (LeftBound15243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15246 .coefficient)
      LeftBound15224.bound (LeftBound15224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15224.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15243.bound, LeftBound15224.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15243.bound, LeftBound15224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15243.actual selector witness, LeftBound15224.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15247

namespace LeftBound15260
def owner : Owner := ⟨.program ⟨214⟩, ⟨24933⟩⟩
def transferEvent : Nat := 15260
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15258 .coefficient, .predecessor 1 15259 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15258 .coefficient)
      LeftBound15081.bound (LeftBound15081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15259 .coefficient)
      LeftBound15064.bound (LeftBound15064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15064.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15081.bound, LeftBound15064.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15081.bound, LeftBound15064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15081.actual selector witness, LeftBound15064.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15260

namespace LeftBound15263
def owner : Owner := ⟨.program ⟨214⟩, ⟨24933⟩⟩
def transferEvent : Nat := 15263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15257 .summary, .result 15071 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15257 .summary)
      LeftBound15083.bound (LeftBound15083.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19043⟩⟩) (rawTerms := some (Proof.Events059.exact15257RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15071 .summary)
      LeftBound15066.bound (LeftBound15066.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24932⟩⟩) (rawTerms := some (Proof.Events058.exact15071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15066.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15083.bound, LeftBound15066.bound]
def bound : CoeffClass := .finite ⟨352011863863296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15083.bound, LeftBound15066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15083.actual selector witness, LeftBound15066.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15263

namespace LeftBound15267
def owner : Owner := ⟨.program ⟨214⟩, ⟨26408⟩⟩
def transferEvent : Nat := 15267
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 15265 .coefficient) (.predecessor 1 15266 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15265 .coefficient)
      LeftBound15260.bound (LeftBound15260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15266 .coefficient)
      LeftAuthority14967.bound (LeftAuthority14967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15260.bound LeftAuthority14967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15260.bound, LeftAuthority14967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15260.actual selector witness) * (LeftAuthority14967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15267

namespace LeftBound15268
def owner : Owner := ⟨.program ⟨214⟩, ⟨26408⟩⟩
def transferEvent : Nat := 15268
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩ [⟨.result 14968 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14968 .coefficient)
      LeftAuthority14967.bound (LeftAuthority14967.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26406⟩⟩) (rawTerms := some (Proof.Events058.exact14968RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14967.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14967.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14967.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound15268

namespace LeftBound15269
def owner : Owner := ⟨.program ⟨214⟩, ⟨26408⟩⟩
def transferEvent : Nat := 15269
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 15264 .summary) (.transfer 15268) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15264 .summary)
      LeftBound15263.bound (LeftBound15263.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24933⟩⟩) (rawTerms := some (Proof.Events059.exact15264RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 15268)
      LeftBound15268.bound (LeftBound15268.actual selector witness) := by
  exact .transfer (LeftBound15268.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15263.bound LeftBound15268.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15263.bound, LeftBound15268.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15263.actual selector witness) * (LeftBound15268.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15269

namespace LeftBound15280
def owner : Owner := ⟨.program ⟨214⟩, ⟨20410⟩⟩
def transferEvent : Nat := 15280
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 15278 .coefficient) (.value (.predecessor 1 15279 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15278 .coefficient)
      LeftAuthority15276.bound (LeftAuthority15276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15276.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15279 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority15276.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15276.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15276.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound15280

namespace LeftBound15284
def owner : Owner := ⟨.program ⟨214⟩, ⟨20411⟩⟩
def transferEvent : Nat := 15284
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 15282 .coefficient) (.predecessor 1 15283 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15282 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15283 .coefficient)
      LeftBound15280.bound (LeftBound15280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound15280.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound15280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound15280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15284

namespace LeftBound15285
def owner : Owner := ⟨.program ⟨214⟩, ⟨20411⟩⟩
def transferEvent : Nat := 15285
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20408⟩⟩]⟩ [⟨.result 15277 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15277 .coefficient)
      LeftAuthority15276.bound (LeftAuthority15276.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20408⟩⟩) (rawTerms := some (Proof.Events059.exact15277RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15276.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15276.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15276.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound15285

namespace LeftBound15286
def owner : Owner := ⟨.program ⟨214⟩, ⟨20411⟩⟩
def transferEvent : Nat := 15286
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 15285) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 15285)
      LeftBound15285.bound (LeftBound15285.actual selector witness) := by
  exact .transfer (LeftBound15285.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound15285.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound15285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound15285.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15286

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
