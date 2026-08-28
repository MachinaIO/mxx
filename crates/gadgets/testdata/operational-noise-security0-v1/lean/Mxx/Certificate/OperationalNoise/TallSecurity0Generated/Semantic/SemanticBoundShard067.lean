import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard066

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound11255
def owner : Owner := ⟨.program ⟨214⟩, ⟨26165⟩⟩
def transferEvent : Nat := 11255
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 11249 .summary, .result 11063 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11249 .summary)
      LeftBound11075.bound (LeftBound11075.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19619⟩⟩) (rawTerms := some (Proof.Events043.exact11249RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11063 .summary)
      LeftBound11058.bound (LeftBound11058.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26164⟩⟩) (rawTerms := some (Proof.Events043.exact11063RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11075.bound, LeftBound11058.bound]
def bound : CoeffClass := .finite ⟨352072932929536, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11075.bound, LeftBound11058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11075.actual selector witness, LeftBound11058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11255

namespace LeftBound11259
def owner : Owner := ⟨.program ⟨214⟩, ⟨28137⟩⟩
def transferEvent : Nat := 11259
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11257 .coefficient) (.predecessor 1 11258 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11257 .coefficient)
      LeftBound11252.bound (LeftBound11252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11258 .coefficient)
      LeftAuthority10959.bound (LeftAuthority10959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10959.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11252.bound LeftAuthority10959.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11252.bound, LeftAuthority10959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11252.actual selector witness) * (LeftAuthority10959.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11259

namespace LeftBound11260
def owner : Owner := ⟨.program ⟨214⟩, ⟨28137⟩⟩
def transferEvent : Nat := 11260
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28135⟩⟩]⟩ [⟨.result 10960 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10960 .coefficient)
      LeftAuthority10959.bound (LeftAuthority10959.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28135⟩⟩) (rawTerms := some (Proof.Events042.exact10960RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10959.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10959.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10959.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound11260

namespace LeftBound11261
def owner : Owner := ⟨.program ⟨214⟩, ⟨28137⟩⟩
def transferEvent : Nat := 11261
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 11256 .summary) (.transfer 11260) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11256 .summary)
      LeftBound11255.bound (LeftBound11255.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26165⟩⟩) (rawTerms := some (Proof.Events043.exact11256RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 11260)
      LeftBound11260.bound (LeftBound11260.actual selector witness) := by
  exact .transfer (LeftBound11260.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11255.bound LeftBound11260.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11255.bound, LeftBound11260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11255.actual selector witness) * (LeftBound11260.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11261

namespace LeftBound11272
def owner : Owner := ⟨.program ⟨214⟩, ⟨21562⟩⟩
def transferEvent : Nat := 11272
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 11270 .coefficient) (.value (.predecessor 1 11271 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11270 .coefficient)
      LeftAuthority11268.bound (LeftAuthority11268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11268.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11271 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority11268.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11268.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11268.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound11272

namespace LeftBound11276
def owner : Owner := ⟨.program ⟨214⟩, ⟨21563⟩⟩
def transferEvent : Nat := 11276
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11274 .coefficient) (.predecessor 1 11275 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11274 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11275 .coefficient)
      LeftBound11272.bound (LeftBound11272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11272.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound11272.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound11272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound11272.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11276

namespace LeftBound11277
def owner : Owner := ⟨.program ⟨214⟩, ⟨21563⟩⟩
def transferEvent : Nat := 11277
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21560⟩⟩]⟩ [⟨.result 11269 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11269 .coefficient)
      LeftAuthority11268.bound (LeftAuthority11268.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21560⟩⟩) (rawTerms := some (Proof.Events044.exact11269RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11268.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11268.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11268.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11268.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11268.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound11277

namespace LeftBound11278
def owner : Owner := ⟨.program ⟨214⟩, ⟨21563⟩⟩
def transferEvent : Nat := 11278
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 11277) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 11277)
      LeftBound11277.bound (LeftBound11277.actual selector witness) := by
  exact .transfer (LeftBound11277.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound11277.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound11277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound11277.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11278

namespace LeftBound11373
def owner : Owner := ⟨.program ⟨214⟩, ⟨16076⟩⟩
def transferEvent : Nat := 11373
def frameStart : Nat := 11334
def rule : BoundRule := .identity (.predecessor 0 11372 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11372 .coefficient)
      LeftAuthority11370.bound (LeftAuthority11370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11370.derived selector witness)

def rawBound : CoeffClass := LeftAuthority11370.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority11370.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11373

namespace LeftBound11390
def owner : Owner := ⟨.program ⟨214⟩, ⟨16150⟩⟩
def transferEvent : Nat := 11390
def frameStart : Nat := 11334
def rule : BoundRule := .sum [.predecessor 0 11388 .coefficient, .predecessor 1 11389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11388 .coefficient)
      LeftBound11373.bound (LeftBound11373.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound11373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11389 .coefficient)
      LeftAuthority11386.bound (LeftAuthority11386.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority11386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11373.bound, LeftAuthority11386.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11373.bound, LeftAuthority11386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11373.actual selector witness, LeftAuthority11386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11390

namespace LeftBound11393
def owner : Owner := ⟨.program ⟨214⟩, ⟨16151⟩⟩
def transferEvent : Nat := 11393
def frameStart : Nat := 11334
def rule : BoundRule := .identity (.predecessor 0 11392 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11392 .coefficient)
      LeftBound11390.bound (LeftBound11390.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound11390.derived selector witness)

def rawBound : CoeffClass := LeftBound11390.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound11390.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11393

namespace LeftBound11399
def owner : Owner := ⟨.program ⟨214⟩, ⟨16152⟩⟩
def transferEvent : Nat := 11399
def frameStart : Nat := 11334
def rule : BoundRule := .product (.predecessor 0 11397 .coefficient) (.predecessor 1 11398 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11397 .coefficient)
      LeftAuthority11395.bound (LeftAuthority11395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11398 .coefficient)
      LeftBound11393.bound (LeftBound11393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11393.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority11395.bound LeftBound11393.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11395.bound, LeftBound11393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority11395.actual selector witness) * (LeftBound11393.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11399

namespace LeftBound11407
def owner : Owner := ⟨.program ⟨214⟩, ⟨16153⟩⟩
def transferEvent : Nat := 11407
def frameStart : Nat := 11334
def rule : BoundRule := .sum [.predecessor 0 11405 .coefficient, .predecessor 1 11406 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11405 .coefficient)
      LeftAuthority11403.bound (LeftAuthority11403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11403.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11406 .coefficient)
      LeftBound11399.bound (LeftBound11399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11399.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11399.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority11403.bound, LeftBound11399.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11403.bound, LeftBound11399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority11403.actual selector witness, LeftBound11399.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11407

namespace LeftBound11411
def owner : Owner := ⟨.program ⟨214⟩, ⟨28136⟩⟩
def transferEvent : Nat := 11411
def frameStart : Nat := 11334
def rule : BoundRule := .product (.predecessor 0 11409 .coefficient) (.predecessor 1 11410 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11409 .coefficient)
      LeftBound11407.bound (LeftBound11407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11410 .coefficient)
      LeftAuthority11384.bound (LeftAuthority11384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11384.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11407.bound LeftAuthority11384.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11407.bound, LeftAuthority11384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11407.actual selector witness) * (LeftAuthority11384.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11411

namespace LeftBound11422
def owner : Owner := ⟨.program ⟨214⟩, ⟨16118⟩⟩
def transferEvent : Nat := 11422
def frameStart : Nat := 11334
def rule : BoundRule := .product (.predecessor 0 11420 .coefficient) (.predecessor 1 11421 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11420 .coefficient)
      LeftAuthority11395.bound (LeftAuthority11395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11421 .coefficient)
      LeftAuthority11418.bound (LeftAuthority11418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11418.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11395.bound LeftAuthority11418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11395.bound, LeftAuthority11418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority11395.actual selector witness) * (LeftAuthority11418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11422

namespace LeftBound11430
def owner : Owner := ⟨.program ⟨214⟩, ⟨16119⟩⟩
def transferEvent : Nat := 11430
def frameStart : Nat := 11334
def rule : BoundRule := .sum [.predecessor 0 11428 .coefficient, .predecessor 1 11429 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11428 .coefficient)
      LeftAuthority11426.bound (LeftAuthority11426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11426.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11429 .coefficient)
      LeftBound11422.bound (LeftBound11422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11422.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11422.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority11426.bound, LeftBound11422.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11426.bound, LeftBound11422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority11426.actual selector witness, LeftBound11422.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11430

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
