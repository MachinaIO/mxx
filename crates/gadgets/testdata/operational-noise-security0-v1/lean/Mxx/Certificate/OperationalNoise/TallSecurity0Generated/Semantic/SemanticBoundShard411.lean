import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard344
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard410

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound61190
def owner : Owner := ⟨.program ⟨214⟩, ⟨29827⟩⟩
def transferEvent : Nat := 61190
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29825⟩⟩]⟩ [⟨.result 61186 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61186 .coefficient)
      LeftAuthority61185.bound (LeftAuthority61185.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29825⟩⟩) (rawTerms := some (Proof.Events239.exact61186RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61185.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61185.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority61185.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61185.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61190

namespace LeftBound61191
def owner : Owner := ⟨.program ⟨214⟩, ⟨29827⟩⟩
def transferEvent : Nat := 61191
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51430 .summary) (.transfer 61190) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51430 .summary)
      LeftBound51429.bound (LeftBound51429.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25688⟩⟩) (rawTerms := some (Proof.Events200.exact51430RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51429.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61190)
      LeftBound61190.bound (LeftBound61190.actual selector witness) := by
  exact .transfer (LeftBound61190.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51429.bound LeftBound61190.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51429.bound, LeftBound61190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51429.actual selector witness) * (LeftBound61190.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61191

namespace LeftBound61202
def owner : Owner := ⟨.program ⟨214⟩, ⟨22630⟩⟩
def transferEvent : Nat := 61202
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 61200 .coefficient) (.value (.predecessor 1 61201 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61200 .coefficient)
      LeftAuthority61198.bound (LeftAuthority61198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61201 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority61198.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61198.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61198.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound61202

namespace LeftBound61206
def owner : Owner := ⟨.program ⟨214⟩, ⟨22631⟩⟩
def transferEvent : Nat := 61206
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61204 .coefficient) (.predecessor 1 61205 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61204 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61205 .coefficient)
      LeftBound61202.bound (LeftBound61202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61202.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound61202.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound61202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound61202.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61206

namespace LeftBound61207
def owner : Owner := ⟨.program ⟨214⟩, ⟨22631⟩⟩
def transferEvent : Nat := 61207
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22628⟩⟩]⟩ [⟨.result 61199 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61199 .coefficient)
      LeftAuthority61198.bound (LeftAuthority61198.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22628⟩⟩) (rawTerms := some (Proof.Events239.exact61199RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61198.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority61198.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61198.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61207

namespace LeftBound61208
def owner : Owner := ⟨.program ⟨214⟩, ⟨22631⟩⟩
def transferEvent : Nat := 61208
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 61207) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61207)
      LeftBound61207.bound (LeftBound61207.actual selector witness) := by
  exact .transfer (LeftBound61207.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound61207.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound61207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound61207.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61208

namespace LeftBound61303
def owner : Owner := ⟨.program ⟨214⟩, ⟨16876⟩⟩
def transferEvent : Nat := 61303
def frameStart : Nat := 61264
def rule : BoundRule := .identity (.predecessor 0 61302 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61302 .coefficient)
      LeftAuthority61300.bound (LeftAuthority61300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61300.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61300.derived selector witness)

def rawBound : CoeffClass := LeftAuthority61300.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority61300.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61303

namespace LeftBound61320
def owner : Owner := ⟨.program ⟨214⟩, ⟨16971⟩⟩
def transferEvent : Nat := 61320
def frameStart : Nat := 61264
def rule : BoundRule := .sum [.predecessor 0 61318 .coefficient, .predecessor 1 61319 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61318 .coefficient)
      LeftBound61303.bound (LeftBound61303.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61319 .coefficient)
      LeftAuthority61316.bound (LeftAuthority61316.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority61316.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61303.bound, LeftAuthority61316.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61303.bound, LeftAuthority61316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61303.actual selector witness, LeftAuthority61316.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61320

namespace LeftBound61323
def owner : Owner := ⟨.program ⟨214⟩, ⟨16972⟩⟩
def transferEvent : Nat := 61323
def frameStart : Nat := 61264
def rule : BoundRule := .identity (.predecessor 0 61322 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61322 .coefficient)
      LeftBound61320.bound (LeftBound61320.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61320.derived selector witness)

def rawBound : CoeffClass := LeftBound61320.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound61320.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61323

namespace LeftBound61329
def owner : Owner := ⟨.program ⟨214⟩, ⟨16973⟩⟩
def transferEvent : Nat := 61329
def frameStart : Nat := 61264
def rule : BoundRule := .product (.predecessor 0 61327 .coefficient) (.predecessor 1 61328 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61327 .coefficient)
      LeftAuthority61325.bound (LeftAuthority61325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61328 .coefficient)
      LeftBound61323.bound (LeftBound61323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61323.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority61325.bound LeftBound61323.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61325.bound, LeftBound61323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority61325.actual selector witness) * (LeftBound61323.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61329

namespace LeftBound61337
def owner : Owner := ⟨.program ⟨214⟩, ⟨16974⟩⟩
def transferEvent : Nat := 61337
def frameStart : Nat := 61264
def rule : BoundRule := .sum [.predecessor 0 61335 .coefficient, .predecessor 1 61336 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61335 .coefficient)
      LeftAuthority61333.bound (LeftAuthority61333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61333.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61336 .coefficient)
      LeftBound61329.bound (LeftBound61329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61329.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61333.bound, LeftBound61329.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61333.bound, LeftBound61329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61333.actual selector witness, LeftBound61329.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61337

namespace LeftBound61341
def owner : Owner := ⟨.program ⟨214⟩, ⟨29826⟩⟩
def transferEvent : Nat := 61341
def frameStart : Nat := 61264
def rule : BoundRule := .product (.predecessor 0 61339 .coefficient) (.predecessor 1 61340 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61339 .coefficient)
      LeftBound61337.bound (LeftBound61337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61340 .coefficient)
      LeftAuthority61314.bound (LeftAuthority61314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61314.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61314.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61337.bound LeftAuthority61314.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61337.bound, LeftAuthority61314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61337.actual selector witness) * (LeftAuthority61314.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61341

namespace LeftBound61352
def owner : Owner := ⟨.program ⟨214⟩, ⟨16933⟩⟩
def transferEvent : Nat := 61352
def frameStart : Nat := 61264
def rule : BoundRule := .product (.predecessor 0 61350 .coefficient) (.predecessor 1 61351 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61350 .coefficient)
      LeftAuthority61325.bound (LeftAuthority61325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61351 .coefficient)
      LeftAuthority61348.bound (LeftAuthority61348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61348.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority61325.bound LeftAuthority61348.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61325.bound, LeftAuthority61348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority61325.actual selector witness) * (LeftAuthority61348.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61352

namespace LeftBound61360
def owner : Owner := ⟨.program ⟨214⟩, ⟨16934⟩⟩
def transferEvent : Nat := 61360
def frameStart : Nat := 61264
def rule : BoundRule := .sum [.predecessor 0 61358 .coefficient, .predecessor 1 61359 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61358 .coefficient)
      LeftAuthority61356.bound (LeftAuthority61356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61356.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61359 .coefficient)
      LeftBound61352.bound (LeftBound61352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61352.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61356.bound, LeftBound61352.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61356.bound, LeftBound61352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61356.actual selector witness, LeftBound61352.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61360

namespace LeftBound61364
def owner : Owner := ⟨.program ⟨214⟩, ⟨29831⟩⟩
def transferEvent : Nat := 61364
def frameStart : Nat := 61264
def rule : BoundRule := .sum [.predecessor 0 61362 .coefficient, .predecessor 1 61363 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61362 .coefficient)
      LeftBound61360.bound (LeftBound61360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61363 .coefficient)
      LeftBound61341.bound (LeftBound61341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61341.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61360.bound, LeftBound61341.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61360.bound, LeftBound61341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61360.actual selector witness, LeftBound61341.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61364

namespace LeftBound61377
def owner : Owner := ⟨.program ⟨214⟩, ⟨29828⟩⟩
def transferEvent : Nat := 61377
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 61375 .coefficient, .predecessor 1 61376 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61375 .coefficient)
      LeftBound61206.bound (LeftBound61206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61376 .coefficient)
      LeftBound61189.bound (LeftBound61189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61189.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61206.bound, LeftBound61189.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61206.bound, LeftBound61189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61206.actual selector witness, LeftBound61189.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61377

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
