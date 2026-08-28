import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard617

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound91056
def owner : Owner := ⟨.program ⟨214⟩, ⟨22195⟩⟩
def transferEvent : Nat := 91056
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91054 .coefficient) (.predecessor 1 91055 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91054 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91055 .coefficient)
      LeftBound91052.bound (LeftBound91052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91052.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound91052.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound91052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound91052.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91056

namespace LeftBound91057
def owner : Owner := ⟨.program ⟨214⟩, ⟨22195⟩⟩
def transferEvent : Nat := 91057
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22192⟩⟩]⟩ [⟨.result 91049 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91049 .coefficient)
      LeftAuthority91048.bound (LeftAuthority91048.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22192⟩⟩) (rawTerms := some (Proof.Events355.exact91049RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91048.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91048.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91048.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91057

namespace LeftBound91058
def owner : Owner := ⟨.program ⟨214⟩, ⟨22195⟩⟩
def transferEvent : Nat := 91058
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 91057) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91057)
      LeftBound91057.bound (LeftBound91057.actual selector witness) := by
  exact .transfer (LeftBound91057.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound91057.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound91057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound91057.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91058

namespace LeftBound91153
def owner : Owner := ⟨.program ⟨214⟩, ⟨16550⟩⟩
def transferEvent : Nat := 91153
def frameStart : Nat := 91114
def rule : BoundRule := .identity (.predecessor 0 91152 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91152 .coefficient)
      LeftAuthority91150.bound (LeftAuthority91150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91150.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91150.derived selector witness)

def rawBound : CoeffClass := LeftAuthority91150.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority91150.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound91153

namespace LeftBound91170
def owner : Owner := ⟨.program ⟨214⟩, ⟨16589⟩⟩
def transferEvent : Nat := 91170
def frameStart : Nat := 91114
def rule : BoundRule := .sum [.predecessor 0 91168 .coefficient, .predecessor 1 91169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91168 .coefficient)
      LeftBound91153.bound (LeftBound91153.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound91153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91169 .coefficient)
      LeftAuthority91166.bound (LeftAuthority91166.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority91166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91153.bound, LeftAuthority91166.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91153.bound, LeftAuthority91166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91153.actual selector witness, LeftAuthority91166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91170

namespace LeftBound91173
def owner : Owner := ⟨.program ⟨214⟩, ⟨16590⟩⟩
def transferEvent : Nat := 91173
def frameStart : Nat := 91114
def rule : BoundRule := .identity (.predecessor 0 91172 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91172 .coefficient)
      LeftBound91170.bound (LeftBound91170.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound91170.derived selector witness)

def rawBound : CoeffClass := LeftBound91170.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound91170.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound91173

namespace LeftBound91179
def owner : Owner := ⟨.program ⟨214⟩, ⟨16591⟩⟩
def transferEvent : Nat := 91179
def frameStart : Nat := 91114
def rule : BoundRule := .product (.predecessor 0 91177 .coefficient) (.predecessor 1 91178 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91177 .coefficient)
      LeftAuthority91175.bound (LeftAuthority91175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91178 .coefficient)
      LeftBound91173.bound (LeftBound91173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority91175.bound LeftBound91173.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91175.bound, LeftBound91173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority91175.actual selector witness) * (LeftBound91173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91179

namespace LeftBound91187
def owner : Owner := ⟨.program ⟨214⟩, ⟨16592⟩⟩
def transferEvent : Nat := 91187
def frameStart : Nat := 91114
def rule : BoundRule := .sum [.predecessor 0 91185 .coefficient, .predecessor 1 91186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91185 .coefficient)
      LeftAuthority91183.bound (LeftAuthority91183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91183.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91183.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91186 .coefficient)
      LeftBound91179.bound (LeftBound91179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91183.bound, LeftBound91179.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91183.bound, LeftBound91179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority91183.actual selector witness, LeftBound91179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91187

namespace LeftBound91191
def owner : Owner := ⟨.program ⟨214⟩, ⟨29162⟩⟩
def transferEvent : Nat := 91191
def frameStart : Nat := 91114
def rule : BoundRule := .product (.predecessor 0 91189 .coefficient) (.predecessor 1 91190 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91189 .coefficient)
      LeftBound91187.bound (LeftBound91187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91190 .coefficient)
      LeftAuthority91164.bound (LeftAuthority91164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91164.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91187.bound LeftAuthority91164.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91187.bound, LeftAuthority91164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91187.actual selector witness) * (LeftAuthority91164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91191

namespace LeftBound91202
def owner : Owner := ⟨.program ⟨214⟩, ⟨17951⟩⟩
def transferEvent : Nat := 91202
def frameStart : Nat := 91114
def rule : BoundRule := .product (.predecessor 0 91200 .coefficient) (.predecessor 1 91201 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91200 .coefficient)
      LeftAuthority91175.bound (LeftAuthority91175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91201 .coefficient)
      LeftAuthority91198.bound (LeftAuthority91198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91198.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority91175.bound LeftAuthority91198.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91175.bound, LeftAuthority91198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority91175.actual selector witness) * (LeftAuthority91198.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91202

namespace LeftBound91210
def owner : Owner := ⟨.program ⟨214⟩, ⟨17952⟩⟩
def transferEvent : Nat := 91210
def frameStart : Nat := 91114
def rule : BoundRule := .sum [.predecessor 0 91208 .coefficient, .predecessor 1 91209 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91208 .coefficient)
      LeftAuthority91206.bound (LeftAuthority91206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91206.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91209 .coefficient)
      LeftBound91202.bound (LeftBound91202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91206.bound, LeftBound91202.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91206.bound, LeftBound91202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority91206.actual selector witness, LeftBound91202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91210

namespace LeftBound91214
def owner : Owner := ⟨.program ⟨214⟩, ⟨29167⟩⟩
def transferEvent : Nat := 91214
def frameStart : Nat := 91114
def rule : BoundRule := .sum [.predecessor 0 91212 .coefficient, .predecessor 1 91213 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91212 .coefficient)
      LeftBound91210.bound (LeftBound91210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91213 .coefficient)
      LeftBound91191.bound (LeftBound91191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91210.bound, LeftBound91191.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91210.bound, LeftBound91191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91210.actual selector witness, LeftBound91191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91214

namespace LeftBound91227
def owner : Owner := ⟨.program ⟨214⟩, ⟨29164⟩⟩
def transferEvent : Nat := 91227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91225 .coefficient, .predecessor 1 91226 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91225 .coefficient)
      LeftBound91056.bound (LeftBound91056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91226 .coefficient)
      LeftBound91039.bound (LeftBound91039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91039.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91056.bound, LeftBound91039.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91056.bound, LeftBound91039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91056.actual selector witness, LeftBound91039.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91227

namespace LeftBound91230
def owner : Owner := ⟨.program ⟨214⟩, ⟨29164⟩⟩
def transferEvent : Nat := 91230
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 91224 .summary, .result 91046 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91224 .summary)
      LeftBound91058.bound (LeftBound91058.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22195⟩⟩) (rawTerms := some (Proof.Events356.exact91224RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91046 .summary)
      LeftBound91041.bound (LeftBound91041.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29163⟩⟩) (rawTerms := some (Proof.Events355.exact91046RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91058.bound, LeftBound91041.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91058.bound, LeftBound91041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91058.actual selector witness, LeftBound91041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91230

namespace LeftBound91234
def owner : Owner := ⟨.program ⟨214⟩, ⟨29165⟩⟩
def transferEvent : Nat := 91234
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91232 .coefficient) (.predecessor 1 91233 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91232 .coefficient)
      LeftBound91227.bound (LeftBound91227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91233 .coefficient)
      LeftBound5598.bound (LeftBound5598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5598.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91227.bound LeftBound5598.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91227.bound, LeftBound5598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91227.actual selector witness) * (LeftBound5598.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91234

namespace LeftBound91235
def owner : Owner := ⟨.program ⟨214⟩, ⟨29165⟩⟩
def transferEvent : Nat := 91235
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩ [⟨.result 5595 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5595 .coefficient)
      LeftAuthority5594.bound (LeftAuthority5594.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6667⟩⟩) (rawTerms := some (Proof.Events021.exact5595RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5594.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5594.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5594.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5594.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91235

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
