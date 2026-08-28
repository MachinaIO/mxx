import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard150

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound23395
def owner : Owner := ⟨.program ⟨214⟩, ⟨7336⟩⟩
def transferEvent : Nat := 23395
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23393 .coefficient) (.predecessor 1 23394 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23393 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23394 .coefficient)
      LeftBound8516.bound (LeftBound8516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound8516.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound8516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound8516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23395

namespace LeftBound23400
def owner : Owner := ⟨.program ⟨214⟩, ⟨9942⟩⟩
def transferEvent : Nat := 23400
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23398 .coefficient, .predecessor 1 23399 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23398 .coefficient)
      LeftBound23395.bound (LeftBound23395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23399 .coefficient)
      LeftBound23390.bound (LeftBound23390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23390.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23395.bound, LeftBound23390.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23395.bound, LeftBound23390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23395.actual selector witness, LeftBound23390.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23400

namespace LeftBound23404
def owner : Owner := ⟨.program ⟨214⟩, ⟨9943⟩⟩
def transferEvent : Nat := 23404
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23402 .coefficient, .predecessor 1 23403 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23402 .coefficient)
      LeftBound23400.bound (LeftBound23400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23400.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23403 .coefficient)
      LeftBound8508.bound (LeftBound8508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23400.bound, LeftBound8508.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23400.bound, LeftBound8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23400.actual selector witness, LeftBound8508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23404

namespace LeftBound23405
def owner : Owner := ⟨.program ⟨214⟩, ⟨9943⟩⟩
def transferEvent : Nat := 23405
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨80⟩⟩]⟩ [⟨.result 8509 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8509 .coefficient)
      LeftBound8508.bound (LeftBound8508.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨80⟩⟩) (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8508.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8508.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23405

namespace LeftBound23410
def owner : Owner := ⟨.program ⟨214⟩, ⟨9944⟩⟩
def transferEvent : Nat := 23410
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23408 .coefficient) (.predecessor 1 23409 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23408 .coefficient)
      LeftBound23404.bound (LeftBound23404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23409 .coefficient)
      LeftBound8505.bound (LeftBound8505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23404.bound LeftBound8505.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23404.bound, LeftBound8505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23404.actual selector witness) * (LeftBound8505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23410

namespace LeftBound23411
def owner : Owner := ⟨.program ⟨214⟩, ⟨9944⟩⟩
def transferEvent : Nat := 23411
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩ [⟨.result 8502 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8502 .coefficient)
      LeftAuthority8501.bound (LeftAuthority8501.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7870⟩⟩) (rawTerms := some (Proof.Events033.exact8502RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8501.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8501.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8501.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23411

namespace LeftBound23412
def owner : Owner := ⟨.program ⟨214⟩, ⟨9944⟩⟩
def transferEvent : Nat := 23412
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23407 .summary) (.transfer 23411) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23407 .summary)
      LeftBound23405.bound (LeftBound23405.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9943⟩⟩) (rawTerms := some (Proof.Events091.exact23407RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23411)
      LeftBound23411.bound (LeftBound23411.actual selector witness) := by
  exact .transfer (LeftBound23411.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23405.bound LeftBound23411.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23405.bound, LeftBound23411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23405.actual selector witness) * (LeftBound23411.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23412

namespace LeftBound23420
def owner : Owner := ⟨.program ⟨214⟩, ⟨12597⟩⟩
def transferEvent : Nat := 23420
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23418 .coefficient, .predecessor 1 23419 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23418 .coefficient)
      LeftBound23410.bound (LeftBound23410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23419 .coefficient)
      LeftBound23382.bound (LeftBound23382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23382.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23410.bound, LeftBound23382.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23410.bound, LeftBound23382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23410.actual selector witness, LeftBound23382.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23420

namespace LeftBound23422
def owner : Owner := ⟨.program ⟨214⟩, ⟨12597⟩⟩
def transferEvent : Nat := 23422
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 23417 .summary, .result 23387 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23417 .summary)
      LeftBound23412.bound (LeftBound23412.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9944⟩⟩) (rawTerms := some (Proof.Events091.exact23417RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23387 .summary)
      LeftBound23384.bound (LeftBound23384.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12596⟩⟩) (rawTerms := some (Proof.Events091.exact23387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23384.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23412.bound, LeftBound23384.bound]
def bound : CoeffClass := .finite ⟨95455360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23412.bound, LeftBound23384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23412.actual selector witness, LeftBound23384.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23422

namespace LeftBound23426
def owner : Owner := ⟨.program ⟨214⟩, ⟨25466⟩⟩
def transferEvent : Nat := 23426
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23424 .coefficient) (.predecessor 1 23425 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23424 .coefficient)
      LeftBound23420.bound (LeftBound23420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23425 .coefficient)
      LeftAuthority23358.bound (LeftAuthority23358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23358.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23358.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23420.bound LeftAuthority23358.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23420.bound, LeftAuthority23358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23420.actual selector witness) * (LeftAuthority23358.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23426

namespace LeftBound23427
def owner : Owner := ⟨.program ⟨214⟩, ⟨25466⟩⟩
def transferEvent : Nat := 23427
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25465⟩⟩]⟩ [⟨.result 23359 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23359 .coefficient)
      LeftAuthority23358.bound (LeftAuthority23358.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25465⟩⟩) (rawTerms := some (Proof.Events091.exact23359RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23358.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23358.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23358.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23358.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23427

namespace LeftBound23428
def owner : Owner := ⟨.program ⟨214⟩, ⟨25466⟩⟩
def transferEvent : Nat := 23428
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23423 .summary) (.transfer 23427) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23423 .summary)
      LeftBound23422.bound (LeftBound23422.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12597⟩⟩) (rawTerms := some (Proof.Events091.exact23423RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23427)
      LeftBound23427.bound (LeftBound23427.actual selector witness) := by
  exact .transfer (LeftBound23427.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23422.bound LeftBound23427.bound
def bound : CoeffClass := .finite ⟨350322698485760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23422.bound, LeftBound23427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23422.actual selector witness) * (LeftBound23427.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23428

namespace LeftBound23439
def owner : Owner := ⟨.program ⟨214⟩, ⟨19974⟩⟩
def transferEvent : Nat := 23439
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 23437 .coefficient) (.value (.predecessor 1 23438 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23437 .coefficient)
      LeftAuthority23435.bound (LeftAuthority23435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23435.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23438 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority23435.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23435.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23435.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23439

namespace LeftBound23443
def owner : Owner := ⟨.program ⟨214⟩, ⟨19975⟩⟩
def transferEvent : Nat := 23443
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23441 .coefficient) (.predecessor 1 23442 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23441 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23442 .coefficient)
      LeftBound23439.bound (LeftBound23439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23439.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound23439.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound23439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound23439.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23443

namespace LeftBound23444
def owner : Owner := ⟨.program ⟨214⟩, ⟨19975⟩⟩
def transferEvent : Nat := 23444
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩ [⟨.result 23436 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23436 .coefficient)
      LeftAuthority23435.bound (LeftAuthority23435.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19972⟩⟩) (rawTerms := some (Proof.Events091.exact23436RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23435.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23435.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23435.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23444

namespace LeftBound23445
def owner : Owner := ⟨.program ⟨214⟩, ⟨19975⟩⟩
def transferEvent : Nat := 23445
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 23444) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23444)
      LeftBound23444.bound (LeftBound23444.actual selector witness) := by
  exact .transfer (LeftBound23444.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound23444.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound23444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound23444.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23445

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
