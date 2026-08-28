import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard655
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard656

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96343
def owner : Owner := ⟨.program ⟨214⟩, ⟨25441⟩⟩
def transferEvent : Nat := 96343
def frameStart : Nat := 96241
def rule : BoundRule := .sum [.predecessor 0 96341 .coefficient, .predecessor 1 96342 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96341 .coefficient)
      LeftBound96339.bound (LeftBound96339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96342 .coefficient)
      LeftBound96320.bound (LeftBound96320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96339.bound, LeftBound96320.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96339.bound, LeftBound96320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96339.actual selector witness, LeftBound96320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96343

namespace LeftBound96356
def owner : Owner := ⟨.program ⟨214⟩, ⟨25439⟩⟩
def transferEvent : Nat := 96356
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96354 .coefficient, .predecessor 1 96355 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96354 .coefficient)
      LeftBound96201.bound (LeftBound96201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96355 .coefficient)
      LeftBound96184.bound (LeftBound96184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96201.bound, LeftBound96184.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96201.bound, LeftBound96184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96201.actual selector witness, LeftBound96184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96356

namespace LeftBound96359
def owner : Owner := ⟨.program ⟨214⟩, ⟨25439⟩⟩
def transferEvent : Nat := 96359
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 96353 .summary, .result 96191 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96353 .summary)
      LeftBound96203.bound (LeftBound96203.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19952⟩⟩) (rawTerms := some (Proof.Events376.exact96353RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96191 .summary)
      LeftBound96186.bound (LeftBound96186.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25438⟩⟩) (rawTerms := some (Proof.Events375.exact96191RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96186.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96203.bound, LeftBound96186.bound]
def bound : CoeffClass := .finite ⟨352134001995776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96203.bound, LeftBound96186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96203.actual selector witness, LeftBound96186.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96359

namespace LeftBound96363
def owner : Owner := ⟨.program ⟨214⟩, ⟨29135⟩⟩
def transferEvent : Nat := 96363
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96361 .coefficient) (.predecessor 1 96362 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96361 .coefficient)
      LeftBound96356.bound (LeftBound96356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96356.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96362 .coefficient)
      LeftAuthority96106.bound (LeftAuthority96106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96106.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96356.bound LeftAuthority96106.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96356.bound, LeftAuthority96106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96356.actual selector witness) * (LeftAuthority96106.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96363

namespace LeftBound96364
def owner : Owner := ⟨.program ⟨214⟩, ⟨29135⟩⟩
def transferEvent : Nat := 96364
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩ [⟨.result 96107 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96107 .coefficient)
      LeftAuthority96106.bound (LeftAuthority96106.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29133⟩⟩) (rawTerms := some (Proof.Events375.exact96107RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96106.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96106.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96106.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96364

namespace LeftBound96365
def owner : Owner := ⟨.program ⟨214⟩, ⟨29135⟩⟩
def transferEvent : Nat := 96365
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96360 .summary) (.transfer 96364) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96360 .summary)
      LeftBound96359.bound (LeftBound96359.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25439⟩⟩) (rawTerms := some (Proof.Events376.exact96360RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96364)
      LeftBound96364.bound (LeftBound96364.actual selector witness) := by
  exact .transfer (LeftBound96364.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96359.bound LeftBound96364.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96359.bound, LeftBound96364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96359.actual selector witness) * (LeftBound96364.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96365

namespace LeftBound96376
def owner : Owner := ⟨.program ⟨214⟩, ⟨22255⟩⟩
def transferEvent : Nat := 96376
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 96374 .coefficient) (.value (.predecessor 1 96375 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96374 .coefficient)
      LeftAuthority96372.bound (LeftAuthority96372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96372.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96375 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority96372.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96372.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96372.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96376

namespace LeftBound96380
def owner : Owner := ⟨.program ⟨214⟩, ⟨22256⟩⟩
def transferEvent : Nat := 96380
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96378 .coefficient) (.predecessor 1 96379 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96378 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96379 .coefficient)
      LeftBound96376.bound (LeftBound96376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96376.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound96376.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound96376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound96376.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96380

namespace LeftBound96381
def owner : Owner := ⟨.program ⟨214⟩, ⟨22256⟩⟩
def transferEvent : Nat := 96381
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩ [⟨.result 96373 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96373 .coefficient)
      LeftAuthority96372.bound (LeftAuthority96372.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22253⟩⟩) (rawTerms := some (Proof.Events376.exact96373RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96372.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96372.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96372.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96372.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96381

namespace LeftBound96382
def owner : Owner := ⟨.program ⟨214⟩, ⟨22256⟩⟩
def transferEvent : Nat := 96382
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 96381) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96381)
      LeftBound96381.bound (LeftBound96381.actual selector witness) := by
  exact .transfer (LeftBound96381.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound96381.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound96381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound96381.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96382

namespace LeftBound96453
def owner : Owner := ⟨.program ⟨214⟩, ⟨16540⟩⟩
def transferEvent : Nat := 96453
def frameStart : Nat := 96426
def rule : BoundRule := .identity (.predecessor 0 96452 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96452 .coefficient)
      LeftAuthority96450.bound (LeftAuthority96450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96450.derived selector witness)

def rawBound : CoeffClass := LeftAuthority96450.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority96450.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96453

namespace LeftBound96470
def owner : Owner := ⟨.program ⟨214⟩, ⟨16581⟩⟩
def transferEvent : Nat := 96470
def frameStart : Nat := 96426
def rule : BoundRule := .sum [.predecessor 0 96468 .coefficient, .predecessor 1 96469 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96468 .coefficient)
      LeftBound96453.bound (LeftBound96453.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96453.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96469 .coefficient)
      LeftAuthority96466.bound (LeftAuthority96466.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96466.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96453.bound, LeftAuthority96466.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96453.bound, LeftAuthority96466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96453.actual selector witness, LeftAuthority96466.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96470

namespace LeftBound96473
def owner : Owner := ⟨.program ⟨214⟩, ⟨16582⟩⟩
def transferEvent : Nat := 96473
def frameStart : Nat := 96426
def rule : BoundRule := .identity (.predecessor 0 96472 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96472 .coefficient)
      LeftBound96470.bound (LeftBound96470.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96470.derived selector witness)

def rawBound : CoeffClass := LeftBound96470.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound96470.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96473

namespace LeftBound96479
def owner : Owner := ⟨.program ⟨214⟩, ⟨16583⟩⟩
def transferEvent : Nat := 96479
def frameStart : Nat := 96426
def rule : BoundRule := .product (.predecessor 0 96477 .coefficient) (.predecessor 1 96478 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96477 .coefficient)
      LeftAuthority96475.bound (LeftAuthority96475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96475.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96478 .coefficient)
      LeftBound96473.bound (LeftBound96473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority96475.bound LeftBound96473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96475.bound, LeftBound96473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority96475.actual selector witness) * (LeftBound96473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96479

namespace LeftBound96487
def owner : Owner := ⟨.program ⟨214⟩, ⟨16584⟩⟩
def transferEvent : Nat := 96487
def frameStart : Nat := 96426
def rule : BoundRule := .sum [.predecessor 0 96485 .coefficient, .predecessor 1 96486 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96485 .coefficient)
      LeftAuthority96483.bound (LeftAuthority96483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96483.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96486 .coefficient)
      LeftBound96479.bound (LeftBound96479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority96483.bound, LeftBound96479.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96483.bound, LeftBound96479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority96483.actual selector witness, LeftBound96479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96487

namespace LeftBound96491
def owner : Owner := ⟨.program ⟨214⟩, ⟨29134⟩⟩
def transferEvent : Nat := 96491
def frameStart : Nat := 96426
def rule : BoundRule := .product (.predecessor 0 96489 .coefficient) (.predecessor 1 96490 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96489 .coefficient)
      LeftBound96487.bound (LeftBound96487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96490 .coefficient)
      LeftAuthority96464.bound (LeftAuthority96464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96487.bound LeftAuthority96464.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96487.bound, LeftAuthority96464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96487.actual selector witness) * (LeftAuthority96464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96491

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
