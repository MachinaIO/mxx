import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard350
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard351

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound52377
def owner : Owner := ⟨.program ⟨214⟩, ⟨25536⟩⟩
def transferEvent : Nat := 52377
def frameStart : Nat := 52263
def rule : BoundRule := .sum [.predecessor 0 52375 .coefficient, .predecessor 1 52376 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52375 .coefficient)
      LeftBound52373.bound (LeftBound52373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52376 .coefficient)
      LeftBound52354.bound (LeftBound52354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52373.bound, LeftBound52354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52373.bound, LeftBound52354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52373.actual selector witness, LeftBound52354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52377

namespace LeftBound52390
def owner : Owner := ⟨.program ⟨214⟩, ⟨25534⟩⟩
def transferEvent : Nat := 52390
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52388 .coefficient, .predecessor 1 52389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52388 .coefficient)
      LeftBound52211.bound (LeftBound52211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52389 .coefficient)
      LeftBound52194.bound (LeftBound52194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52211.bound, LeftBound52194.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52211.bound, LeftBound52194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52211.actual selector witness, LeftBound52194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52390

namespace LeftBound52393
def owner : Owner := ⟨.program ⟨214⟩, ⟨25534⟩⟩
def transferEvent : Nat := 52393
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 52387 .summary, .result 52201 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52387 .summary)
      LeftBound52213.bound (LeftBound52213.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20039⟩⟩) (rawTerms := some (Proof.Events204.exact52387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52201 .summary)
      LeftBound52196.bound (LeftBound52196.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25533⟩⟩) (rawTerms := some (Proof.Events203.exact52201RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52196.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52213.bound, LeftBound52196.bound]
def bound : CoeffClass := .finite ⟨352146215809024, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52213.bound, LeftBound52196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52213.actual selector witness, LeftBound52196.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52393

namespace LeftBound52397
def owner : Owner := ⟨.program ⟨214⟩, ⟨29400⟩⟩
def transferEvent : Nat := 52397
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52395 .coefficient) (.predecessor 1 52396 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52395 .coefficient)
      LeftBound52390.bound (LeftBound52390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52396 .coefficient)
      LeftAuthority52116.bound (LeftAuthority52116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52116.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52116.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52390.bound LeftAuthority52116.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52390.bound, LeftAuthority52116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52390.actual selector witness) * (LeftAuthority52116.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52397

namespace LeftBound52398
def owner : Owner := ⟨.program ⟨214⟩, ⟨29400⟩⟩
def transferEvent : Nat := 52398
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29398⟩⟩]⟩ [⟨.result 52117 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52117 .coefficient)
      LeftAuthority52116.bound (LeftAuthority52116.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29398⟩⟩) (rawTerms := some (Proof.Events203.exact52117RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52116.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52116.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority52116.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52116.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound52398

namespace LeftBound52399
def owner : Owner := ⟨.program ⟨214⟩, ⟨29400⟩⟩
def transferEvent : Nat := 52399
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 52394 .summary) (.transfer 52398) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52394 .summary)
      LeftBound52393.bound (LeftBound52393.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25534⟩⟩) (rawTerms := some (Proof.Events204.exact52394RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 52398)
      LeftBound52398.bound (LeftBound52398.actual selector witness) := by
  exact .transfer (LeftBound52398.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52393.bound LeftBound52398.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52393.bound, LeftBound52398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52393.actual selector witness) * (LeftBound52398.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52399

namespace LeftBound52410
def owner : Owner := ⟨.program ⟨214⟩, ⟨22414⟩⟩
def transferEvent : Nat := 52410
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 52408 .coefficient) (.value (.predecessor 1 52409 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52408 .coefficient)
      LeftAuthority52406.bound (LeftAuthority52406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52409 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority52406.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52406.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52406.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52410

namespace LeftBound52414
def owner : Owner := ⟨.program ⟨214⟩, ⟨22415⟩⟩
def transferEvent : Nat := 52414
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52412 .coefficient) (.predecessor 1 52413 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52412 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52413 .coefficient)
      LeftBound52410.bound (LeftBound52410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52410.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound52410.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound52410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound52410.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52414

namespace LeftBound52415
def owner : Owner := ⟨.program ⟨214⟩, ⟨22415⟩⟩
def transferEvent : Nat := 52415
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩ [⟨.result 52407 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52407 .coefficient)
      LeftAuthority52406.bound (LeftAuthority52406.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22412⟩⟩) (rawTerms := some (Proof.Events204.exact52407RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52406.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority52406.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52406.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound52415

namespace LeftBound52416
def owner : Owner := ⟨.program ⟨214⟩, ⟨22415⟩⟩
def transferEvent : Nat := 52416
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 52415) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 52415)
      LeftBound52415.bound (LeftBound52415.actual selector witness) := by
  exact .transfer (LeftBound52415.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound52415.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound52415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound52415.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52416

namespace LeftBound52511
def owner : Owner := ⟨.program ⟨214⟩, ⟨16638⟩⟩
def transferEvent : Nat := 52511
def frameStart : Nat := 52472
def rule : BoundRule := .identity (.predecessor 0 52510 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52510 .coefficient)
      LeftAuthority52508.bound (LeftAuthority52508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52508.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52508.derived selector witness)

def rawBound : CoeffClass := LeftAuthority52508.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority52508.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52511

namespace LeftBound52528
def owner : Owner := ⟨.program ⟨214⟩, ⟨16712⟩⟩
def transferEvent : Nat := 52528
def frameStart : Nat := 52472
def rule : BoundRule := .sum [.predecessor 0 52526 .coefficient, .predecessor 1 52527 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52526 .coefficient)
      LeftBound52511.bound (LeftBound52511.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52527 .coefficient)
      LeftAuthority52524.bound (LeftAuthority52524.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority52524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52511.bound, LeftAuthority52524.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52511.bound, LeftAuthority52524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52511.actual selector witness, LeftAuthority52524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52528

namespace LeftBound52531
def owner : Owner := ⟨.program ⟨214⟩, ⟨16713⟩⟩
def transferEvent : Nat := 52531
def frameStart : Nat := 52472
def rule : BoundRule := .identity (.predecessor 0 52530 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52530 .coefficient)
      LeftBound52528.bound (LeftBound52528.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52528.derived selector witness)

def rawBound : CoeffClass := LeftBound52528.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound52528.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52531

namespace LeftBound52537
def owner : Owner := ⟨.program ⟨214⟩, ⟨16714⟩⟩
def transferEvent : Nat := 52537
def frameStart : Nat := 52472
def rule : BoundRule := .product (.predecessor 0 52535 .coefficient) (.predecessor 1 52536 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52535 .coefficient)
      LeftAuthority52533.bound (LeftAuthority52533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52536 .coefficient)
      LeftBound52531.bound (LeftBound52531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52531.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority52533.bound LeftBound52531.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52533.bound, LeftBound52531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority52533.actual selector witness) * (LeftBound52531.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52537

namespace LeftBound52545
def owner : Owner := ⟨.program ⟨214⟩, ⟨16715⟩⟩
def transferEvent : Nat := 52545
def frameStart : Nat := 52472
def rule : BoundRule := .sum [.predecessor 0 52543 .coefficient, .predecessor 1 52544 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52543 .coefficient)
      LeftAuthority52541.bound (LeftAuthority52541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52541.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52544 .coefficient)
      LeftBound52537.bound (LeftBound52537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52537.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority52541.bound, LeftBound52537.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52541.bound, LeftBound52537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority52541.actual selector witness, LeftBound52537.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52545

namespace LeftBound52549
def owner : Owner := ⟨.program ⟨214⟩, ⟨29399⟩⟩
def transferEvent : Nat := 52549
def frameStart : Nat := 52472
def rule : BoundRule := .product (.predecessor 0 52547 .coefficient) (.predecessor 1 52548 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52547 .coefficient)
      LeftBound52545.bound (LeftBound52545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52548 .coefficient)
      LeftAuthority52522.bound (LeftAuthority52522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52522.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52545.bound LeftAuthority52522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52545.bound, LeftAuthority52522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52545.actual selector witness) * (LeftAuthority52522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52549

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
