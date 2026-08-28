import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard582

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound85296
def owner : Owner := ⟨.program ⟨214⟩, ⟨19459⟩⟩
def transferEvent : Nat := 85296
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩ [⟨.result 85288 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85288 .coefficient)
      LeftAuthority85287.bound (LeftAuthority85287.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19456⟩⟩) (rawTerms := some (Proof.Events333.exact85288RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85287.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85287.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority85287.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85287.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85287.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85296

namespace LeftBound85297
def owner : Owner := ⟨.program ⟨214⟩, ⟨19459⟩⟩
def transferEvent : Nat := 85297
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 85296) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 85296)
      LeftBound85296.bound (LeftBound85296.actual selector witness) := by
  exact .transfer (LeftBound85296.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound85296.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound85296.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound85296.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85297

namespace LeftBound85376
def owner : Owner := ⟨.program ⟨214⟩, ⟨13991⟩⟩
def transferEvent : Nat := 85376
def frameStart : Nat := 85347
def rule : BoundRule := .product (.predecessor 0 85374 .coefficient) (.predecessor 1 85375 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85374 .coefficient)
      LeftAuthority85372.bound (LeftAuthority85372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85372.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85375 .coefficient)
      LeftAuthority85369.bound (LeftAuthority85369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85369.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority85372.bound LeftAuthority85369.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85372.bound, LeftAuthority85369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority85372.actual selector witness) * (LeftAuthority85369.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85376

namespace LeftBound85380
def owner : Owner := ⟨.program ⟨214⟩, ⟨13992⟩⟩
def transferEvent : Nat := 85380
def frameStart : Nat := 85347
def rule : BoundRule := .identity (.predecessor 0 85379 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85379 .coefficient)
      LeftBound85376.bound (LeftBound85376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85376.derived selector witness)

def rawBound : CoeffClass := LeftBound85376.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound85376.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85380

namespace LeftBound85397
def owner : Owner := ⟨.program ⟨214⟩, ⟨14097⟩⟩
def transferEvent : Nat := 85397
def frameStart : Nat := 85347
def rule : BoundRule := .sum [.predecessor 0 85395 .coefficient, .predecessor 1 85396 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85395 .coefficient)
      LeftBound85380.bound (LeftBound85380.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound85380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85396 .coefficient)
      LeftAuthority85393.bound (LeftAuthority85393.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority85393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85380.bound, LeftAuthority85393.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85380.bound, LeftAuthority85393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85380.actual selector witness, LeftAuthority85393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85397

namespace LeftBound85400
def owner : Owner := ⟨.program ⟨214⟩, ⟨14098⟩⟩
def transferEvent : Nat := 85400
def frameStart : Nat := 85347
def rule : BoundRule := .identity (.predecessor 0 85399 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85399 .coefficient)
      LeftBound85397.bound (LeftBound85397.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound85397.derived selector witness)

def rawBound : CoeffClass := LeftBound85397.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound85397.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85400

namespace LeftBound85406
def owner : Owner := ⟨.program ⟨214⟩, ⟨14099⟩⟩
def transferEvent : Nat := 85406
def frameStart : Nat := 85347
def rule : BoundRule := .product (.predecessor 0 85404 .coefficient) (.predecessor 1 85405 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85404 .coefficient)
      LeftAuthority85402.bound (LeftAuthority85402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85402.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85405 .coefficient)
      LeftBound85400.bound (LeftBound85400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85400.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority85402.bound LeftBound85400.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85402.bound, LeftBound85400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority85402.actual selector witness) * (LeftBound85400.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85406

namespace LeftBound85420
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def transferEvent : Nat := 85420
def frameStart : Nat := 85347
def rule : BoundRule := .scale (.predecessor 0 85418 .coefficient) (.value (.predecessor 1 85419 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85418 .coefficient)
      LeftAuthority85416.bound (LeftAuthority85416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85419 .coefficient)
      LeftAuthority85350.bound (LeftAuthority85350.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority85350.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority85416.bound LeftAuthority85350.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85416.bound, LeftAuthority85350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85416.actual selector witness) * (LeftAuthority85350.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound85420

namespace LeftBound85423
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def transferEvent : Nat := 85423
def frameStart : Nat := 85347
def rule : BoundRule := .identity (.predecessor 0 85422 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85422 .coefficient)
      LeftAuthority85410.bound (LeftAuthority85410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85410.derived selector witness)

def rawBound : CoeffClass := LeftAuthority85410.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority85410.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85423

namespace LeftBound85427
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def transferEvent : Nat := 85427
def frameStart : Nat := 85347
def rule : BoundRule := .product (.predecessor 0 85425 .coefficient) (.predecessor 1 85426 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85425 .coefficient)
      LeftBound85423.bound (LeftBound85423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85423.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85426 .coefficient)
      LeftBound85420.bound (LeftBound85420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85420.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85423.bound LeftBound85420.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85423.bound, LeftBound85420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85423.actual selector witness) * (LeftBound85420.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85427

namespace LeftBound85432
def owner : Owner := ⟨.program ⟨214⟩, ⟨14100⟩⟩
def transferEvent : Nat := 85432
def frameStart : Nat := 85347
def rule : BoundRule := .sum [.predecessor 0 85430 .coefficient, .predecessor 1 85431 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85430 .coefficient)
      LeftBound85427.bound (LeftBound85427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85427.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85431 .coefficient)
      LeftBound85406.bound (LeftBound85406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85427.bound, LeftBound85406.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85427.bound, LeftBound85406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85427.actual selector witness, LeftBound85406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85432

namespace LeftBound85436
def owner : Owner := ⟨.program ⟨214⟩, ⟨25992⟩⟩
def transferEvent : Nat := 85436
def frameStart : Nat := 85347
def rule : BoundRule := .product (.predecessor 0 85434 .coefficient) (.predecessor 1 85435 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85434 .coefficient)
      LeftBound85432.bound (LeftBound85432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85435 .coefficient)
      LeftAuthority85391.bound (LeftAuthority85391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85391.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85391.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85432.bound LeftAuthority85391.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85432.bound, LeftAuthority85391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85432.actual selector witness) * (LeftAuthority85391.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85436

namespace LeftBound85447
def owner : Owner := ⟨.program ⟨214⟩, ⟨15823⟩⟩
def transferEvent : Nat := 85447
def frameStart : Nat := 85347
def rule : BoundRule := .product (.predecessor 0 85445 .coefficient) (.predecessor 1 85446 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85445 .coefficient)
      LeftAuthority85402.bound (LeftAuthority85402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85402.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85446 .coefficient)
      LeftAuthority85443.bound (LeftAuthority85443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85443.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85443.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority85402.bound LeftAuthority85443.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85402.bound, LeftAuthority85443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority85402.actual selector witness) * (LeftAuthority85443.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85447

namespace LeftBound85455
def owner : Owner := ⟨.program ⟨214⟩, ⟨15824⟩⟩
def transferEvent : Nat := 85455
def frameStart : Nat := 85347
def rule : BoundRule := .sum [.predecessor 0 85453 .coefficient, .predecessor 1 85454 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85453 .coefficient)
      LeftAuthority85451.bound (LeftAuthority85451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85454 .coefficient)
      LeftBound85447.bound (LeftBound85447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority85451.bound, LeftBound85447.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85451.bound, LeftBound85447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority85451.actual selector witness, LeftBound85447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85455

namespace LeftBound85459
def owner : Owner := ⟨.program ⟨214⟩, ⟨25993⟩⟩
def transferEvent : Nat := 85459
def frameStart : Nat := 85347
def rule : BoundRule := .sum [.predecessor 0 85457 .coefficient, .predecessor 1 85458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85457 .coefficient)
      LeftBound85455.bound (LeftBound85455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85458 .coefficient)
      LeftBound85436.bound (LeftBound85436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85455.bound, LeftBound85436.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85455.bound, LeftBound85436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85455.actual selector witness, LeftBound85436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85459

namespace LeftBound85472
def owner : Owner := ⟨.program ⟨214⟩, ⟨25991⟩⟩
def transferEvent : Nat := 85472
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85470 .coefficient, .predecessor 1 85471 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85470 .coefficient)
      LeftBound85295.bound (LeftBound85295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85471 .coefficient)
      LeftBound85278.bound (LeftBound85278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85278.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85295.bound, LeftBound85278.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85295.bound, LeftBound85278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85295.actual selector witness, LeftBound85278.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85472

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
