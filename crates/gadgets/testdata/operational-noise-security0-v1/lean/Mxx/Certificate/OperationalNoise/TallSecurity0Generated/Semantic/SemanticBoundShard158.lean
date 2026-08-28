import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard157

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24346
def owner : Owner := ⟨.program ⟨214⟩, ⟨11987⟩⟩
def transferEvent : Nat := 24346
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24344 .coefficient) (.predecessor 1 24345 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24344 .coefficient)
      LeftBound24340.bound (LeftBound24340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24345 .coefficient)
      LeftAuthority982.bound (LeftAuthority982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority982.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority982.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound24340.bound LeftAuthority982.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24340.bound, LeftAuthority982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound24340.actual selector witness) * (LeftAuthority982.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24346

namespace LeftBound24347
def owner : Owner := ⟨.program ⟨214⟩, ⟨11987⟩⟩
def transferEvent : Nat := 24347
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9730⟩⟩], []⟩ [⟨.result 983 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 983 .coefficient)
      LeftAuthority982.bound (LeftAuthority982.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9730⟩⟩) (rawTerms := some (Proof.Events003.exact983RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority982.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority982.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority982.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority982.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24347

namespace LeftBound24348
def owner : Owner := ⟨.program ⟨214⟩, ⟨11987⟩⟩
def transferEvent : Nat := 24348
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 24343 .summary) (.transfer 24347) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24343 .summary)
      LeftBound24341.bound (LeftBound24341.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11986⟩⟩) (rawTerms := some (Proof.Events095.exact24343RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24347)
      LeftBound24347.bound (LeftBound24347.actual selector witness) := by
  exact .transfer (LeftBound24347.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound24341.bound LeftBound24347.bound
def bound : CoeffClass := .finite ⟨29952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24341.bound, LeftBound24347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound24341.actual selector witness) * (LeftBound24347.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24348

namespace LeftBound24354
def owner : Owner := ⟨.program ⟨214⟩, ⟨9731⟩⟩
def transferEvent : Nat := 24354
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 24352 .coefficient) (.predecessor 1 24353 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24352 .coefficient)
      LeftAuthority982.bound (LeftAuthority982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority982.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24353 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority982.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority982.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority982.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24354

namespace LeftBound24359
def owner : Owner := ⟨.program ⟨214⟩, ⟨7334⟩⟩
def transferEvent : Nat := 24359
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24357 .coefficient) (.predecessor 1 24358 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24357 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24358 .coefficient)
      LeftBound9518.bound (LeftBound9518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound9518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound9518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound9518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24359

namespace LeftBound24364
def owner : Owner := ⟨.program ⟨214⟩, ⟨9732⟩⟩
def transferEvent : Nat := 24364
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24362 .coefficient, .predecessor 1 24363 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24362 .coefficient)
      LeftBound24359.bound (LeftBound24359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24359.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24363 .coefficient)
      LeftBound24354.bound (LeftBound24354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24359.bound, LeftBound24354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24359.bound, LeftBound24354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24359.actual selector witness, LeftBound24354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24364

namespace LeftBound24368
def owner : Owner := ⟨.program ⟨214⟩, ⟨9733⟩⟩
def transferEvent : Nat := 24368
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24366 .coefficient, .predecessor 1 24367 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24366 .coefficient)
      LeftBound24364.bound (LeftBound24364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24364.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24367 .coefficient)
      LeftBound9510.bound (LeftBound9510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24364.bound, LeftBound9510.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24364.bound, LeftBound9510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24364.actual selector witness, LeftBound9510.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24368

namespace LeftBound24369
def owner : Owner := ⟨.program ⟨214⟩, ⟨9733⟩⟩
def transferEvent : Nat := 24369
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩ [⟨.result 9511 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9511 .coefficient)
      LeftBound9510.bound (LeftBound9510.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨78⟩⟩) (rawTerms := some (Proof.Events037.exact9511RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9510.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9510.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24369

namespace LeftBound24374
def owner : Owner := ⟨.program ⟨214⟩, ⟨9734⟩⟩
def transferEvent : Nat := 24374
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24372 .coefficient) (.predecessor 1 24373 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24372 .coefficient)
      LeftBound24368.bound (LeftBound24368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24373 .coefficient)
      LeftBound9507.bound (LeftBound9507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9507.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24368.bound LeftBound9507.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24368.bound, LeftBound9507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24368.actual selector witness) * (LeftBound9507.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24374

namespace LeftBound24375
def owner : Owner := ⟨.program ⟨214⟩, ⟨9734⟩⟩
def transferEvent : Nat := 24375
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩ [⟨.result 9504 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9504 .coefficient)
      LeftAuthority9503.bound (LeftAuthority9503.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7864⟩⟩) (rawTerms := some (Proof.Events037.exact9504RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9503.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9503.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9503.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24375

namespace LeftBound24376
def owner : Owner := ⟨.program ⟨214⟩, ⟨9734⟩⟩
def transferEvent : Nat := 24376
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 24371 .summary) (.transfer 24375) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24371 .summary)
      LeftBound24369.bound (LeftBound24369.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9733⟩⟩) (rawTerms := some (Proof.Events095.exact24371RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24375)
      LeftBound24375.bound (LeftBound24375.actual selector witness) := by
  exact .transfer (LeftBound24375.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24369.bound LeftBound24375.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24369.bound, LeftBound24375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24369.actual selector witness) * (LeftBound24375.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24376

namespace LeftBound24384
def owner : Owner := ⟨.program ⟨214⟩, ⟨11988⟩⟩
def transferEvent : Nat := 24384
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24382 .coefficient, .predecessor 1 24383 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24382 .coefficient)
      LeftBound24374.bound (LeftBound24374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24383 .coefficient)
      LeftBound24346.bound (LeftBound24346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24374.bound, LeftBound24346.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24374.bound, LeftBound24346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24374.actual selector witness, LeftBound24346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24384

namespace LeftBound24386
def owner : Owner := ⟨.program ⟨214⟩, ⟨11988⟩⟩
def transferEvent : Nat := 24386
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 24381 .summary, .result 24351 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24381 .summary)
      LeftBound24376.bound (LeftBound24376.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9734⟩⟩) (rawTerms := some (Proof.Events095.exact24381RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24351 .summary)
      LeftBound24348.bound (LeftBound24348.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11987⟩⟩) (rawTerms := some (Proof.Events095.exact24351RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24348.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24376.bound, LeftBound24348.bound]
def bound : CoeffClass := .finite ⟨95450368, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24376.bound, LeftBound24348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24376.actual selector witness, LeftBound24348.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24386

namespace LeftBound24390
def owner : Owner := ⟨.program ⟨214⟩, ⟨25235⟩⟩
def transferEvent : Nat := 24390
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24388 .coefficient) (.predecessor 1 24389 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24388 .coefficient)
      LeftBound24384.bound (LeftBound24384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24389 .coefficient)
      LeftAuthority24322.bound (LeftAuthority24322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24322.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24384.bound LeftAuthority24322.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24384.bound, LeftAuthority24322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24384.actual selector witness) * (LeftAuthority24322.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24390

namespace LeftBound24391
def owner : Owner := ⟨.program ⟨214⟩, ⟨25235⟩⟩
def transferEvent : Nat := 24391
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25234⟩⟩]⟩ [⟨.result 24323 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24323 .coefficient)
      LeftAuthority24322.bound (LeftAuthority24322.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25234⟩⟩) (rawTerms := some (Proof.Events095.exact24323RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24322.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24322.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24322.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24391

namespace LeftBound24392
def owner : Owner := ⟨.program ⟨214⟩, ⟨25235⟩⟩
def transferEvent : Nat := 24392
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 24387 .summary) (.transfer 24391) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24387 .summary)
      LeftBound24386.bound (LeftBound24386.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11988⟩⟩) (rawTerms := some (Proof.Events095.exact24387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24391)
      LeftBound24391.bound (LeftBound24391.actual selector witness) := by
  exact .transfer (LeftBound24391.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24386.bound LeftBound24391.bound
def bound : CoeffClass := .finite ⟨350304377765888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24386.bound, LeftBound24391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24386.actual selector witness) * (LeftBound24391.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24392

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
