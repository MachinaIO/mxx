import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard128
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard533

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79346
def owner : Owner := ⟨.program ⟨214⟩, ⟨14830⟩⟩
def transferEvent : Nat := 79346
def frameStart : Nat := 79281
def rule : BoundRule := .product (.predecessor 0 79344 .coefficient) (.predecessor 1 79345 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79344 .coefficient)
      LeftAuthority79342.bound (LeftAuthority79342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79345 .coefficient)
      LeftBound79340.bound (LeftBound79340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79340.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority79342.bound LeftBound79340.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79342.bound, LeftBound79340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority79342.actual selector witness) * (LeftBound79340.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79346

namespace LeftBound79354
def owner : Owner := ⟨.program ⟨214⟩, ⟨14831⟩⟩
def transferEvent : Nat := 79354
def frameStart : Nat := 79281
def rule : BoundRule := .sum [.predecessor 0 79352 .coefficient, .predecessor 1 79353 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79352 .coefficient)
      LeftAuthority79350.bound (LeftAuthority79350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79350.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79353 .coefficient)
      LeftBound79346.bound (LeftBound79346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority79350.bound, LeftBound79346.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79350.bound, LeftBound79346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority79350.actual selector witness, LeftBound79346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79354

namespace LeftBound79358
def owner : Owner := ⟨.program ⟨214⟩, ⟨26340⟩⟩
def transferEvent : Nat := 79358
def frameStart : Nat := 79281
def rule : BoundRule := .product (.predecessor 0 79356 .coefficient) (.predecessor 1 79357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79356 .coefficient)
      LeftBound79354.bound (LeftBound79354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79357 .coefficient)
      LeftAuthority79331.bound (LeftAuthority79331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79331.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79331.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79354.bound LeftAuthority79331.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79354.bound, LeftAuthority79331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79354.actual selector witness) * (LeftAuthority79331.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79358

namespace LeftBound79369
def owner : Owner := ⟨.program ⟨214⟩, ⟨14884⟩⟩
def transferEvent : Nat := 79369
def frameStart : Nat := 79281
def rule : BoundRule := .product (.predecessor 0 79367 .coefficient) (.predecessor 1 79368 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79367 .coefficient)
      LeftAuthority79342.bound (LeftAuthority79342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79368 .coefficient)
      LeftAuthority79365.bound (LeftAuthority79365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79365.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79365.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority79342.bound LeftAuthority79365.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79342.bound, LeftAuthority79365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority79342.actual selector witness) * (LeftAuthority79365.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79369

namespace LeftBound79377
def owner : Owner := ⟨.program ⟨214⟩, ⟨14885⟩⟩
def transferEvent : Nat := 79377
def frameStart : Nat := 79281
def rule : BoundRule := .sum [.predecessor 0 79375 .coefficient, .predecessor 1 79376 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79375 .coefficient)
      LeftAuthority79373.bound (LeftAuthority79373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79373.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79376 .coefficient)
      LeftBound79369.bound (LeftBound79369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79369.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority79373.bound, LeftBound79369.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79373.bound, LeftBound79369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority79373.actual selector witness, LeftBound79369.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79377

namespace LeftBound79381
def owner : Owner := ⟨.program ⟨214⟩, ⟨26345⟩⟩
def transferEvent : Nat := 79381
def frameStart : Nat := 79281
def rule : BoundRule := .sum [.predecessor 0 79379 .coefficient, .predecessor 1 79380 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79379 .coefficient)
      LeftBound79377.bound (LeftBound79377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79377.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79380 .coefficient)
      LeftBound79358.bound (LeftBound79358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79358.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79377.bound, LeftBound79358.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79377.bound, LeftBound79358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79377.actual selector witness, LeftBound79358.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79381

namespace LeftBound79394
def owner : Owner := ⟨.program ⟨214⟩, ⟨26342⟩⟩
def transferEvent : Nat := 79394
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79392 .coefficient, .predecessor 1 79393 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79392 .coefficient)
      LeftBound79223.bound (LeftBound79223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79393 .coefficient)
      LeftBound79206.bound (LeftBound79206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79206.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79223.bound, LeftBound79206.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79223.bound, LeftBound79206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79223.actual selector witness, LeftBound79206.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79394

namespace LeftBound79397
def owner : Owner := ⟨.program ⟨214⟩, ⟨26342⟩⟩
def transferEvent : Nat := 79397
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79391 .summary, .result 79213 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79391 .summary)
      LeftBound79225.bound (LeftBound79225.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20319⟩⟩) (rawTerms := some (Proof.Events310.exact79391RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79213 .summary)
      LeftBound79208.bound (LeftBound79208.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26341⟩⟩) (rawTerms := some (Proof.Events309.exact79213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79208.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79225.bound, LeftBound79208.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79225.bound, LeftBound79208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79225.actual selector witness, LeftBound79208.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79397

namespace LeftBound79401
def owner : Owner := ⟨.program ⟨214⟩, ⟨26343⟩⟩
def transferEvent : Nat := 79401
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79399 .coefficient) (.predecessor 1 79400 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79399 .coefficient)
      LeftBound79394.bound (LeftBound79394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79400 .coefficient)
      LeftBound5858.bound (LeftBound5858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79394.bound LeftBound5858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79394.bound, LeftBound5858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79394.actual selector witness) * (LeftBound5858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79401

namespace LeftBound79402
def owner : Owner := ⟨.program ⟨214⟩, ⟨26343⟩⟩
def transferEvent : Nat := 79402
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩ [⟨.result 5855 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5855 .coefficient)
      LeftAuthority5854.bound (LeftAuthority5854.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6679⟩⟩) (rawTerms := some (Proof.Events022.exact5855RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5854.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5854.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5854.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79402

namespace LeftBound79403
def owner : Owner := ⟨.program ⟨214⟩, ⟨26343⟩⟩
def transferEvent : Nat := 79403
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 79398 .summary) (.transfer 79402) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79398 .summary)
      LeftBound79397.bound (LeftBound79397.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26342⟩⟩) (rawTerms := some (Proof.Events310.exact79398RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79402)
      LeftBound79402.bound (LeftBound79402.actual selector witness) := by
  exact .transfer (LeftBound79402.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79397.bound LeftBound79402.bound
def bound : CoeffClass := .finite ⟨4741253940199267499646124032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79397.bound, LeftBound79402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79397.actual selector witness) * (LeftBound79402.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79403

namespace LeftBound79411
def owner : Owner := ⟨.program ⟨214⟩, ⟨6625⟩⟩
def transferEvent : Nat := 79411
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 79409 .coefficient) (.predecessor 1 79410 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79409 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79410 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound79411

namespace LeftBound79416
def owner : Owner := ⟨.program ⟨214⟩, ⟨7178⟩⟩
def transferEvent : Nat := 79416
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79414 .coefficient) (.predecessor 1 79415 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79414 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79415 .coefficient)
      LeftBound5872.bound (LeftBound5872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5872.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound5872.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound5872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound5872.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79416

namespace LeftBound79421
def owner : Owner := ⟨.program ⟨214⟩, ⟨7749⟩⟩
def transferEvent : Nat := 79421
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79419 .coefficient, .predecessor 1 79420 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79419 .coefficient)
      LeftBound79416.bound (LeftBound79416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79420 .coefficient)
      LeftBound79411.bound (LeftBound79411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79411.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79411.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79416.bound, LeftBound79411.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79416.bound, LeftBound79411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79416.actual selector witness, LeftBound79411.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79421

namespace LeftBound79425
def owner : Owner := ⟨.program ⟨214⟩, ⟨7750⟩⟩
def transferEvent : Nat := 79425
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79423 .coefficient, .predecessor 1 79424 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79423 .coefficient)
      LeftBound79421.bound (LeftBound79421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79424 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79421.bound, LeftBound20907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79421.bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79421.actual selector witness, LeftBound20907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79425

namespace LeftBound79426
def owner : Owner := ⟨.program ⟨214⟩, ⟨7750⟩⟩
def transferEvent : Nat := 79426
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩ [⟨.result 20908 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20908 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨74⟩⟩) (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20907.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound20907.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79426

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
