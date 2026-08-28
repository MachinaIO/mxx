import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard634

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound93808
def owner : Owner := ⟨.program ⟨214⟩, ⟨20322⟩⟩
def transferEvent : Nat := 93808
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 93806 .coefficient) (.value (.predecessor 1 93807 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93806 .coefficient)
      LeftAuthority93804.bound (LeftAuthority93804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93807 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority93804.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93804.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93804.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound93808

namespace LeftBound93812
def owner : Owner := ⟨.program ⟨214⟩, ⟨20323⟩⟩
def transferEvent : Nat := 93812
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93810 .coefficient) (.predecessor 1 93811 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93810 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93811 .coefficient)
      LeftBound93808.bound (LeftBound93808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93808.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound93808.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound93808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound93808.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93812

namespace LeftBound93813
def owner : Owner := ⟨.program ⟨214⟩, ⟨20323⟩⟩
def transferEvent : Nat := 93813
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩ [⟨.result 93805 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93805 .coefficient)
      LeftAuthority93804.bound (LeftAuthority93804.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20320⟩⟩) (rawTerms := some (Proof.Events366.exact93805RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93804.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority93804.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93804.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93813

namespace LeftBound93814
def owner : Owner := ⟨.program ⟨214⟩, ⟨20323⟩⟩
def transferEvent : Nat := 93814
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 93813) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93813)
      LeftBound93813.bound (LeftBound93813.actual selector witness) := by
  exact .transfer (LeftBound93813.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound93813.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound93813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound93813.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93814

namespace LeftBound93909
def owner : Owner := ⟨.program ⟨214⟩, ⟨14793⟩⟩
def transferEvent : Nat := 93909
def frameStart : Nat := 93870
def rule : BoundRule := .identity (.predecessor 0 93908 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93908 .coefficient)
      LeftAuthority93906.bound (LeftAuthority93906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93906.derived selector witness)

def rawBound : CoeffClass := LeftAuthority93906.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority93906.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93909

namespace LeftBound93926
def owner : Owner := ⟨.program ⟨214⟩, ⟨14832⟩⟩
def transferEvent : Nat := 93926
def frameStart : Nat := 93870
def rule : BoundRule := .sum [.predecessor 0 93924 .coefficient, .predecessor 1 93925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93924 .coefficient)
      LeftBound93909.bound (LeftBound93909.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93925 .coefficient)
      LeftAuthority93922.bound (LeftAuthority93922.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority93922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93909.bound, LeftAuthority93922.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93909.bound, LeftAuthority93922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93909.actual selector witness, LeftAuthority93922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93926

namespace LeftBound93929
def owner : Owner := ⟨.program ⟨214⟩, ⟨14833⟩⟩
def transferEvent : Nat := 93929
def frameStart : Nat := 93870
def rule : BoundRule := .identity (.predecessor 0 93928 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93928 .coefficient)
      LeftBound93926.bound (LeftBound93926.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93926.derived selector witness)

def rawBound : CoeffClass := LeftBound93926.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound93926.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93929

namespace LeftBound93935
def owner : Owner := ⟨.program ⟨214⟩, ⟨14834⟩⟩
def transferEvent : Nat := 93935
def frameStart : Nat := 93870
def rule : BoundRule := .product (.predecessor 0 93933 .coefficient) (.predecessor 1 93934 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93933 .coefficient)
      LeftAuthority93931.bound (LeftAuthority93931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93934 .coefficient)
      LeftBound93929.bound (LeftBound93929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93929.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority93931.bound LeftBound93929.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93931.bound, LeftBound93929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority93931.actual selector witness) * (LeftBound93929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93935

namespace LeftBound93943
def owner : Owner := ⟨.program ⟨214⟩, ⟨14835⟩⟩
def transferEvent : Nat := 93943
def frameStart : Nat := 93870
def rule : BoundRule := .sum [.predecessor 0 93941 .coefficient, .predecessor 1 93942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93941 .coefficient)
      LeftAuthority93939.bound (LeftAuthority93939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93942 .coefficient)
      LeftBound93935.bound (LeftBound93935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93939.bound, LeftBound93935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93939.bound, LeftBound93935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93939.actual selector witness, LeftBound93935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93943

namespace LeftBound93947
def owner : Owner := ⟨.program ⟨214⟩, ⟨26352⟩⟩
def transferEvent : Nat := 93947
def frameStart : Nat := 93870
def rule : BoundRule := .product (.predecessor 0 93945 .coefficient) (.predecessor 1 93946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93945 .coefficient)
      LeftBound93943.bound (LeftBound93943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93946 .coefficient)
      LeftAuthority93920.bound (LeftAuthority93920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93920.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93943.bound LeftAuthority93920.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93943.bound, LeftAuthority93920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93943.actual selector witness) * (LeftAuthority93920.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93947

namespace LeftBound93958
def owner : Owner := ⟨.program ⟨214⟩, ⟨14889⟩⟩
def transferEvent : Nat := 93958
def frameStart : Nat := 93870
def rule : BoundRule := .product (.predecessor 0 93956 .coefficient) (.predecessor 1 93957 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93956 .coefficient)
      LeftAuthority93931.bound (LeftAuthority93931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93957 .coefficient)
      LeftAuthority93954.bound (LeftAuthority93954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93954.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93954.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority93931.bound LeftAuthority93954.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93931.bound, LeftAuthority93954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority93931.actual selector witness) * (LeftAuthority93954.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93958

namespace LeftBound93966
def owner : Owner := ⟨.program ⟨214⟩, ⟨14890⟩⟩
def transferEvent : Nat := 93966
def frameStart : Nat := 93870
def rule : BoundRule := .sum [.predecessor 0 93964 .coefficient, .predecessor 1 93965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93964 .coefficient)
      LeftAuthority93962.bound (LeftAuthority93962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93962.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93965 .coefficient)
      LeftBound93958.bound (LeftBound93958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93958.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93962.bound, LeftBound93958.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93962.bound, LeftBound93958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93962.actual selector witness, LeftBound93958.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93966

namespace LeftBound93970
def owner : Owner := ⟨.program ⟨214⟩, ⟨26357⟩⟩
def transferEvent : Nat := 93970
def frameStart : Nat := 93870
def rule : BoundRule := .sum [.predecessor 0 93968 .coefficient, .predecessor 1 93969 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93968 .coefficient)
      LeftBound93966.bound (LeftBound93966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93969 .coefficient)
      LeftBound93947.bound (LeftBound93947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93966.bound, LeftBound93947.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93966.bound, LeftBound93947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93966.actual selector witness, LeftBound93947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93970

namespace LeftBound93983
def owner : Owner := ⟨.program ⟨214⟩, ⟨26354⟩⟩
def transferEvent : Nat := 93983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 93981 .coefficient, .predecessor 1 93982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93981 .coefficient)
      LeftBound93812.bound (LeftBound93812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93982 .coefficient)
      LeftBound93795.bound (LeftBound93795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93795.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93812.bound, LeftBound93795.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93812.bound, LeftBound93795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93812.actual selector witness, LeftBound93795.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93983

namespace LeftBound93986
def owner : Owner := ⟨.program ⟨214⟩, ⟨26354⟩⟩
def transferEvent : Nat := 93986
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 93980 .summary, .result 93802 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93980 .summary)
      LeftBound93814.bound (LeftBound93814.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20323⟩⟩) (rawTerms := some (Proof.Events367.exact93980RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93802 .summary)
      LeftBound93797.bound (LeftBound93797.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26353⟩⟩) (rawTerms := some (Proof.Events366.exact93802RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93797.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93814.bound, LeftBound93797.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93814.bound, LeftBound93797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93814.actual selector witness, LeftBound93797.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93986

namespace LeftBound93990
def owner : Owner := ⟨.program ⟨214⟩, ⟨26355⟩⟩
def transferEvent : Nat := 93990
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93988 .coefficient) (.predecessor 1 93989 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93988 .coefficient)
      LeftBound93983.bound (LeftBound93983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93989 .coefficient)
      LeftBound5858.bound (LeftBound5858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93983.bound LeftBound5858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93983.bound, LeftBound5858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93983.actual selector witness) * (LeftBound5858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93990

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
