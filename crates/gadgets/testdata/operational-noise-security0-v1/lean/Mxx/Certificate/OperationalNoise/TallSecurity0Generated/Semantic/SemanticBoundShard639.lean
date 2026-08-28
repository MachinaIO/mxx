import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard022
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard024
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard539
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard638

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound94129
def owner : Owner := ⟨.program ⟨214⟩, ⟨7729⟩⟩
def transferEvent : Nat := 94129
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94127 .coefficient, .predecessor 1 94128 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94127 .coefficient)
      LeftBound94124.bound (LeftBound94124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94128 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94124.bound, LeftBound79918.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94124.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94124.actual selector witness, LeftBound79918.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94129

namespace LeftBound94133
def owner : Owner := ⟨.program ⟨214⟩, ⟨7730⟩⟩
def transferEvent : Nat := 94133
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94131 .coefficient, .predecessor 1 94132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94131 .coefficient)
      LeftBound94129.bound (LeftBound94129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94132 .coefficient)
      LeftAuthority94120.bound (LeftAuthority94120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94129.bound, LeftAuthority94120.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94129.bound, LeftAuthority94120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94129.actual selector witness, LeftAuthority94120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94133

namespace LeftBound94134
def owner : Owner := ⟨.program ⟨214⟩, ⟨7730⟩⟩
def transferEvent : Nat := 94134
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨33⟩⟩]⟩ [⟨.result 94121 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94121 .coefficient)
      LeftAuthority94120.bound (LeftAuthority94120.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨33⟩⟩) (rawTerms := some (Proof.Events367.exact94121RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94120.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority94120.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94120.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94134

namespace LeftBound94139
def owner : Owner := ⟨.program ⟨214⟩, ⟨7904⟩⟩
def transferEvent : Nat := 94139
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94137 .coefficient) (.predecessor 1 94138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94137 .coefficient)
      LeftBound94133.bound (LeftBound94133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94138 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94133.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94133.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94133.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94139

namespace LeftBound94140
def owner : Owner := ⟨.program ⟨214⟩, ⟨7904⟩⟩
def transferEvent : Nat := 94140
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩ [⟨.result 5957 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5957 .coefficient)
      LeftAuthority5956.bound (LeftAuthority5956.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7885⟩⟩) (rawTerms := some (Proof.Events023.exact5957RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5956.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5956.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5956.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94140

namespace LeftBound94141
def owner : Owner := ⟨.program ⟨214⟩, ⟨7904⟩⟩
def transferEvent : Nat := 94141
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94136 .summary) (.transfer 94140) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94136 .summary)
      LeftBound94134.bound (LeftBound94134.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7730⟩⟩) (rawTerms := some (Proof.Events367.exact94136RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94140)
      LeftBound94140.bound (LeftBound94140.actual selector witness) := by
  exact .transfer (LeftBound94140.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94134.bound LeftBound94140.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94134.bound, LeftBound94140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94134.actual selector witness) * (LeftBound94140.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94141

namespace LeftBound94167
def owner : Owner := ⟨.program ⟨214⟩, ⟨30126⟩⟩
def transferEvent : Nat := 94167
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94165 .coefficient, .predecessor 1 94166 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94165 .coefficient)
      LeftBound94139.bound (LeftBound94139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94166 .coefficient)
      LeftBound94117.bound (LeftBound94117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94117.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94139.bound, LeftBound94117.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94139.bound, LeftBound94117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94139.actual selector witness, LeftBound94117.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94167

namespace LeftBound94187
def owner : Owner := ⟨.program ⟨214⟩, ⟨30126⟩⟩
def transferEvent : Nat := 94187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94164 .summary, .result 94119 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94164 .summary)
      LeftBound94141.bound (LeftBound94141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7904⟩⟩) (rawTerms := some (Proof.Events367.exact94164RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94119 .summary)
      LeftBound94118.bound (LeftBound94118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30125⟩⟩) (rawTerms := some (Proof.Events367.exact94119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94141.bound, LeftBound94118.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789483581492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94141.bound, LeftBound94118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94141.actual selector witness, LeftBound94118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94187

namespace LeftBound94191
def owner : Owner := ⟨.program ⟨214⟩, ⟨30127⟩⟩
def transferEvent : Nat := 94191
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94189 .coefficient) (.predecessor 1 94190 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94189 .coefficient)
      LeftBound94167.bound (LeftBound94167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94167.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94190 .coefficient)
      LeftBound6160.bound (LeftBound6160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6160.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94167.bound LeftBound6160.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94167.bound, LeftBound6160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94167.actual selector witness) * (LeftBound6160.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94191

namespace LeftBound94192
def owner : Owner := ⟨.program ⟨214⟩, ⟨30127⟩⟩
def transferEvent : Nat := 94192
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7829⟩⟩]⟩ [⟨.result 6157 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6157 .coefficient)
      LeftAuthority6156.bound (LeftAuthority6156.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7829⟩⟩) (rawTerms := some (Proof.Events024.exact6157RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6156.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6156.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6156.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94192

namespace LeftBound94193
def owner : Owner := ⟨.program ⟨214⟩, ⟨30127⟩⟩
def transferEvent : Nat := 94193
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94188 .summary) (.transfer 94192) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94188 .summary)
      LeftBound94187.bound (LeftBound94187.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30126⟩⟩) (rawTerms := some (Proof.Events367.exact94188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94192)
      LeftBound94192.bound (LeftBound94192.actual selector witness) := by
  exact .transfer (LeftBound94192.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94187.bound LeftBound94192.bound
def bound : CoeffClass := .finite ⟨1149729608724517268372876178953375812943872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94187.bound, LeftBound94192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94187.actual selector witness) * (LeftBound94192.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94193

namespace LeftBound94255
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def transferEvent : Nat := 94255
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94253 .coefficient, .predecessor 1 94254 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94253 .coefficient)
      LeftBound94191.bound (LeftBound94191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94254 .coefficient)
      LeftBound79808.bound (LeftBound79808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94191.bound, LeftBound79808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94191.bound, LeftBound79808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94191.actual selector witness, LeftBound79808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94255

namespace LeftBound94275
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def transferEvent : Nat := 94275
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94252 .summary, .result 79885 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94252 .summary)
      LeftBound94193.bound (LeftBound94193.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30127⟩⟩) (rawTerms := some (Proof.Events368.exact94252RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79885 .summary)
      LeftBound79846.bound (LeftBound79846.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18846⟩⟩) (rawTerms := some (Proof.Events312.exact79885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79846.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94193.bound, LeftBound79846.bound]
def bound : CoeffClass := .finite ⟨1149729608724524008718218297164355856419136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94193.bound, LeftBound79846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94193.actual selector witness, LeftBound79846.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94275

namespace LeftBound94279
def owner : Owner := ⟨.program ⟨214⟩, ⟨30129⟩⟩
def transferEvent : Nat := 94279
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94277 .coefficient) (.predecessor 1 94278 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94277 .coefficient)
      LeftBound94255.bound (LeftBound94255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94278 .coefficient)
      LeftBound6150.bound (LeftBound6150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6150.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94255.bound LeftBound6150.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94255.bound, LeftBound6150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94255.actual selector witness) * (LeftBound6150.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94279

namespace LeftBound94280
def owner : Owner := ⟨.program ⟨214⟩, ⟨30129⟩⟩
def transferEvent : Nat := 94280
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩ [⟨.result 6147 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6147 .coefficient)
      LeftAuthority6146.bound (LeftAuthority6146.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6683⟩⟩) (rawTerms := some (Proof.Events024.exact6147RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6146.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6146.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6146.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94280

namespace LeftBound94281
def owner : Owner := ⟨.program ⟨214⟩, ⟨30129⟩⟩
def transferEvent : Nat := 94281
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94276 .summary) (.transfer 94280) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94276 .summary)
      LeftBound94275.bound (LeftBound94275.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30128⟩⟩) (rawTerms := some (Proof.Events368.exact94276RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94275.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94280)
      LeftBound94280.bound (LeftBound94280.actual selector witness) := by
  exact .transfer (LeftBound94280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94275.bound LeftBound94280.bound
def bound : CoeffClass := .finite ⟨4219526059692742704380000642085940622751931826176, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94275.bound, LeftBound94280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94275.actual selector witness) * (LeftBound94280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94281

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
