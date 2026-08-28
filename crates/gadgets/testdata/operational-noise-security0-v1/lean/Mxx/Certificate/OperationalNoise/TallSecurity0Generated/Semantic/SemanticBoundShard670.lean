import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard669

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound97904
def owner : Owner := ⟨.program ⟨214⟩, ⟨14621⟩⟩
def transferEvent : Nat := 97904
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97902 .coefficient) (.predecessor 1 97903 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97902 .coefficient)
      LeftBound97898.bound (LeftBound97898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97903 .coefficient)
      LeftBound10509.bound (LeftBound10509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10509.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97898.bound LeftBound10509.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97898.bound, LeftBound10509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97898.actual selector witness) * (LeftBound10509.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97904

namespace LeftBound97905
def owner : Owner := ⟨.program ⟨214⟩, ⟨14621⟩⟩
def transferEvent : Nat := 97905
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩ [⟨.result 10506 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10506 .coefficient)
      LeftAuthority10505.bound (LeftAuthority10505.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7858⟩⟩) (rawTerms := some (Proof.Events041.exact10506RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10505.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10505.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10505.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97905

namespace LeftBound97906
def owner : Owner := ⟨.program ⟨214⟩, ⟨14621⟩⟩
def transferEvent : Nat := 97906
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97901 .summary) (.transfer 97905) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97901 .summary)
      LeftBound97899.bound (LeftBound97899.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14620⟩⟩) (rawTerms := some (Proof.Events382.exact97901RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97905)
      LeftBound97905.bound (LeftBound97905.actual selector witness) := by
  exact .transfer (LeftBound97905.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97899.bound LeftBound97905.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97899.bound, LeftBound97905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97899.actual selector witness) * (LeftBound97905.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97906

namespace LeftBound97914
def owner : Owner := ⟨.program ⟨214⟩, ⟨14622⟩⟩
def transferEvent : Nat := 97914
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97912 .coefficient, .predecessor 1 97913 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97912 .coefficient)
      LeftBound97904.bound (LeftBound97904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97913 .coefficient)
      LeftBound97876.bound (LeftBound97876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97876.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97904.bound, LeftBound97876.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97904.bound, LeftBound97876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97904.actual selector witness, LeftBound97876.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97914

namespace LeftBound97916
def owner : Owner := ⟨.program ⟨214⟩, ⟨14622⟩⟩
def transferEvent : Nat := 97916
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 97911 .summary, .result 97881 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97911 .summary)
      LeftBound97906.bound (LeftBound97906.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14621⟩⟩) (rawTerms := some (Proof.Events382.exact97911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97881 .summary)
      LeftBound97878.bound (LeftBound97878.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14617⟩⟩) (rawTerms := some (Proof.Events382.exact97881RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97906.bound, LeftBound97878.bound]
def bound : CoeffClass := .finite ⟨95443712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97906.bound, LeftBound97878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97906.actual selector witness, LeftBound97878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97916

namespace LeftBound97920
def owner : Owner := ⟨.program ⟨214⟩, ⟨26208⟩⟩
def transferEvent : Nat := 97920
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97918 .coefficient) (.predecessor 1 97919 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97918 .coefficient)
      LeftBound97914.bound (LeftBound97914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97919 .coefficient)
      LeftAuthority97852.bound (LeftAuthority97852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97852.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97914.bound LeftAuthority97852.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97914.bound, LeftAuthority97852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97914.actual selector witness) * (LeftAuthority97852.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97920

namespace LeftBound97921
def owner : Owner := ⟨.program ⟨214⟩, ⟨26208⟩⟩
def transferEvent : Nat := 97921
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩ [⟨.result 97853 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97853 .coefficient)
      LeftAuthority97852.bound (LeftAuthority97852.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26207⟩⟩) (rawTerms := some (Proof.Events382.exact97853RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97852.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority97852.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97852.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97921

namespace LeftBound97922
def owner : Owner := ⟨.program ⟨214⟩, ⟨26208⟩⟩
def transferEvent : Nat := 97922
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97917 .summary) (.transfer 97921) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97917 .summary)
      LeftBound97916.bound (LeftBound97916.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14622⟩⟩) (rawTerms := some (Proof.Events382.exact97917RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97916.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97921)
      LeftBound97921.bound (LeftBound97921.actual selector witness) := by
  exact .transfer (LeftBound97921.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97916.bound LeftBound97921.bound
def bound : CoeffClass := .finite ⟨350279950139392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97916.bound, LeftBound97921.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97916.actual selector witness) * (LeftBound97921.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97922

namespace LeftBound97933
def owner : Owner := ⟨.program ⟨214⟩, ⟨19663⟩⟩
def transferEvent : Nat := 97933
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 97931 .coefficient) (.value (.predecessor 1 97932 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97931 .coefficient)
      LeftAuthority97929.bound (LeftAuthority97929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97932 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority97929.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97929.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97929.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97933

namespace LeftBound97937
def owner : Owner := ⟨.program ⟨214⟩, ⟨19664⟩⟩
def transferEvent : Nat := 97937
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97935 .coefficient) (.predecessor 1 97936 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97935 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97936 .coefficient)
      LeftBound97933.bound (LeftBound97933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97933.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97933.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound97933.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound97933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound97933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97937

namespace LeftBound97938
def owner : Owner := ⟨.program ⟨214⟩, ⟨19664⟩⟩
def transferEvent : Nat := 97938
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩ [⟨.result 97930 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97930 .coefficient)
      LeftAuthority97929.bound (LeftAuthority97929.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19661⟩⟩) (rawTerms := some (Proof.Events382.exact97930RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97929.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority97929.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97929.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97938

namespace LeftBound97939
def owner : Owner := ⟨.program ⟨214⟩, ⟨19664⟩⟩
def transferEvent : Nat := 97939
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 97938) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97938)
      LeftBound97938.bound (LeftBound97938.actual selector witness) := by
  exact .transfer (LeftBound97938.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound97938.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound97938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound97938.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97939

namespace LeftBound97994
def owner : Owner := ⟨.program ⟨214⟩, ⟨14615⟩⟩
def transferEvent : Nat := 97994
def frameStart : Nat := 97977
def rule : BoundRule := .product (.predecessor 0 97992 .coefficient) (.predecessor 1 97993 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97992 .coefficient)
      LeftAuthority97990.bound (LeftAuthority97990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97993 .coefficient)
      LeftAuthority97987.bound (LeftAuthority97987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97987.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97987.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority97990.bound LeftAuthority97987.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97990.bound, LeftAuthority97987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority97990.actual selector witness) * (LeftAuthority97987.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97994

namespace LeftBound97998
def owner : Owner := ⟨.program ⟨214⟩, ⟨14616⟩⟩
def transferEvent : Nat := 97998
def frameStart : Nat := 97977
def rule : BoundRule := .identity (.predecessor 0 97997 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97997 .coefficient)
      LeftBound97994.bound (LeftBound97994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97994.derived selector witness)

def rawBound : CoeffClass := LeftBound97994.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound97994.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97998

namespace LeftBound98015
def owner : Owner := ⟨.program ⟨214⟩, ⟨14740⟩⟩
def transferEvent : Nat := 98015
def frameStart : Nat := 97977
def rule : BoundRule := .sum [.predecessor 0 98013 .coefficient, .predecessor 1 98014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98013 .coefficient)
      LeftBound97998.bound (LeftBound97998.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98014 .coefficient)
      LeftAuthority98011.bound (LeftAuthority98011.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97998.bound, LeftAuthority98011.bound]
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97998.bound, LeftAuthority98011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97998.actual selector witness, LeftAuthority98011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98015

namespace LeftBound98018
def owner : Owner := ⟨.program ⟨214⟩, ⟨14741⟩⟩
def transferEvent : Nat := 98018
def frameStart : Nat := 97977
def rule : BoundRule := .identity (.predecessor 0 98017 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98017 .coefficient)
      LeftBound98015.bound (LeftBound98015.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98015.derived selector witness)

def rawBound : CoeffClass := LeftBound98015.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound98015.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98018

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
