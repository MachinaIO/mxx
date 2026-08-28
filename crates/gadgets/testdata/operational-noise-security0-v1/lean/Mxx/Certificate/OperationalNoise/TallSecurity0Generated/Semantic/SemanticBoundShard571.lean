import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard570

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound83794
def owner : Owner := ⟨.program ⟨214⟩, ⟨14644⟩⟩
def transferEvent : Nat := 83794
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83792 .coefficient) (.predecessor 1 83793 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83792 .coefficient)
      LeftBound83788.bound (LeftBound83788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83791RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83788.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83788.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83793 .coefficient)
      LeftAuthority4014.bound (LeftAuthority4014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4014.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound83788.bound LeftAuthority4014.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83788.bound, LeftAuthority4014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound83788.actual selector witness) * (LeftAuthority4014.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83794

namespace LeftBound83795
def owner : Owner := ⟨.program ⟨214⟩, ⟨14644⟩⟩
def transferEvent : Nat := 83795
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩ [⟨.result 4015 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4015 .coefficient)
      LeftAuthority4014.bound (LeftAuthority4014.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14641⟩⟩) (rawTerms := some (Proof.Events015.exact4015RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4014.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4014.bound []
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4014.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83795

namespace LeftBound83796
def owner : Owner := ⟨.program ⟨214⟩, ⟨14644⟩⟩
def transferEvent : Nat := 83796
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83791 .summary) (.transfer 83795) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83791 .summary)
      LeftBound83789.bound (LeftBound83789.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11640⟩⟩) (rawTerms := some (Proof.Events327.exact83791RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83795)
      LeftBound83795.bound (LeftBound83795.actual selector witness) := by
  exact .transfer (LeftBound83795.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound83789.bound LeftBound83795.bound
def bound : CoeffClass := .finite ⟨23296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83789.bound, LeftBound83795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound83789.actual selector witness) * (LeftBound83795.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83796

namespace LeftBound83802
def owner : Owner := ⟨.program ⟨214⟩, ⟨14645⟩⟩
def transferEvent : Nat := 83802
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 83800 .coefficient) (.predecessor 1 83801 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83800 .coefficient)
      LeftAuthority4014.bound (LeftAuthority4014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83801 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4014.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4014.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4014.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83802

namespace LeftBound83807
def owner : Owner := ⟨.program ⟨214⟩, ⟨7218⟩⟩
def transferEvent : Nat := 83807
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83805 .coefficient) (.predecessor 1 83806 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83805 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83806 .coefficient)
      LeftBound10520.bound (LeftBound10520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound10520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound10520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound10520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83807

namespace LeftBound83812
def owner : Owner := ⟨.program ⟨214⟩, ⟨14646⟩⟩
def transferEvent : Nat := 83812
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83810 .coefficient, .predecessor 1 83811 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83810 .coefficient)
      LeftBound83807.bound (LeftBound83807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83807.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83811 .coefficient)
      LeftBound83802.bound (LeftBound83802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83807.bound, LeftBound83802.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83807.bound, LeftBound83802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83807.actual selector witness, LeftBound83802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83812

namespace LeftBound83816
def owner : Owner := ⟨.program ⟨214⟩, ⟨14647⟩⟩
def transferEvent : Nat := 83816
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83814 .coefficient, .predecessor 1 83815 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83814 .coefficient)
      LeftBound83812.bound (LeftBound83812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83815 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83812.bound, LeftBound10512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83812.bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83812.actual selector witness, LeftBound10512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83816

namespace LeftBound83817
def owner : Owner := ⟨.program ⟨214⟩, ⟨14647⟩⟩
def transferEvent : Nat := 83817
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩ [⟨.result 10513 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10513 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨76⟩⟩) (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10512.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10512.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83817

namespace LeftBound83822
def owner : Owner := ⟨.program ⟨214⟩, ⟨14648⟩⟩
def transferEvent : Nat := 83822
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83820 .coefficient) (.predecessor 1 83821 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83820 .coefficient)
      LeftBound83816.bound (LeftBound83816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83821 .coefficient)
      LeftBound10509.bound (LeftBound10509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10509.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83816.bound LeftBound10509.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83816.bound, LeftBound10509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83816.actual selector witness) * (LeftBound10509.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83822

namespace LeftBound83823
def owner : Owner := ⟨.program ⟨214⟩, ⟨14648⟩⟩
def transferEvent : Nat := 83823
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
end LeftBound83823

namespace LeftBound83824
def owner : Owner := ⟨.program ⟨214⟩, ⟨14648⟩⟩
def transferEvent : Nat := 83824
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83819 .summary) (.transfer 83823) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83819 .summary)
      LeftBound83817.bound (LeftBound83817.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14647⟩⟩) (rawTerms := some (Proof.Events327.exact83819RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83823)
      LeftBound83823.bound (LeftBound83823.actual selector witness) := by
  exact .transfer (LeftBound83823.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83817.bound LeftBound83823.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83817.bound, LeftBound83823.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83817.actual selector witness) * (LeftBound83823.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83824

namespace LeftBound83832
def owner : Owner := ⟨.program ⟨214⟩, ⟨14649⟩⟩
def transferEvent : Nat := 83832
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83830 .coefficient, .predecessor 1 83831 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83830 .coefficient)
      LeftBound83822.bound (LeftBound83822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83822.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83831 .coefficient)
      LeftBound83794.bound (LeftBound83794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83794.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83794.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83822.bound, LeftBound83794.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83822.bound, LeftBound83794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83822.actual selector witness, LeftBound83794.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83832

namespace LeftBound83834
def owner : Owner := ⟨.program ⟨214⟩, ⟨14649⟩⟩
def transferEvent : Nat := 83834
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 83829 .summary, .result 83799 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83829 .summary)
      LeftBound83824.bound (LeftBound83824.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14648⟩⟩) (rawTerms := some (Proof.Events327.exact83829RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83799 .summary)
      LeftBound83796.bound (LeftBound83796.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14644⟩⟩) (rawTerms := some (Proof.Events327.exact83799RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83796.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83824.bound, LeftBound83796.bound]
def bound : CoeffClass := .finite ⟨95443712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83824.bound, LeftBound83796.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83824.actual selector witness, LeftBound83796.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83834

namespace LeftBound83838
def owner : Owner := ⟨.program ⟨214⟩, ⟨26221⟩⟩
def transferEvent : Nat := 83838
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83836 .coefficient) (.predecessor 1 83837 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83836 .coefficient)
      LeftBound83832.bound (LeftBound83832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83832.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83837 .coefficient)
      LeftAuthority83770.bound (LeftAuthority83770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83770.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83770.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83832.bound LeftAuthority83770.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83832.bound, LeftAuthority83770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83832.actual selector witness) * (LeftAuthority83770.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83838

namespace LeftBound83839
def owner : Owner := ⟨.program ⟨214⟩, ⟨26221⟩⟩
def transferEvent : Nat := 83839
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩ [⟨.result 83771 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83771 .coefficient)
      LeftAuthority83770.bound (LeftAuthority83770.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26220⟩⟩) (rawTerms := some (Proof.Events327.exact83771RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83770.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83770.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority83770.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83770.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83839

namespace LeftBound83840
def owner : Owner := ⟨.program ⟨214⟩, ⟨26221⟩⟩
def transferEvent : Nat := 83840
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83835 .summary) (.transfer 83839) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83835 .summary)
      LeftBound83834.bound (LeftBound83834.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14649⟩⟩) (rawTerms := some (Proof.Events327.exact83835RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83834.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83839)
      LeftBound83839.bound (LeftBound83839.actual selector witness) := by
  exact .transfer (LeftBound83839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83834.bound LeftBound83839.bound
def bound : CoeffClass := .finite ⟨350279950139392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83834.bound, LeftBound83839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83834.actual selector witness) * (LeftBound83839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83840

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
