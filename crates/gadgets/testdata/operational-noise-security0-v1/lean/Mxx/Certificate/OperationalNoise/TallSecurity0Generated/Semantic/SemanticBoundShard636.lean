import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard128
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard632
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard633
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard634
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard635

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound93991
def owner : Owner := ⟨.program ⟨214⟩, ⟨26355⟩⟩
def transferEvent : Nat := 93991
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
end LeftBound93991

namespace LeftBound93992
def owner : Owner := ⟨.program ⟨214⟩, ⟨26355⟩⟩
def transferEvent : Nat := 93992
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 93987 .summary) (.transfer 93991) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93987 .summary)
      LeftBound93986.bound (LeftBound93986.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26354⟩⟩) (rawTerms := some (Proof.Events367.exact93987RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93991)
      LeftBound93991.bound (LeftBound93991.actual selector witness) := by
  exact .transfer (LeftBound93991.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93986.bound LeftBound93991.bound
def bound : CoeffClass := .finite ⟨4741253940199267499646124032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93986.bound, LeftBound93991.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93986.actual selector witness) * (LeftBound93991.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93992

namespace LeftBound94000
def owner : Owner := ⟨.program ⟨214⟩, ⟨6626⟩⟩
def transferEvent : Nat := 94000
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 93998 .coefficient) (.predecessor 1 93999 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93998 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93999 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound94000

namespace LeftBound94005
def owner : Owner := ⟨.program ⟨214⟩, ⟨7216⟩⟩
def transferEvent : Nat := 94005
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94003 .coefficient) (.predecessor 1 94004 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94003 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94004 .coefficient)
      LeftBound5872.bound (LeftBound5872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5872.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound5872.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound5872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound5872.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94005

namespace LeftBound94010
def owner : Owner := ⟨.program ⟨214⟩, ⟨7753⟩⟩
def transferEvent : Nat := 94010
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94008 .coefficient, .predecessor 1 94009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94008 .coefficient)
      LeftBound94005.bound (LeftBound94005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94005.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94009 .coefficient)
      LeftBound94000.bound (LeftBound94000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94000.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94005.bound, LeftBound94000.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94005.bound, LeftBound94000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94005.actual selector witness, LeftBound94000.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94010

namespace LeftBound94014
def owner : Owner := ⟨.program ⟨214⟩, ⟨7754⟩⟩
def transferEvent : Nat := 94014
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94012 .coefficient, .predecessor 1 94013 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94012 .coefficient)
      LeftBound94010.bound (LeftBound94010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94013 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94010.bound, LeftBound20907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94010.bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94010.actual selector witness, LeftBound20907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94014

namespace LeftBound94015
def owner : Owner := ⟨.program ⟨214⟩, ⟨7754⟩⟩
def transferEvent : Nat := 94015
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
end LeftBound94015

namespace LeftBound94020
def owner : Owner := ⟨.program ⟨214⟩, ⟨7808⟩⟩
def transferEvent : Nat := 94020
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94018 .coefficient, .predecessor 1 94019 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94018 .coefficient)
      LeftBound94014.bound (LeftBound94014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94019 .coefficient)
      LeftBound94014.bound (LeftBound94014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94014.bound, LeftBound94014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94014.bound, LeftBound94014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94014.actual selector witness, LeftBound94014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94020

namespace LeftBound94023
def owner : Owner := ⟨.program ⟨214⟩, ⟨7808⟩⟩
def transferEvent : Nat := 94023
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94017 .summary, .result 94017 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94017 .summary)
      LeftBound94015.bound (LeftBound94015.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7754⟩⟩) (rawTerms := some (Proof.Events367.exact94017RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94017 .summary)
      LeftBound94015.bound (LeftBound94015.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7754⟩⟩) (rawTerms := some (Proof.Events367.exact94017RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94015.bound, LeftBound94015.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94015.bound, LeftBound94015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94015.actual selector witness, LeftBound94015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94023

namespace LeftBound94027
def owner : Owner := ⟨.program ⟨214⟩, ⟨26356⟩⟩
def transferEvent : Nat := 94027
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94025 .coefficient, .predecessor 1 94026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94025 .coefficient)
      LeftBound94020.bound (LeftBound94020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94026 .coefficient)
      LeftBound93990.bound (LeftBound93990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93990.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94020.bound, LeftBound93990.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94020.bound, LeftBound93990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94020.actual selector witness, LeftBound93990.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94027

namespace LeftBound94028
def owner : Owner := ⟨.program ⟨214⟩, ⟨26356⟩⟩
def transferEvent : Nat := 94028
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94024 .summary, .result 93997 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94024 .summary)
      LeftBound94023.bound (LeftBound94023.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7808⟩⟩) (rawTerms := some (Proof.Events367.exact94024RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93997 .summary)
      LeftBound93992.bound (LeftBound93992.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26355⟩⟩) (rawTerms := some (Proof.Events367.exact93997RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93992.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94023.bound, LeftBound93992.bound]
def bound : CoeffClass := .finite ⟨4741253940199267499646124084, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94023.bound, LeftBound93992.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94023.actual selector witness, LeftBound93992.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94028

namespace LeftBound94032
def owner : Owner := ⟨.program ⟨214⟩, ⟨26562⟩⟩
def transferEvent : Nat := 94032
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94030 .coefficient, .predecessor 1 94031 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94030 .coefficient)
      LeftBound94027.bound (LeftBound94027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94031 .coefficient)
      LeftBound93778.bound (LeftBound93778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93778.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94027.bound, LeftBound93778.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94027.bound, LeftBound93778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94027.actual selector witness, LeftBound93778.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94032

namespace LeftBound94033
def owner : Owner := ⟨.program ⟨214⟩, ⟨26562⟩⟩
def transferEvent : Nat := 94033
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94029 .summary, .result 93785 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94029 .summary)
      LeftBound94028.bound (LeftBound94028.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26356⟩⟩) (rawTerms := some (Proof.Events367.exact94029RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93785 .summary)
      LeftBound93780.bound (LeftBound93780.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26561⟩⟩) (rawTerms := some (Proof.Events366.exact93785RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93780.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94028.bound, LeftBound93780.bound]
def bound : CoeffClass := .finite ⟨9482549007414447334737575988, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94028.bound, LeftBound93780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94028.actual selector witness, LeftBound93780.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94033

namespace LeftBound94037
def owner : Owner := ⟨.program ⟨214⟩, ⟨26779⟩⟩
def transferEvent : Nat := 94037
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94035 .coefficient, .predecessor 1 94036 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94035 .coefficient)
      LeftBound94032.bound (LeftBound94032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94036 .coefficient)
      LeftBound93566.bound (LeftBound93566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94032.bound, LeftBound93566.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94032.bound, LeftBound93566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94032.actual selector witness, LeftBound93566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94037

namespace LeftBound94038
def owner : Owner := ⟨.program ⟨214⟩, ⟨26779⟩⟩
def transferEvent : Nat := 94038
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94034 .summary, .result 93573 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94034 .summary)
      LeftBound94033.bound (LeftBound94033.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26562⟩⟩) (rawTerms := some (Proof.Events367.exact94034RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93573 .summary)
      LeftBound93568.bound (LeftBound93568.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26778⟩⟩) (rawTerms := some (Proof.Events365.exact93573RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94033.bound, LeftBound93568.bound]
def bound : CoeffClass := .finite ⟨14223885201645539505274355764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94033.bound, LeftBound93568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94033.actual selector witness, LeftBound93568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94038

namespace LeftBound94042
def owner : Owner := ⟨.program ⟨214⟩, ⟨26996⟩⟩
def transferEvent : Nat := 94042
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94040 .coefficient, .predecessor 1 94041 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94040 .coefficient)
      LeftBound94037.bound (LeftBound94037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94037.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94041 .coefficient)
      LeftBound93354.bound (LeftBound93354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94037.bound, LeftBound93354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94037.bound, LeftBound93354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94037.actual selector witness, LeftBound93354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94042

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
