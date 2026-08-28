import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard218
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard219
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard221
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard222
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard223
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard224
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard225
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard226
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard227
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard229
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard230

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35569
def owner : Owner := ⟨.program ⟨214⟩, ⟨26601⟩⟩
def transferEvent : Nat := 35569
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35565 .summary, .result 35321 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35565 .summary)
      LeftBound35564.bound (LeftBound35564.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26392⟩⟩) (rawTerms := some (Proof.Events138.exact35565RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35321 .summary)
      LeftBound35316.bound (LeftBound35316.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26600⟩⟩) (rawTerms := some (Proof.Events137.exact35321RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35316.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35564.bound, LeftBound35316.bound]
def bound : CoeffClass := .finite ⟨9482549007414447334737575988, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35564.bound, LeftBound35316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35564.actual selector witness, LeftBound35316.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35569

namespace LeftBound35573
def owner : Owner := ⟨.program ⟨214⟩, ⟨26818⟩⟩
def transferEvent : Nat := 35573
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35571 .coefficient, .predecessor 1 35572 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35571 .coefficient)
      LeftBound35568.bound (LeftBound35568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35572 .coefficient)
      LeftBound35102.bound (LeftBound35102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35568.bound, LeftBound35102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35568.bound, LeftBound35102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35568.actual selector witness, LeftBound35102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35573

namespace LeftBound35574
def owner : Owner := ⟨.program ⟨214⟩, ⟨26818⟩⟩
def transferEvent : Nat := 35574
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35570 .summary, .result 35109 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35570 .summary)
      LeftBound35569.bound (LeftBound35569.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26601⟩⟩) (rawTerms := some (Proof.Events138.exact35570RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35109 .summary)
      LeftBound35104.bound (LeftBound35104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26817⟩⟩) (rawTerms := some (Proof.Events137.exact35109RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35104.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35569.bound, LeftBound35104.bound]
def bound : CoeffClass := .finite ⟨14223885201645539505274355764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35569.bound, LeftBound35104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35569.actual selector witness, LeftBound35104.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35574

namespace LeftBound35578
def owner : Owner := ⟨.program ⟨214⟩, ⟨27035⟩⟩
def transferEvent : Nat := 35578
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35576 .coefficient, .predecessor 1 35577 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35576 .coefficient)
      LeftBound35573.bound (LeftBound35573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35577 .coefficient)
      LeftBound34890.bound (LeftBound34890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35573.bound, LeftBound34890.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35573.bound, LeftBound34890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35573.actual selector witness, LeftBound34890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35578

namespace LeftBound35579
def owner : Owner := ⟨.program ⟨214⟩, ⟨27035⟩⟩
def transferEvent : Nat := 35579
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35575 .summary, .result 34897 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35575 .summary)
      LeftBound35574.bound (LeftBound35574.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26818⟩⟩) (rawTerms := some (Proof.Events138.exact35575RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34897 .summary)
      LeftBound34892.bound (LeftBound34892.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27034⟩⟩) (rawTerms := some (Proof.Events136.exact34897RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35574.bound, LeftBound34892.bound]
def bound : CoeffClass := .finite ⟨18965303649908456346701791284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35574.bound, LeftBound34892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35574.actual selector witness, LeftBound34892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35579

namespace LeftBound35583
def owner : Owner := ⟨.program ⟨214⟩, ⟨27252⟩⟩
def transferEvent : Nat := 35583
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35581 .coefficient, .predecessor 1 35582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35581 .coefficient)
      LeftBound35578.bound (LeftBound35578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35582 .coefficient)
      LeftBound34678.bound (LeftBound34678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35578.bound, LeftBound34678.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35578.bound, LeftBound34678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35578.actual selector witness, LeftBound34678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35583

namespace LeftBound35584
def owner : Owner := ⟨.program ⟨214⟩, ⟨27252⟩⟩
def transferEvent : Nat := 35584
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35580 .summary, .result 34685 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35580 .summary)
      LeftBound35579.bound (LeftBound35579.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27035⟩⟩) (rawTerms := some (Proof.Events138.exact35580RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34685 .summary)
      LeftBound34680.bound (LeftBound34680.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27251⟩⟩) (rawTerms := some (Proof.Events135.exact34685RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34680.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35579.bound, LeftBound34680.bound]
def bound : CoeffClass := .finite ⟨23706886606235022529910538292, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35579.bound, LeftBound34680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35579.actual selector witness, LeftBound34680.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35584

namespace LeftBound35588
def owner : Owner := ⟨.program ⟨214⟩, ⟨27469⟩⟩
def transferEvent : Nat := 35588
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35586 .coefficient, .predecessor 1 35587 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35586 .coefficient)
      LeftBound35583.bound (LeftBound35583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35587 .coefficient)
      LeftBound34466.bound (LeftBound34466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34466.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35583.bound, LeftBound34466.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35583.bound, LeftBound34466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35583.actual selector witness, LeftBound34466.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35588

namespace LeftBound35589
def owner : Owner := ⟨.program ⟨214⟩, ⟨27469⟩⟩
def transferEvent : Nat := 35589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35585 .summary, .result 34473 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35585 .summary)
      LeftBound35584.bound (LeftBound35584.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27252⟩⟩) (rawTerms := some (Proof.Events139.exact35585RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34473 .summary)
      LeftBound34468.bound (LeftBound34468.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27468⟩⟩) (rawTerms := some (Proof.Events134.exact34473RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34468.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35584.bound, LeftBound34468.bound]
def bound : CoeffClass := .finite ⟨28448551816593413384009941044, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35584.bound, LeftBound34468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35584.actual selector witness, LeftBound34468.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35589

namespace LeftBound35593
def owner : Owner := ⟨.program ⟨214⟩, ⟨27686⟩⟩
def transferEvent : Nat := 35593
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35591 .coefficient, .predecessor 1 35592 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35591 .coefficient)
      LeftBound35588.bound (LeftBound35588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35592 .coefficient)
      LeftBound34254.bound (LeftBound34254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events133.exact34261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34254.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34254.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35588.bound, LeftBound34254.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35588.bound, LeftBound34254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35588.actual selector witness, LeftBound34254.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35593

namespace LeftBound35594
def owner : Owner := ⟨.program ⟨214⟩, ⟨27686⟩⟩
def transferEvent : Nat := 35594
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35590 .summary, .result 34261 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35590 .summary)
      LeftBound35589.bound (LeftBound35589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27469⟩⟩) (rawTerms := some (Proof.Events139.exact35590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34261 .summary)
      LeftBound34256.bound (LeftBound34256.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27685⟩⟩) (rawTerms := some (Proof.Events133.exact34261RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34256.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35589.bound, LeftBound34256.bound]
def bound : CoeffClass := .finite ⟨33190381535015453579890655284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35589.bound, LeftBound34256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35589.actual selector witness, LeftBound34256.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35594

namespace LeftBound35598
def owner : Owner := ⟨.program ⟨214⟩, ⟨27903⟩⟩
def transferEvent : Nat := 35598
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35596 .coefficient, .predecessor 1 35597 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35596 .coefficient)
      LeftBound35593.bound (LeftBound35593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35593.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35597 .coefficient)
      LeftBound34042.bound (LeftBound34042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events133.exact34049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35593.bound, LeftBound34042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35593.bound, LeftBound34042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35593.actual selector witness, LeftBound34042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35598

namespace LeftBound35599
def owner : Owner := ⟨.program ⟨214⟩, ⟨27903⟩⟩
def transferEvent : Nat := 35599
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35595 .summary, .result 34049 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35595 .summary)
      LeftBound35594.bound (LeftBound35594.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27686⟩⟩) (rawTerms := some (Proof.Events139.exact35595RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34049 .summary)
      LeftBound34044.bound (LeftBound34044.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27902⟩⟩) (rawTerms := some (Proof.Events133.exact34049RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34044.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35594.bound, LeftBound34044.bound]
def bound : CoeffClass := .finite ⟨37932293507469318446662025268, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35594.bound, LeftBound34044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35594.actual selector witness, LeftBound34044.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35599

namespace LeftBound35603
def owner : Owner := ⟨.program ⟨214⟩, ⟨28120⟩⟩
def transferEvent : Nat := 35603
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35601 .coefficient, .predecessor 1 35602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35601 .coefficient)
      LeftBound35598.bound (LeftBound35598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35602 .coefficient)
      LeftBound33830.bound (LeftBound33830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33830.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33830.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35598.bound, LeftBound33830.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35598.bound, LeftBound33830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35598.actual selector witness, LeftBound33830.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35603

namespace LeftBound35604
def owner : Owner := ⟨.program ⟨214⟩, ⟨28120⟩⟩
def transferEvent : Nat := 35604
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35600 .summary, .result 33837 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35600 .summary)
      LeftBound35599.bound (LeftBound35599.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27903⟩⟩) (rawTerms := some (Proof.Events139.exact35600RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33837 .summary)
      LeftBound33832.bound (LeftBound33832.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28119⟩⟩) (rawTerms := some (Proof.Events132.exact33837RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33832.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35599.bound, LeftBound33832.bound]
def bound : CoeffClass := .finite ⟨42674369987986832655214706740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35599.bound, LeftBound33832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35599.actual selector witness, LeftBound33832.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35604

namespace LeftBound35608
def owner : Owner := ⟨.program ⟨214⟩, ⟨28337⟩⟩
def transferEvent : Nat := 35608
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35606 .coefficient, .predecessor 1 35607 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35606 .coefficient)
      LeftBound35603.bound (LeftBound35603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35607 .coefficient)
      LeftBound33618.bound (LeftBound33618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33618.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35603.bound, LeftBound33618.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35603.bound, LeftBound33618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35603.actual selector witness, LeftBound33618.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35608

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
