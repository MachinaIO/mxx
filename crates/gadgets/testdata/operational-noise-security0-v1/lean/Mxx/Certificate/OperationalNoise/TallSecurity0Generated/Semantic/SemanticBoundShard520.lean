import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard468
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard519

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound77065
def owner : Owner := ⟨.program ⟨214⟩, ⟨28717⟩⟩
def transferEvent : Nat := 77065
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 77059 .summary, .result 76881 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77059 .summary)
      LeftBound76893.bound (LeftBound76893.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21903⟩⟩) (rawTerms := some (Proof.Events301.exact77059RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76881 .summary)
      LeftBound76876.bound (LeftBound76876.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28716⟩⟩) (rawTerms := some (Proof.Events300.exact76881RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76876.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76893.bound, LeftBound76876.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76893.bound, LeftBound76876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76893.actual selector witness, LeftBound76876.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77065

namespace LeftBound77069
def owner : Owner := ⟨.program ⟨214⟩, ⟨28718⟩⟩
def transferEvent : Nat := 77069
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77067 .coefficient) (.predecessor 1 77068 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77067 .coefficient)
      LeftBound77062.bound (LeftBound77062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77068 .coefficient)
      LeftBound5638.bound (LeftBound5638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77062.bound LeftBound5638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77062.bound, LeftBound5638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77062.actual selector witness) * (LeftBound5638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77069

namespace LeftBound77070
def owner : Owner := ⟨.program ⟨214⟩, ⟨28718⟩⟩
def transferEvent : Nat := 77070
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩ [⟨.result 5635 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5635 .coefficient)
      LeftAuthority5634.bound (LeftAuthority5634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6673⟩⟩) (rawTerms := some (Proof.Events022.exact5635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5634.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5634.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77070

namespace LeftBound77071
def owner : Owner := ⟨.program ⟨214⟩, ⟨28718⟩⟩
def transferEvent : Nat := 77071
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 77066 .summary) (.transfer 77070) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77066 .summary)
      LeftBound77065.bound (LeftBound77065.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28717⟩⟩) (rawTerms := some (Proof.Events301.exact77066RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77065.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77070)
      LeftBound77070.bound (LeftBound77070.actual selector witness) := by
  exact .transfer (LeftBound77070.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77065.bound LeftBound77070.bound
def bound : CoeffClass := .finite ⟨4742652258740286904787271680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77065.bound, LeftBound77070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77065.actual selector witness) * (LeftBound77070.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77071

namespace LeftBound77086
def owner : Owner := ⟨.program ⟨214⟩, ⟨28499⟩⟩
def transferEvent : Nat := 77086
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77084 .coefficient) (.predecessor 1 77085 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77084 .coefficient)
      LeftBound68943.bound (LeftBound68943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77085 .coefficient)
      LeftAuthority77082.bound (LeftAuthority77082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77082.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68943.bound LeftAuthority77082.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68943.bound, LeftAuthority77082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68943.actual selector witness) * (LeftAuthority77082.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77086

namespace LeftBound77087
def owner : Owner := ⟨.program ⟨214⟩, ⟨28499⟩⟩
def transferEvent : Nat := 77087
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28497⟩⟩]⟩ [⟨.result 77083 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77083 .coefficient)
      LeftAuthority77082.bound (LeftAuthority77082.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28497⟩⟩) (rawTerms := some (Proof.Events301.exact77083RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77082.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77082.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77082.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77087

namespace LeftBound77088
def owner : Owner := ⟨.program ⟨214⟩, ⟨28499⟩⟩
def transferEvent : Nat := 77088
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68947 .summary) (.transfer 77087) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68947 .summary)
      LeftBound68946.bound (LeftBound68946.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25139⟩⟩) (rawTerms := some (Proof.Events269.exact68947RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77087)
      LeftBound77087.bound (LeftBound77087.actual selector witness) := by
  exact .transfer (LeftBound77087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68946.bound LeftBound77087.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68946.bound, LeftBound77087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68946.actual selector witness) * (LeftBound77087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77088

namespace LeftBound77099
def owner : Owner := ⟨.program ⟨214⟩, ⟨21758⟩⟩
def transferEvent : Nat := 77099
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 77097 .coefficient) (.value (.predecessor 1 77098 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77097 .coefficient)
      LeftAuthority77095.bound (LeftAuthority77095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77098 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority77095.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77095.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77095.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound77099

namespace LeftBound77103
def owner : Owner := ⟨.program ⟨214⟩, ⟨21759⟩⟩
def transferEvent : Nat := 77103
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77101 .coefficient) (.predecessor 1 77102 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77101 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77102 .coefficient)
      LeftBound77099.bound (LeftBound77099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77099.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound77099.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound77099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound77099.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77103

namespace LeftBound77104
def owner : Owner := ⟨.program ⟨214⟩, ⟨21759⟩⟩
def transferEvent : Nat := 77104
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21756⟩⟩]⟩ [⟨.result 77096 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77096 .coefficient)
      LeftAuthority77095.bound (LeftAuthority77095.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21756⟩⟩) (rawTerms := some (Proof.Events301.exact77096RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77095.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77095.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77095.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77104

namespace LeftBound77105
def owner : Owner := ⟨.program ⟨214⟩, ⟨21759⟩⟩
def transferEvent : Nat := 77105
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 77104) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77104)
      LeftBound77104.bound (LeftBound77104.actual selector witness) := by
  exact .transfer (LeftBound77104.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound77104.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound77104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound77104.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77105

namespace LeftBound77200
def owner : Owner := ⟨.program ⟨214⟩, ⟨16259⟩⟩
def transferEvent : Nat := 77200
def frameStart : Nat := 77161
def rule : BoundRule := .identity (.predecessor 0 77199 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77199 .coefficient)
      LeftAuthority77197.bound (LeftAuthority77197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77197.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77197.derived selector witness)

def rawBound : CoeffClass := LeftAuthority77197.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority77197.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77200

namespace LeftBound77217
def owner : Owner := ⟨.program ⟨214⟩, ⟨16333⟩⟩
def transferEvent : Nat := 77217
def frameStart : Nat := 77161
def rule : BoundRule := .sum [.predecessor 0 77215 .coefficient, .predecessor 1 77216 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77215 .coefficient)
      LeftBound77200.bound (LeftBound77200.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77216 .coefficient)
      LeftAuthority77213.bound (LeftAuthority77213.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority77213.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77200.bound, LeftAuthority77213.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77200.bound, LeftAuthority77213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77200.actual selector witness, LeftAuthority77213.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77217

namespace LeftBound77220
def owner : Owner := ⟨.program ⟨214⟩, ⟨16334⟩⟩
def transferEvent : Nat := 77220
def frameStart : Nat := 77161
def rule : BoundRule := .identity (.predecessor 0 77219 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77219 .coefficient)
      LeftBound77217.bound (LeftBound77217.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77217.derived selector witness)

def rawBound : CoeffClass := LeftBound77217.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound77217.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77220

namespace LeftBound77226
def owner : Owner := ⟨.program ⟨214⟩, ⟨16335⟩⟩
def transferEvent : Nat := 77226
def frameStart : Nat := 77161
def rule : BoundRule := .product (.predecessor 0 77224 .coefficient) (.predecessor 1 77225 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77224 .coefficient)
      LeftAuthority77222.bound (LeftAuthority77222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77222.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77225 .coefficient)
      LeftBound77220.bound (LeftBound77220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77220.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority77222.bound LeftBound77220.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77222.bound, LeftBound77220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority77222.actual selector witness) * (LeftBound77220.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77226

namespace LeftBound77234
def owner : Owner := ⟨.program ⟨214⟩, ⟨16336⟩⟩
def transferEvent : Nat := 77234
def frameStart : Nat := 77161
def rule : BoundRule := .sum [.predecessor 0 77232 .coefficient, .predecessor 1 77233 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77232 .coefficient)
      LeftAuthority77230.bound (LeftAuthority77230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77230.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77233 .coefficient)
      LeftBound77226.bound (LeftBound77226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77226.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77230.bound, LeftBound77226.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77230.bound, LeftBound77226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77230.actual selector witness, LeftBound77226.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77234

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
