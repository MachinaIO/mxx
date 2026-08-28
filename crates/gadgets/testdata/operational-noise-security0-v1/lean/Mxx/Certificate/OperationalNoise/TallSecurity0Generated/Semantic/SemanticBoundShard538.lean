import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard024
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard439
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard537

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79598
def owner : Owner := ⟨.program ⟨214⟩, ⟨30105⟩⟩
def transferEvent : Nat := 79598
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79575 .summary, .result 79530 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79575 .summary)
      LeftBound79552.bound (LeftBound79552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7903⟩⟩) (rawTerms := some (Proof.Events310.exact79575RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79530 .summary)
      LeftBound79529.bound (LeftBound79529.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30104⟩⟩) (rawTerms := some (Proof.Events310.exact79530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79529.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79552.bound, LeftBound79529.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789483581492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79552.bound, LeftBound79529.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79552.actual selector witness, LeftBound79529.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79598

namespace LeftBound79602
def owner : Owner := ⟨.program ⟨214⟩, ⟨30106⟩⟩
def transferEvent : Nat := 79602
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79600 .coefficient) (.predecessor 1 79601 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79600 .coefficient)
      LeftBound79578.bound (LeftBound79578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79601 .coefficient)
      LeftBound6120.bound (LeftBound6120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79578.bound LeftBound6120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79578.bound, LeftBound6120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79578.actual selector witness) * (LeftBound6120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79602

namespace LeftBound79603
def owner : Owner := ⟨.program ⟨214⟩, ⟨30106⟩⟩
def transferEvent : Nat := 79603
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩ [⟨.result 6117 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6117 .coefficient)
      LeftAuthority6116.bound (LeftAuthority6116.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7827⟩⟩) (rawTerms := some (Proof.Events023.exact6117RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6116.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6116.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6116.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6116.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79603

namespace LeftBound79604
def owner : Owner := ⟨.program ⟨214⟩, ⟨30106⟩⟩
def transferEvent : Nat := 79604
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 79599 .summary) (.transfer 79603) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79599 .summary)
      LeftBound79598.bound (LeftBound79598.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30105⟩⟩) (rawTerms := some (Proof.Events310.exact79599RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79603)
      LeftBound79603.bound (LeftBound79603.actual selector witness) := by
  exact .transfer (LeftBound79603.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79598.bound LeftBound79603.bound
def bound : CoeffClass := .finite ⟨1149729608724517268372876178953375812943872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79598.bound, LeftBound79603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79598.actual selector witness) * (LeftBound79603.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79604

namespace LeftBound79666
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def transferEvent : Nat := 79666
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79664 .coefficient, .predecessor 1 79665 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79664 .coefficient)
      LeftBound79602.bound (LeftBound79602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79665 .coefficient)
      LeftBound65183.bound (LeftBound65183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65183.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65183.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79602.bound, LeftBound65183.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79602.bound, LeftBound65183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79602.actual selector witness, LeftBound65183.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79666

namespace LeftBound79686
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def transferEvent : Nat := 79686
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79663 .summary, .result 65260 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79663 .summary)
      LeftBound79604.bound (LeftBound79604.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30106⟩⟩) (rawTerms := some (Proof.Events311.exact79663RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65260 .summary)
      LeftBound65221.bound (LeftBound65221.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18830⟩⟩) (rawTerms := some (Proof.Events254.exact65260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79604.bound, LeftBound65221.bound]
def bound : CoeffClass := .finite ⟨1149729608724524008718218297164355856419136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79604.bound, LeftBound65221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79604.actual selector witness, LeftBound65221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79686

namespace LeftBound79690
def owner : Owner := ⟨.program ⟨214⟩, ⟨30108⟩⟩
def transferEvent : Nat := 79690
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79688 .coefficient) (.predecessor 1 79689 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79688 .coefficient)
      LeftBound79666.bound (LeftBound79666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79666.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79689 .coefficient)
      LeftBound6110.bound (LeftBound6110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79666.bound LeftBound6110.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79666.bound, LeftBound6110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79666.actual selector witness) * (LeftBound6110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79690

namespace LeftBound79691
def owner : Owner := ⟨.program ⟨214⟩, ⟨30108⟩⟩
def transferEvent : Nat := 79691
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩ [⟨.result 6107 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6107 .coefficient)
      LeftAuthority6106.bound (LeftAuthority6106.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6685⟩⟩) (rawTerms := some (Proof.Events023.exact6107RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6106.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6106.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6106.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79691

namespace LeftBound79692
def owner : Owner := ⟨.program ⟨214⟩, ⟨30108⟩⟩
def transferEvent : Nat := 79692
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 79687 .summary) (.transfer 79691) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79687 .summary)
      LeftBound79686.bound (LeftBound79686.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30107⟩⟩) (rawTerms := some (Proof.Events311.exact79687RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79691)
      LeftBound79691.bound (LeftBound79691.actual selector witness) := by
  exact .transfer (LeftBound79691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79686.bound LeftBound79691.bound
def bound : CoeffClass := .finite ⟨4219526059692742704380000642085940622751931826176, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79686.bound, LeftBound79691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79686.actual selector witness) * (LeftBound79691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79692

namespace LeftBound79773
def owner : Owner := ⟨.program ⟨214⟩, ⟨5607⟩⟩
def transferEvent : Nat := 79773
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 79768 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79768 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79773

namespace LeftBound79777
def owner : Owner := ⟨.program ⟨214⟩, ⟨6579⟩⟩
def transferEvent : Nat := 79777
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79775 .coefficient) (.predecessor 1 79776 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79775 .coefficient)
      LeftBound79773.bound (LeftBound79773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79776 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79773.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79773.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79773.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79777

namespace LeftBound79789
def owner : Owner := ⟨.program ⟨214⟩, ⟨5539⟩⟩
def transferEvent : Nat := 79789
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 79784 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79784 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79789

namespace LeftBound79793
def owner : Owner := ⟨.program ⟨214⟩, ⟨7213⟩⟩
def transferEvent : Nat := 79793
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79791 .coefficient) (.predecessor 1 79792 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79791 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79792 .coefficient)
      LeftAuthority6153.bound (LeftAuthority6153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6153.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6153.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftAuthority6153.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftAuthority6153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftAuthority6153.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79793

namespace LeftBound79798
def owner : Owner := ⟨.program ⟨214⟩, ⟨7751⟩⟩
def transferEvent : Nat := 79798
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79796 .coefficient, .predecessor 1 79797 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79796 .coefficient)
      LeftBound79793.bound (LeftBound79793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79793.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79793.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79797 .coefficient)
      LeftBound79777.bound (LeftBound79777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79777.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79793.bound, LeftBound79777.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79793.bound, LeftBound79777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79793.actual selector witness, LeftBound79777.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79798

namespace LeftBound79802
def owner : Owner := ⟨.program ⟨214⟩, ⟨7752⟩⟩
def transferEvent : Nat := 79802
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79800 .coefficient, .predecessor 1 79801 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79800 .coefficient)
      LeftBound79798.bound (LeftBound79798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79801 .coefficient)
      LeftAuthority79752.bound (LeftAuthority79752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79752.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79798.bound, LeftAuthority79752.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79798.bound, LeftAuthority79752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79798.actual selector witness, LeftAuthority79752.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79802

namespace LeftBound79803
def owner : Owner := ⟨.program ⟨214⟩, ⟨7752⟩⟩
def transferEvent : Nat := 79803
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨44⟩⟩]⟩ [⟨.result 79753 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79753 .coefficient)
      LeftAuthority79752.bound (LeftAuthority79752.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨44⟩⟩) (rawTerms := some (Proof.Events311.exact79753RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79752.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority79752.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79752.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79803

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
