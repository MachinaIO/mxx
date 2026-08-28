import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard551
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard614

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90599
def owner : Owner := ⟨.program ⟨214⟩, ⟨29816⟩⟩
def transferEvent : Nat := 90599
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩ [⟨.result 5535 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5535 .coefficient)
      LeftAuthority5534.bound (LeftAuthority5534.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6659⟩⟩) (rawTerms := some (Proof.Events021.exact5535RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5534.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5534.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5534.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5534.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90599

namespace LeftBound90600
def owner : Owner := ⟨.program ⟨214⟩, ⟨29816⟩⟩
def transferEvent : Nat := 90600
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90595 .summary) (.transfer 90599) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90595 .summary)
      LeftBound90594.bound (LeftBound90594.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29815⟩⟩) (rawTerms := some (Proof.Events353.exact90595RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90599)
      LeftBound90599.bound (LeftBound90599.actual selector witness) := by
  exact .transfer (LeftBound90599.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90594.bound LeftBound90599.bound
def bound : CoeffClass := .finite ⟨4743557053090358284584484864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90594.bound, LeftBound90599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90594.actual selector witness) * (LeftBound90599.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90600

namespace LeftBound90615
def owner : Owner := ⟨.program ⟨214⟩, ⟨29597⟩⟩
def transferEvent : Nat := 90615
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90613 .coefficient) (.predecessor 1 90614 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90613 .coefficient)
      LeftBound81152.bound (LeftBound81152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90614 .coefficient)
      LeftAuthority90611.bound (LeftAuthority90611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90611.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90611.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81152.bound LeftAuthority90611.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81152.bound, LeftAuthority90611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81152.actual selector witness) * (LeftAuthority90611.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90615

namespace LeftBound90616
def owner : Owner := ⟨.program ⟨214⟩, ⟨29597⟩⟩
def transferEvent : Nat := 90616
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩ [⟨.result 90612 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90612 .coefficient)
      LeftAuthority90611.bound (LeftAuthority90611.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29595⟩⟩) (rawTerms := some (Proof.Events353.exact90612RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90611.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90611.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority90611.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90611.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90616

namespace LeftBound90617
def owner : Owner := ⟨.program ⟨214⟩, ⟨29597⟩⟩
def transferEvent : Nat := 90617
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 81156 .summary) (.transfer 90616) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81156 .summary)
      LeftBound81155.bound (LeftBound81155.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25606⟩⟩) (rawTerms := some (Proof.Events317.exact81156RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90616)
      LeftBound90616.bound (LeftBound90616.actual selector witness) := by
  exact .transfer (LeftBound90616.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81155.bound LeftBound90616.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81155.bound, LeftBound90616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81155.actual selector witness) * (LeftBound90616.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90617

namespace LeftBound90628
def owner : Owner := ⟨.program ⟨214⟩, ⟨22482⟩⟩
def transferEvent : Nat := 90628
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 90626 .coefficient) (.value (.predecessor 1 90627 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90626 .coefficient)
      LeftAuthority90624.bound (LeftAuthority90624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90627 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority90624.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90624.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90624.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound90628

namespace LeftBound90632
def owner : Owner := ⟨.program ⟨214⟩, ⟨22483⟩⟩
def transferEvent : Nat := 90632
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90630 .coefficient) (.predecessor 1 90631 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90630 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90631 .coefficient)
      LeftBound90628.bound (LeftBound90628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90628.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound90628.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound90628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound90628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90632

namespace LeftBound90633
def owner : Owner := ⟨.program ⟨214⟩, ⟨22483⟩⟩
def transferEvent : Nat := 90633
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩ [⟨.result 90625 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90625 .coefficient)
      LeftAuthority90624.bound (LeftAuthority90624.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22480⟩⟩) (rawTerms := some (Proof.Events354.exact90625RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90624.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority90624.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90624.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90633

namespace LeftBound90634
def owner : Owner := ⟨.program ⟨214⟩, ⟨22483⟩⟩
def transferEvent : Nat := 90634
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 90633) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90633)
      LeftBound90633.bound (LeftBound90633.actual selector witness) := by
  exact .transfer (LeftBound90633.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound90633.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound90633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound90633.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90634

namespace LeftBound90729
def owner : Owner := ⟨.program ⟨214⟩, ⟨16753⟩⟩
def transferEvent : Nat := 90729
def frameStart : Nat := 90690
def rule : BoundRule := .identity (.predecessor 0 90728 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90728 .coefficient)
      LeftAuthority90726.bound (LeftAuthority90726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90726.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90726.derived selector witness)

def rawBound : CoeffClass := LeftAuthority90726.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority90726.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90729

namespace LeftBound90746
def owner : Owner := ⟨.program ⟨214⟩, ⟨16827⟩⟩
def transferEvent : Nat := 90746
def frameStart : Nat := 90690
def rule : BoundRule := .sum [.predecessor 0 90744 .coefficient, .predecessor 1 90745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90744 .coefficient)
      LeftBound90729.bound (LeftBound90729.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound90729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90745 .coefficient)
      LeftAuthority90742.bound (LeftAuthority90742.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority90742.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90729.bound, LeftAuthority90742.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90729.bound, LeftAuthority90742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90729.actual selector witness, LeftAuthority90742.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90746

namespace LeftBound90749
def owner : Owner := ⟨.program ⟨214⟩, ⟨16828⟩⟩
def transferEvent : Nat := 90749
def frameStart : Nat := 90690
def rule : BoundRule := .identity (.predecessor 0 90748 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90748 .coefficient)
      LeftBound90746.bound (LeftBound90746.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound90746.derived selector witness)

def rawBound : CoeffClass := LeftBound90746.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound90746.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90749

namespace LeftBound90755
def owner : Owner := ⟨.program ⟨214⟩, ⟨16829⟩⟩
def transferEvent : Nat := 90755
def frameStart : Nat := 90690
def rule : BoundRule := .product (.predecessor 0 90753 .coefficient) (.predecessor 1 90754 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90753 .coefficient)
      LeftAuthority90751.bound (LeftAuthority90751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90751.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90754 .coefficient)
      LeftBound90749.bound (LeftBound90749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90749.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority90751.bound LeftBound90749.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90751.bound, LeftBound90749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority90751.actual selector witness) * (LeftBound90749.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90755

namespace LeftBound90763
def owner : Owner := ⟨.program ⟨214⟩, ⟨16830⟩⟩
def transferEvent : Nat := 90763
def frameStart : Nat := 90690
def rule : BoundRule := .sum [.predecessor 0 90761 .coefficient, .predecessor 1 90762 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90761 .coefficient)
      LeftAuthority90759.bound (LeftAuthority90759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90762 .coefficient)
      LeftBound90755.bound (LeftBound90755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90759.bound, LeftBound90755.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90759.bound, LeftBound90755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority90759.actual selector witness, LeftBound90755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90763

namespace LeftBound90767
def owner : Owner := ⟨.program ⟨214⟩, ⟨29596⟩⟩
def transferEvent : Nat := 90767
def frameStart : Nat := 90690
def rule : BoundRule := .product (.predecessor 0 90765 .coefficient) (.predecessor 1 90766 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90765 .coefficient)
      LeftBound90763.bound (LeftBound90763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90766 .coefficient)
      LeftAuthority90740.bound (LeftAuthority90740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90740.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90740.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90763.bound LeftAuthority90740.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90763.bound, LeftAuthority90740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90763.actual selector witness) * (LeftAuthority90740.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90767

namespace LeftBound90778
def owner : Owner := ⟨.program ⟨214⟩, ⟨17496⟩⟩
def transferEvent : Nat := 90778
def frameStart : Nat := 90690
def rule : BoundRule := .product (.predecessor 0 90776 .coefficient) (.predecessor 1 90777 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90776 .coefficient)
      LeftAuthority90751.bound (LeftAuthority90751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90751.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90777 .coefficient)
      LeftAuthority90774.bound (LeftAuthority90774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90774.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority90751.bound LeftAuthority90774.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90751.bound, LeftAuthority90774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority90751.actual selector witness) * (LeftAuthority90774.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90778

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
