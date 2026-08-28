import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard587
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard627

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound92714
def owner : Owner := ⟨.program ⟨214⟩, ⟨27645⟩⟩
def transferEvent : Nat := 92714
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 92708 .summary, .result 92530 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92708 .summary)
      LeftBound92542.bound (LeftBound92542.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21187⟩⟩) (rawTerms := some (Proof.Events362.exact92708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92530 .summary)
      LeftBound92525.bound (LeftBound92525.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27644⟩⟩) (rawTerms := some (Proof.Events361.exact92530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92525.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92542.bound, LeftBound92525.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92542.bound, LeftBound92525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92542.actual selector witness, LeftBound92525.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92714

namespace LeftBound92718
def owner : Owner := ⟨.program ⟨214⟩, ⟨27646⟩⟩
def transferEvent : Nat := 92718
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92716 .coefficient) (.predecessor 1 92717 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92716 .coefficient)
      LeftBound92711.bound (LeftBound92711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92717 .coefficient)
      LeftBound5738.bound (LeftBound5738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92711.bound LeftBound5738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92711.bound, LeftBound5738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92711.actual selector witness) * (LeftBound5738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92718

namespace LeftBound92719
def owner : Owner := ⟨.program ⟨214⟩, ⟨27646⟩⟩
def transferEvent : Nat := 92719
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩ [⟨.result 5735 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5735 .coefficient)
      LeftAuthority5734.bound (LeftAuthority5734.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6643⟩⟩) (rawTerms := some (Proof.Events022.exact5735RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5734.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5734.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5734.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92719

namespace LeftBound92720
def owner : Owner := ⟨.program ⟨214⟩, ⟨27646⟩⟩
def transferEvent : Nat := 92720
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 92715 .summary) (.transfer 92719) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92715 .summary)
      LeftBound92714.bound (LeftBound92714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27645⟩⟩) (rawTerms := some (Proof.Events362.exact92715RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92719)
      LeftBound92719.bound (LeftBound92719.actual selector witness) := by
  exact .transfer (LeftBound92719.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92714.bound LeftBound92719.bound
def bound : CoeffClass := .finite ⟨4741829718422040195880714240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92714.bound, LeftBound92719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92714.actual selector witness) * (LeftBound92719.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92720

namespace LeftBound92735
def owner : Owner := ⟨.program ⟨214⟩, ⟨27427⟩⟩
def transferEvent : Nat := 92735
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92733 .coefficient) (.predecessor 1 92734 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92733 .coefficient)
      LeftBound85952.bound (LeftBound85952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92734 .coefficient)
      LeftAuthority92731.bound (LeftAuthority92731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92731.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85952.bound LeftAuthority92731.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85952.bound, LeftAuthority92731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85952.actual selector witness) * (LeftAuthority92731.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92735

namespace LeftBound92736
def owner : Owner := ⟨.program ⟨214⟩, ⟨27427⟩⟩
def transferEvent : Nat := 92736
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27425⟩⟩]⟩ [⟨.result 92732 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92732 .coefficient)
      LeftAuthority92731.bound (LeftAuthority92731.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27425⟩⟩) (rawTerms := some (Proof.Events362.exact92732RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92731.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92731.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92731.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92736

namespace LeftBound92737
def owner : Owner := ⟨.program ⟨214⟩, ⟨27427⟩⟩
def transferEvent : Nat := 92737
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 85956 .summary) (.transfer 92736) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85956 .summary)
      LeftBound85955.bound (LeftBound85955.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25914⟩⟩) (rawTerms := some (Proof.Events335.exact85956RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92736)
      LeftBound92736.bound (LeftBound92736.actual selector witness) := by
  exact .transfer (LeftBound92736.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85955.bound LeftBound92736.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85955.bound, LeftBound92736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85955.actual selector witness) * (LeftBound92736.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92737

namespace LeftBound92748
def owner : Owner := ⟨.program ⟨214⟩, ⟨21042⟩⟩
def transferEvent : Nat := 92748
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 92746 .coefficient) (.value (.predecessor 1 92747 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92746 .coefficient)
      LeftAuthority92744.bound (LeftAuthority92744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92747 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority92744.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92744.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92744.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound92748

namespace LeftBound92752
def owner : Owner := ⟨.program ⟨214⟩, ⟨21043⟩⟩
def transferEvent : Nat := 92752
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92750 .coefficient) (.predecessor 1 92751 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92750 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92751 .coefficient)
      LeftBound92748.bound (LeftBound92748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92748.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound92748.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound92748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound92748.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92752

namespace LeftBound92753
def owner : Owner := ⟨.program ⟨214⟩, ⟨21043⟩⟩
def transferEvent : Nat := 92753
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21040⟩⟩]⟩ [⟨.result 92745 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92745 .coefficient)
      LeftAuthority92744.bound (LeftAuthority92744.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21040⟩⟩) (rawTerms := some (Proof.Events362.exact92745RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92744.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92744.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92744.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92753

namespace LeftBound92754
def owner : Owner := ⟨.program ⟨214⟩, ⟨21043⟩⟩
def transferEvent : Nat := 92754
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 92753) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92753)
      LeftBound92753.bound (LeftBound92753.actual selector witness) := by
  exact .transfer (LeftBound92753.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound92753.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound92753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound92753.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92754

namespace LeftBound92849
def owner : Owner := ⟨.program ⟨214⟩, ⟨15703⟩⟩
def transferEvent : Nat := 92849
def frameStart : Nat := 92810
def rule : BoundRule := .identity (.predecessor 0 92848 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92848 .coefficient)
      LeftAuthority92846.bound (LeftAuthority92846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92846.derived selector witness)

def rawBound : CoeffClass := LeftAuthority92846.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority92846.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92849

namespace LeftBound92866
def owner : Owner := ⟨.program ⟨214⟩, ⟨15777⟩⟩
def transferEvent : Nat := 92866
def frameStart : Nat := 92810
def rule : BoundRule := .sum [.predecessor 0 92864 .coefficient, .predecessor 1 92865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92864 .coefficient)
      LeftBound92849.bound (LeftBound92849.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92865 .coefficient)
      LeftAuthority92862.bound (LeftAuthority92862.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority92862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92849.bound, LeftAuthority92862.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92849.bound, LeftAuthority92862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92849.actual selector witness, LeftAuthority92862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92866

namespace LeftBound92869
def owner : Owner := ⟨.program ⟨214⟩, ⟨15778⟩⟩
def transferEvent : Nat := 92869
def frameStart : Nat := 92810
def rule : BoundRule := .identity (.predecessor 0 92868 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92868 .coefficient)
      LeftBound92866.bound (LeftBound92866.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92866.derived selector witness)

def rawBound : CoeffClass := LeftBound92866.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound92866.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92869

namespace LeftBound92875
def owner : Owner := ⟨.program ⟨214⟩, ⟨15779⟩⟩
def transferEvent : Nat := 92875
def frameStart : Nat := 92810
def rule : BoundRule := .product (.predecessor 0 92873 .coefficient) (.predecessor 1 92874 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92873 .coefficient)
      LeftAuthority92871.bound (LeftAuthority92871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92874 .coefficient)
      LeftBound92869.bound (LeftBound92869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92869.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority92871.bound LeftBound92869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92871.bound, LeftBound92869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority92871.actual selector witness) * (LeftBound92869.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92875

namespace LeftBound92883
def owner : Owner := ⟨.program ⟨214⟩, ⟨15780⟩⟩
def transferEvent : Nat := 92883
def frameStart : Nat := 92810
def rule : BoundRule := .sum [.predecessor 0 92881 .coefficient, .predecessor 1 92882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92881 .coefficient)
      LeftAuthority92879.bound (LeftAuthority92879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92879.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92882 .coefficient)
      LeftBound92875.bound (LeftBound92875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92879.bound, LeftBound92875.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92879.bound, LeftBound92875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92879.actual selector witness, LeftBound92875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92883

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
