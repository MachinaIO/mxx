import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard092
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard295

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound43809
def owner : Owner := ⟨.program ⟨214⟩, ⟨9517⟩⟩
def transferEvent : Nat := 43809
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43807 .coefficient, .predecessor 1 43808 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43807 .coefficient)
      LeftBound43804.bound (LeftBound43804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43804.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43808 .coefficient)
      LeftBound43799.bound (LeftBound43799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43804.bound, LeftBound43799.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43804.bound, LeftBound43799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43804.actual selector witness, LeftBound43799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43809

namespace LeftBound43813
def owner : Owner := ⟨.program ⟨214⟩, ⟨9518⟩⟩
def transferEvent : Nat := 43813
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43811 .coefficient, .predecessor 1 43812 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43811 .coefficient)
      LeftBound43809.bound (LeftBound43809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43812 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43809.bound, LeftBound14520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43809.bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43809.actual selector witness, LeftBound14520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43813

namespace LeftBound43814
def owner : Owner := ⟨.program ⟨214⟩, ⟨9518⟩⟩
def transferEvent : Nat := 43814
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨96⟩⟩]⟩ [⟨.result 14521 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14521 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨96⟩⟩) (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14520.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14520.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43814

namespace LeftBound43819
def owner : Owner := ⟨.program ⟨214⟩, ⟨9519⟩⟩
def transferEvent : Nat := 43819
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43817 .coefficient) (.predecessor 1 43818 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43817 .coefficient)
      LeftBound43813.bound (LeftBound43813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43818 .coefficient)
      LeftBound14517.bound (LeftBound14517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43813.bound LeftBound14517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43813.bound, LeftBound14517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43813.actual selector witness) * (LeftBound14517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43819

namespace LeftBound43820
def owner : Owner := ⟨.program ⟨214⟩, ⟨9519⟩⟩
def transferEvent : Nat := 43820
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩ [⟨.result 14514 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14514 .coefficient)
      LeftAuthority14513.bound (LeftAuthority14513.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7834⟩⟩) (rawTerms := some (Proof.Events056.exact14514RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14513.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14513.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14513.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43820

namespace LeftBound43821
def owner : Owner := ⟨.program ⟨214⟩, ⟨9519⟩⟩
def transferEvent : Nat := 43821
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43816 .summary) (.transfer 43820) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43816 .summary)
      LeftBound43814.bound (LeftBound43814.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9518⟩⟩) (rawTerms := some (Proof.Events171.exact43816RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43820)
      LeftBound43820.bound (LeftBound43820.actual selector witness) := by
  exact .transfer (LeftBound43820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43814.bound LeftBound43820.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43814.bound, LeftBound43820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43814.actual selector witness) * (LeftBound43820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43821

namespace LeftBound43829
def owner : Owner := ⟨.program ⟨214⟩, ⟨10699⟩⟩
def transferEvent : Nat := 43829
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43827 .coefficient, .predecessor 1 43828 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43827 .coefficient)
      LeftBound43819.bound (LeftBound43819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43828 .coefficient)
      LeftBound43791.bound (LeftBound43791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43819.bound, LeftBound43791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43819.bound, LeftBound43791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43819.actual selector witness, LeftBound43791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43829

namespace LeftBound43831
def owner : Owner := ⟨.program ⟨214⟩, ⟨10699⟩⟩
def transferEvent : Nat := 43831
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 43826 .summary, .result 43796 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43826 .summary)
      LeftBound43821.bound (LeftBound43821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9519⟩⟩) (rawTerms := some (Proof.Events171.exact43826RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43796 .summary)
      LeftBound43793.bound (LeftBound43793.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10698⟩⟩) (rawTerms := some (Proof.Events171.exact43796RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43793.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43821.bound, LeftBound43793.bound]
def bound : CoeffClass := .finite ⟨95422912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43821.bound, LeftBound43793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43821.actual selector witness, LeftBound43793.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43831

namespace LeftBound43835
def owner : Owner := ⟨.program ⟨214⟩, ⟨24999⟩⟩
def transferEvent : Nat := 43835
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43833 .coefficient) (.predecessor 1 43834 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43833 .coefficient)
      LeftBound43829.bound (LeftBound43829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43829.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43834 .coefficient)
      LeftAuthority43767.bound (LeftAuthority43767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43767.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43767.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43829.bound LeftAuthority43767.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43829.bound, LeftAuthority43767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43829.actual selector witness) * (LeftAuthority43767.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43835

namespace LeftBound43836
def owner : Owner := ⟨.program ⟨214⟩, ⟨24999⟩⟩
def transferEvent : Nat := 43836
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩ [⟨.result 43768 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43768 .coefficient)
      LeftAuthority43767.bound (LeftAuthority43767.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24998⟩⟩) (rawTerms := some (Proof.Events170.exact43768RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43767.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43767.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43767.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43767.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43836

namespace LeftBound43837
def owner : Owner := ⟨.program ⟨214⟩, ⟨24999⟩⟩
def transferEvent : Nat := 43837
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43832 .summary) (.transfer 43836) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43832 .summary)
      LeftBound43831.bound (LeftBound43831.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10699⟩⟩) (rawTerms := some (Proof.Events171.exact43832RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43836)
      LeftBound43836.bound (LeftBound43836.actual selector witness) := by
  exact .transfer (LeftBound43836.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43831.bound LeftBound43836.bound
def bound : CoeffClass := .finite ⟨350203613806592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43831.bound, LeftBound43836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43831.actual selector witness) * (LeftBound43836.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43837

namespace LeftBound43848
def owner : Owner := ⟨.program ⟨214⟩, ⟨19106⟩⟩
def transferEvent : Nat := 43848
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 43846 .coefficient) (.value (.predecessor 1 43847 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43846 .coefficient)
      LeftAuthority43844.bound (LeftAuthority43844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43847 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority43844.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43844.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43844.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43848

namespace LeftBound43852
def owner : Owner := ⟨.program ⟨214⟩, ⟨19107⟩⟩
def transferEvent : Nat := 43852
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43850 .coefficient) (.predecessor 1 43851 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43850 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43851 .coefficient)
      LeftBound43848.bound (LeftBound43848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43848.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound43848.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound43848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound43848.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43852

namespace LeftBound43853
def owner : Owner := ⟨.program ⟨214⟩, ⟨19107⟩⟩
def transferEvent : Nat := 43853
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩ [⟨.result 43845 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43845 .coefficient)
      LeftAuthority43844.bound (LeftAuthority43844.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19104⟩⟩) (rawTerms := some (Proof.Events171.exact43845RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43844.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43844.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43844.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43853

namespace LeftBound43854
def owner : Owner := ⟨.program ⟨214⟩, ⟨19107⟩⟩
def transferEvent : Nat := 43854
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 43853) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43853)
      LeftBound43853.bound (LeftBound43853.actual selector witness) := by
  exact .transfer (LeftBound43853.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound43853.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound43853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound43853.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43854

namespace LeftBound43933
def owner : Owner := ⟨.program ⟨214⟩, ⟨10693⟩⟩
def transferEvent : Nat := 43933
def frameStart : Nat := 43904
def rule : BoundRule := .product (.predecessor 0 43931 .coefficient) (.predecessor 1 43932 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43931 .coefficient)
      LeftAuthority43929.bound (LeftAuthority43929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43932 .coefficient)
      LeftAuthority43926.bound (LeftAuthority43926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43926.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority43929.bound LeftAuthority43926.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43929.bound, LeftAuthority43926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority43929.actual selector witness) * (LeftAuthority43926.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43933

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
