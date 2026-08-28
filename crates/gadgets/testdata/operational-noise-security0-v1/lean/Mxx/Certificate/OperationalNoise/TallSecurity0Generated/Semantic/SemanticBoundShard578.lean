import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard577

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84739
def owner : Owner := ⟨.program ⟨214⟩, ⟨7235⟩⟩
def transferEvent : Nat := 84739
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84737 .coefficient) (.predecessor 1 84738 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84737 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84738 .coefficient)
      LeftBound11481.bound (LeftBound11481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound11481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound11481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound11481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84739

namespace LeftBound84744
def owner : Owner := ⟨.program ⟨214⟩, ⟨11471⟩⟩
def transferEvent : Nat := 84744
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84742 .coefficient, .predecessor 1 84743 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84742 .coefficient)
      LeftBound84739.bound (LeftBound84739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84743 .coefficient)
      LeftBound84734.bound (LeftBound84734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84734.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84734.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84739.bound, LeftBound84734.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84739.bound, LeftBound84734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84739.actual selector witness, LeftBound84734.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84744

namespace LeftBound84748
def owner : Owner := ⟨.program ⟨214⟩, ⟨11472⟩⟩
def transferEvent : Nat := 84748
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84746 .coefficient, .predecessor 1 84747 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84746 .coefficient)
      LeftBound84744.bound (LeftBound84744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84747 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84744.bound, LeftBound11473.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84744.bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84744.actual selector witness, LeftBound11473.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84748

namespace LeftBound84749
def owner : Owner := ⟨.program ⟨214⟩, ⟨11472⟩⟩
def transferEvent : Nat := 84749
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩ [⟨.result 11474 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11474 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨93⟩⟩) (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11473.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11473.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84749

namespace LeftBound84754
def owner : Owner := ⟨.program ⟨214⟩, ⟨14210⟩⟩
def transferEvent : Nat := 84754
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84752 .coefficient) (.predecessor 1 84753 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84752 .coefficient)
      LeftBound84748.bound (LeftBound84748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84753 .coefficient)
      LeftAuthority4060.bound (LeftAuthority4060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4060.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4060.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound84748.bound LeftAuthority4060.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84748.bound, LeftAuthority4060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound84748.actual selector witness) * (LeftAuthority4060.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84754

namespace LeftBound84755
def owner : Owner := ⟨.program ⟨214⟩, ⟨14210⟩⟩
def transferEvent : Nat := 84755
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩ [⟨.result 4061 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4061 .coefficient)
      LeftAuthority4060.bound (LeftAuthority4060.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14207⟩⟩) (rawTerms := some (Proof.Events015.exact4061RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4060.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4060.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4060.bound []
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4060.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84755

namespace LeftBound84756
def owner : Owner := ⟨.program ⟨214⟩, ⟨14210⟩⟩
def transferEvent : Nat := 84756
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84751 .summary) (.transfer 84755) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84751 .summary)
      LeftBound84749.bound (LeftBound84749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11472⟩⟩) (rawTerms := some (Proof.Events331.exact84751RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84755)
      LeftBound84755.bound (LeftBound84755.actual selector witness) := by
  exact .transfer (LeftBound84755.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound84749.bound LeftBound84755.bound
def bound : CoeffClass := .finite ⟨14976, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84749.bound, LeftBound84755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound84749.actual selector witness) * (LeftBound84755.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84756

namespace LeftBound84762
def owner : Owner := ⟨.program ⟨214⟩, ⟨14211⟩⟩
def transferEvent : Nat := 84762
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 84760 .coefficient) (.predecessor 1 84761 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84760 .coefficient)
      LeftAuthority4060.bound (LeftAuthority4060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4060.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4060.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84761 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4060.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4060.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4060.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84762

namespace LeftBound84767
def owner : Owner := ⟨.program ⟨214⟩, ⟨7215⟩⟩
def transferEvent : Nat := 84767
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84765 .coefficient) (.predecessor 1 84766 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84765 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84766 .coefficient)
      LeftBound11522.bound (LeftBound11522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound11522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound11522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound11522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84767

namespace LeftBound84772
def owner : Owner := ⟨.program ⟨214⟩, ⟨14212⟩⟩
def transferEvent : Nat := 84772
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84770 .coefficient, .predecessor 1 84771 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84770 .coefficient)
      LeftBound84767.bound (LeftBound84767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84771 .coefficient)
      LeftBound84762.bound (LeftBound84762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84762.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84767.bound, LeftBound84762.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84767.bound, LeftBound84762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84767.actual selector witness, LeftBound84762.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84772

namespace LeftBound84776
def owner : Owner := ⟨.program ⟨214⟩, ⟨14213⟩⟩
def transferEvent : Nat := 84776
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84774 .coefficient, .predecessor 1 84775 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84774 .coefficient)
      LeftBound84772.bound (LeftBound84772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84775 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84772.bound, LeftBound11514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84772.bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84772.actual selector witness, LeftBound11514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84776

namespace LeftBound84777
def owner : Owner := ⟨.program ⟨214⟩, ⟨14213⟩⟩
def transferEvent : Nat := 84777
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩ [⟨.result 11515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11515 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨73⟩⟩) (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11514.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84777

namespace LeftBound84782
def owner : Owner := ⟨.program ⟨214⟩, ⟨14214⟩⟩
def transferEvent : Nat := 84782
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84780 .coefficient) (.predecessor 1 84781 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84780 .coefficient)
      LeftBound84776.bound (LeftBound84776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84781 .coefficient)
      LeftBound11511.bound (LeftBound11511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84776.bound LeftBound11511.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84776.bound, LeftBound11511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84776.actual selector witness) * (LeftBound11511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84782

namespace LeftBound84783
def owner : Owner := ⟨.program ⟨214⟩, ⟨14214⟩⟩
def transferEvent : Nat := 84783
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩ [⟨.result 11508 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11508 .coefficient)
      LeftAuthority11507.bound (LeftAuthority11507.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7852⟩⟩) (rawTerms := some (Proof.Events044.exact11508RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11507.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11507.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11507.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84783

namespace LeftBound84784
def owner : Owner := ⟨.program ⟨214⟩, ⟨14214⟩⟩
def transferEvent : Nat := 84784
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84779 .summary) (.transfer 84783) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84779 .summary)
      LeftBound84777.bound (LeftBound84777.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14213⟩⟩) (rawTerms := some (Proof.Events331.exact84779RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84783)
      LeftBound84783.bound (LeftBound84783.actual selector witness) := by
  exact .transfer (LeftBound84783.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84777.bound LeftBound84783.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84777.bound, LeftBound84783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84777.actual selector witness) * (LeftBound84783.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84784

namespace LeftBound84792
def owner : Owner := ⟨.program ⟨214⟩, ⟨14215⟩⟩
def transferEvent : Nat := 84792
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84790 .coefficient, .predecessor 1 84791 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84790 .coefficient)
      LeftBound84782.bound (LeftBound84782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84791 .coefficient)
      LeftBound84754.bound (LeftBound84754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84754.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84782.bound, LeftBound84754.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84782.bound, LeftBound84754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84782.actual selector witness, LeftBound84754.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84792

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
