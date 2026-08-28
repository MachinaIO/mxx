import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard192

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28944
def owner : Owner := ⟨.program ⟨214⟩, ⟨20694⟩⟩
def transferEvent : Nat := 28944
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 28942 .coefficient) (.value (.predecessor 1 28943 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28942 .coefficient)
      LeftAuthority28940.bound (LeftAuthority28940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact28941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28940.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28943 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority28940.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28940.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28940.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28944

namespace LeftBound28948
def owner : Owner := ⟨.program ⟨214⟩, ⟨20695⟩⟩
def transferEvent : Nat := 28948
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28946 .coefficient) (.predecessor 1 28947 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28946 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28947 .coefficient)
      LeftBound28944.bound (LeftBound28944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact28945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28944.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound28944.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound28944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound28944.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28948

namespace LeftBound28949
def owner : Owner := ⟨.program ⟨214⟩, ⟨20695⟩⟩
def transferEvent : Nat := 28949
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20692⟩⟩]⟩ [⟨.result 28941 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28941 .coefficient)
      LeftAuthority28940.bound (LeftAuthority28940.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20692⟩⟩) (rawTerms := some (Proof.Events113.exact28941RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28940.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28940.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28940.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28940.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28949

namespace LeftBound28950
def owner : Owner := ⟨.program ⟨214⟩, ⟨20695⟩⟩
def transferEvent : Nat := 28950
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 28949) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28949)
      LeftBound28949.bound (LeftBound28949.actual selector witness) := by
  exact .transfer (LeftBound28949.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound28949.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound28949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound28949.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28950

namespace LeftBound29045
def owner : Owner := ⟨.program ⟨214⟩, ⟨15127⟩⟩
def transferEvent : Nat := 29045
def frameStart : Nat := 29006
def rule : BoundRule := .identity (.predecessor 0 29044 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29044 .coefficient)
      LeftAuthority29042.bound (LeftAuthority29042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29042.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29042.derived selector witness)

def rawBound : CoeffClass := LeftAuthority29042.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority29042.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29045

namespace LeftBound29062
def owner : Owner := ⟨.program ⟨214⟩, ⟨15166⟩⟩
def transferEvent : Nat := 29062
def frameStart : Nat := 29006
def rule : BoundRule := .sum [.predecessor 0 29060 .coefficient, .predecessor 1 29061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29060 .coefficient)
      LeftBound29045.bound (LeftBound29045.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound29045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29061 .coefficient)
      LeftAuthority29058.bound (LeftAuthority29058.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority29058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29045.bound, LeftAuthority29058.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29045.bound, LeftAuthority29058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29045.actual selector witness, LeftAuthority29058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29062

namespace LeftBound29065
def owner : Owner := ⟨.program ⟨214⟩, ⟨15167⟩⟩
def transferEvent : Nat := 29065
def frameStart : Nat := 29006
def rule : BoundRule := .identity (.predecessor 0 29064 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29064 .coefficient)
      LeftBound29062.bound (LeftBound29062.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound29062.derived selector witness)

def rawBound : CoeffClass := LeftBound29062.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound29062.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29065

namespace LeftBound29071
def owner : Owner := ⟨.program ⟨214⟩, ⟨15168⟩⟩
def transferEvent : Nat := 29071
def frameStart : Nat := 29006
def rule : BoundRule := .product (.predecessor 0 29069 .coefficient) (.predecessor 1 29070 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29069 .coefficient)
      LeftAuthority29067.bound (LeftAuthority29067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29070 .coefficient)
      LeftBound29065.bound (LeftBound29065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29065.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29065.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority29067.bound LeftBound29065.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29067.bound, LeftBound29065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority29067.actual selector witness) * (LeftBound29065.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29071

namespace LeftBound29079
def owner : Owner := ⟨.program ⟨214⟩, ⟨15169⟩⟩
def transferEvent : Nat := 29079
def frameStart : Nat := 29006
def rule : BoundRule := .sum [.predecessor 0 29077 .coefficient, .predecessor 1 29078 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29077 .coefficient)
      LeftAuthority29075.bound (LeftAuthority29075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29075.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29078 .coefficient)
      LeftBound29071.bound (LeftBound29071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29071.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority29075.bound, LeftBound29071.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29075.bound, LeftBound29071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority29075.actual selector witness, LeftBound29071.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29079

namespace LeftBound29083
def owner : Owner := ⟨.program ⟨214⟩, ⟨26821⟩⟩
def transferEvent : Nat := 29083
def frameStart : Nat := 29006
def rule : BoundRule := .product (.predecessor 0 29081 .coefficient) (.predecessor 1 29082 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29081 .coefficient)
      LeftBound29079.bound (LeftBound29079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29082 .coefficient)
      LeftAuthority29056.bound (LeftAuthority29056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29056.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29079.bound LeftAuthority29056.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29079.bound, LeftAuthority29056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29079.actual selector witness) * (LeftAuthority29056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29083

namespace LeftBound29094
def owner : Owner := ⟨.program ⟨214⟩, ⟨15380⟩⟩
def transferEvent : Nat := 29094
def frameStart : Nat := 29006
def rule : BoundRule := .product (.predecessor 0 29092 .coefficient) (.predecessor 1 29093 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29092 .coefficient)
      LeftAuthority29067.bound (LeftAuthority29067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29093 .coefficient)
      LeftAuthority29090.bound (LeftAuthority29090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29090.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29090.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority29067.bound LeftAuthority29090.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29067.bound, LeftAuthority29090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority29067.actual selector witness) * (LeftAuthority29090.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29094

namespace LeftBound29102
def owner : Owner := ⟨.program ⟨214⟩, ⟨15381⟩⟩
def transferEvent : Nat := 29102
def frameStart : Nat := 29006
def rule : BoundRule := .sum [.predecessor 0 29100 .coefficient, .predecessor 1 29101 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29100 .coefficient)
      LeftAuthority29098.bound (LeftAuthority29098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29098.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29101 .coefficient)
      LeftBound29094.bound (LeftBound29094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority29098.bound, LeftBound29094.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29098.bound, LeftBound29094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority29098.actual selector witness, LeftBound29094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29102

namespace LeftBound29106
def owner : Owner := ⟨.program ⟨214⟩, ⟨26825⟩⟩
def transferEvent : Nat := 29106
def frameStart : Nat := 29006
def rule : BoundRule := .sum [.predecessor 0 29104 .coefficient, .predecessor 1 29105 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29104 .coefficient)
      LeftBound29102.bound (LeftBound29102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29105 .coefficient)
      LeftBound29083.bound (LeftBound29083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29083.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29102.bound, LeftBound29083.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29102.bound, LeftBound29083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29102.actual selector witness, LeftBound29083.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29106

namespace LeftBound29119
def owner : Owner := ⟨.program ⟨214⟩, ⟨26823⟩⟩
def transferEvent : Nat := 29119
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29117 .coefficient, .predecessor 1 29118 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29117 .coefficient)
      LeftBound28948.bound (LeftBound28948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29118 .coefficient)
      LeftBound28931.bound (LeftBound28931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact28938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28931.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28948.bound, LeftBound28931.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28948.bound, LeftBound28931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28948.actual selector witness, LeftBound28931.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29119

namespace LeftBound29122
def owner : Owner := ⟨.program ⟨214⟩, ⟨26823⟩⟩
def transferEvent : Nat := 29122
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 29116 .summary, .result 28938 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29116 .summary)
      LeftBound28950.bound (LeftBound28950.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20695⟩⟩) (rawTerms := some (Proof.Events113.exact29116RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28938 .summary)
      LeftBound28933.bound (LeftBound28933.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26822⟩⟩) (rawTerms := some (Proof.Events113.exact28938RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28933.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28950.bound, LeftBound28933.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28950.bound, LeftBound28933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28950.actual selector witness, LeftBound28933.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29122

namespace LeftBound29146
def owner : Owner := ⟨.program ⟨214⟩, ⟨10703⟩⟩
def transferEvent : Nat := 29146
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 29144 .coefficient) (.predecessor 1 29145 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29144 .coefficient)
      LeftAuthority1209.bound (LeftAuthority1209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1209.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29145 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1209.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1209.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1209.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29146

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
