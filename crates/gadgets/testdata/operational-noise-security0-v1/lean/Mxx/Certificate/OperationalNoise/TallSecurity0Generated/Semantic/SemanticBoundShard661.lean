import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard660

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96810
def owner : Owner := ⟨.program ⟨214⟩, ⟨22111⟩⟩
def transferEvent : Nat := 96810
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 96808 .coefficient) (.value (.predecessor 1 96809 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96808 .coefficient)
      LeftAuthority96806.bound (LeftAuthority96806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96809 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority96806.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96806.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96806.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96810

namespace LeftBound96814
def owner : Owner := ⟨.program ⟨214⟩, ⟨22112⟩⟩
def transferEvent : Nat := 96814
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96812 .coefficient) (.predecessor 1 96813 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96812 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96813 .coefficient)
      LeftBound96810.bound (LeftBound96810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96810.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound96810.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound96810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound96810.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96814

namespace LeftBound96815
def owner : Owner := ⟨.program ⟨214⟩, ⟨22112⟩⟩
def transferEvent : Nat := 96815
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩ [⟨.result 96807 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96807 .coefficient)
      LeftAuthority96806.bound (LeftAuthority96806.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22109⟩⟩) (rawTerms := some (Proof.Events378.exact96807RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96806.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96806.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96806.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96815

namespace LeftBound96816
def owner : Owner := ⟨.program ⟨214⟩, ⟨22112⟩⟩
def transferEvent : Nat := 96816
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 96815) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96815)
      LeftBound96815.bound (LeftBound96815.actual selector witness) := by
  exact .transfer (LeftBound96815.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound96815.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound96815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound96815.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96816

namespace LeftBound96887
def owner : Owner := ⟨.program ⟨214⟩, ⟨16456⟩⟩
def transferEvent : Nat := 96887
def frameStart : Nat := 96860
def rule : BoundRule := .identity (.predecessor 0 96886 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96886 .coefficient)
      LeftAuthority96884.bound (LeftAuthority96884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96884.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96884.derived selector witness)

def rawBound : CoeffClass := LeftAuthority96884.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority96884.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96887

namespace LeftBound96904
def owner : Owner := ⟨.program ⟨214⟩, ⟨16497⟩⟩
def transferEvent : Nat := 96904
def frameStart : Nat := 96860
def rule : BoundRule := .sum [.predecessor 0 96902 .coefficient, .predecessor 1 96903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96902 .coefficient)
      LeftBound96887.bound (LeftBound96887.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96903 .coefficient)
      LeftAuthority96900.bound (LeftAuthority96900.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96887.bound, LeftAuthority96900.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96887.bound, LeftAuthority96900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96887.actual selector witness, LeftAuthority96900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96904

namespace LeftBound96907
def owner : Owner := ⟨.program ⟨214⟩, ⟨16498⟩⟩
def transferEvent : Nat := 96907
def frameStart : Nat := 96860
def rule : BoundRule := .identity (.predecessor 0 96906 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96906 .coefficient)
      LeftBound96904.bound (LeftBound96904.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96904.derived selector witness)

def rawBound : CoeffClass := LeftBound96904.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound96904.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96907

namespace LeftBound96913
def owner : Owner := ⟨.program ⟨214⟩, ⟨16499⟩⟩
def transferEvent : Nat := 96913
def frameStart : Nat := 96860
def rule : BoundRule := .product (.predecessor 0 96911 .coefficient) (.predecessor 1 96912 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96911 .coefficient)
      LeftAuthority96909.bound (LeftAuthority96909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96909.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96912 .coefficient)
      LeftBound96907.bound (LeftBound96907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96907.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority96909.bound LeftBound96907.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96909.bound, LeftBound96907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority96909.actual selector witness) * (LeftBound96907.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96913

namespace LeftBound96921
def owner : Owner := ⟨.program ⟨214⟩, ⟨16500⟩⟩
def transferEvent : Nat := 96921
def frameStart : Nat := 96860
def rule : BoundRule := .sum [.predecessor 0 96919 .coefficient, .predecessor 1 96920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96919 .coefficient)
      LeftAuthority96917.bound (LeftAuthority96917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96920 .coefficient)
      LeftBound96913.bound (LeftBound96913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96913.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority96917.bound, LeftBound96913.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96917.bound, LeftBound96913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority96917.actual selector witness, LeftBound96913.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96921

namespace LeftBound96925
def owner : Owner := ⟨.program ⟨214⟩, ⟨28917⟩⟩
def transferEvent : Nat := 96925
def frameStart : Nat := 96860
def rule : BoundRule := .product (.predecessor 0 96923 .coefficient) (.predecessor 1 96924 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96923 .coefficient)
      LeftBound96921.bound (LeftBound96921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96924 .coefficient)
      LeftAuthority96898.bound (LeftAuthority96898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96898.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96921.bound LeftAuthority96898.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96921.bound, LeftAuthority96898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96921.actual selector witness) * (LeftAuthority96898.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96925

namespace LeftBound96936
def owner : Owner := ⟨.program ⟨214⟩, ⟨17898⟩⟩
def transferEvent : Nat := 96936
def frameStart : Nat := 96860
def rule : BoundRule := .product (.predecessor 0 96934 .coefficient) (.predecessor 1 96935 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96934 .coefficient)
      LeftAuthority96909.bound (LeftAuthority96909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96909.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96935 .coefficient)
      LeftAuthority96932.bound (LeftAuthority96932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority96909.bound LeftAuthority96932.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96909.bound, LeftAuthority96932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority96909.actual selector witness) * (LeftAuthority96932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96936

namespace LeftBound96944
def owner : Owner := ⟨.program ⟨214⟩, ⟨17899⟩⟩
def transferEvent : Nat := 96944
def frameStart : Nat := 96860
def rule : BoundRule := .sum [.predecessor 0 96942 .coefficient, .predecessor 1 96943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96942 .coefficient)
      LeftAuthority96940.bound (LeftAuthority96940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96940.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96943 .coefficient)
      LeftBound96936.bound (LeftBound96936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96936.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority96940.bound, LeftBound96936.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96940.bound, LeftBound96936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority96940.actual selector witness, LeftBound96936.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96944

namespace LeftBound96948
def owner : Owner := ⟨.program ⟨214⟩, ⟨28921⟩⟩
def transferEvent : Nat := 96948
def frameStart : Nat := 96860
def rule : BoundRule := .sum [.predecessor 0 96946 .coefficient, .predecessor 1 96947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96946 .coefficient)
      LeftBound96944.bound (LeftBound96944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96947 .coefficient)
      LeftBound96925.bound (LeftBound96925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96944.bound, LeftBound96925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96944.bound, LeftBound96925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96944.actual selector witness, LeftBound96925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96948

namespace LeftBound96961
def owner : Owner := ⟨.program ⟨214⟩, ⟨28919⟩⟩
def transferEvent : Nat := 96961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96959 .coefficient, .predecessor 1 96960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96959 .coefficient)
      LeftBound96814.bound (LeftBound96814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96960 .coefficient)
      LeftBound96797.bound (LeftBound96797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96797.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96814.bound, LeftBound96797.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96814.bound, LeftBound96797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96814.actual selector witness, LeftBound96797.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96961

namespace LeftBound96964
def owner : Owner := ⟨.program ⟨214⟩, ⟨28919⟩⟩
def transferEvent : Nat := 96964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 96958 .summary, .result 96804 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96958 .summary)
      LeftBound96816.bound (LeftBound96816.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22112⟩⟩) (rawTerms := some (Proof.Events378.exact96958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96804 .summary)
      LeftBound96799.bound (LeftBound96799.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28918⟩⟩) (rawTerms := some (Proof.Events378.exact96804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96816.bound, LeftBound96799.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96816.bound, LeftBound96799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96816.actual selector witness, LeftBound96799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96964

namespace LeftBound96988
def owner : Owner := ⟨.program ⟨214⟩, ⟨11936⟩⟩
def transferEvent : Nat := 96988
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 96986 .coefficient) (.predecessor 1 96987 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96986 .coefficient)
      LeftAuthority4703.bound (LeftAuthority4703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4703.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96987 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4703.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4703.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4703.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96988

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
