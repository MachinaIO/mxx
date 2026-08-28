import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard304

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44885
def owner : Owner := ⟨.program ⟨214⟩, ⟨18570⟩⟩
def transferEvent : Nat := 44885
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44883 .coefficient) (.predecessor 1 44884 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44883 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44884 .coefficient)
      LeftBound44881.bound (LeftBound44881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact44882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44881.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound44881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound44881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound44881.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44885

namespace LeftBound44886
def owner : Owner := ⟨.program ⟨214⟩, ⟨18570⟩⟩
def transferEvent : Nat := 44886
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18567⟩⟩]⟩ [⟨.result 44878 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44878 .coefficient)
      LeftAuthority44877.bound (LeftAuthority44877.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18567⟩⟩) (rawTerms := some (Proof.Events175.exact44878RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44877.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority44877.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44877.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44886

namespace LeftBound44887
def owner : Owner := ⟨.program ⟨214⟩, ⟨18570⟩⟩
def transferEvent : Nat := 44887
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 44886) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 44886)
      LeftBound44886.bound (LeftBound44886.actual selector witness) := by
  exact .transfer (LeftBound44886.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound44886.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound44886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound44886.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44887

namespace LeftBound45915
def owner : Owner := ⟨.program ⟨214⟩, ⟨15319⟩⟩
def transferEvent : Nat := 45915
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45913 .coefficient, .predecessor 1 45914 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45913 .coefficient)
      LeftAuthority45911.bound (LeftAuthority45911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45911.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45914 .coefficient)
      LeftAuthority45888.bound (LeftAuthority45888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45888.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority45911.bound, LeftAuthority45888.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45911.bound, LeftAuthority45888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority45911.actual selector witness, LeftAuthority45888.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45915

namespace LeftBound45919
def owner : Owner := ⟨.program ⟨214⟩, ⟨15375⟩⟩
def transferEvent : Nat := 45919
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45917 .coefficient, .predecessor 1 45918 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45917 .coefficient)
      LeftBound45915.bound (LeftBound45915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45915.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45918 .coefficient)
      LeftAuthority45865.bound (LeftAuthority45865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45865.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45915.bound, LeftAuthority45865.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45915.bound, LeftAuthority45865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45915.actual selector witness, LeftAuthority45865.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45919

namespace LeftBound45923
def owner : Owner := ⟨.program ⟨214⟩, ⟨17346⟩⟩
def transferEvent : Nat := 45923
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45921 .coefficient, .predecessor 1 45922 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45921 .coefficient)
      LeftBound45919.bound (LeftBound45919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45922 .coefficient)
      LeftAuthority45842.bound (LeftAuthority45842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45919.bound, LeftAuthority45842.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45919.bound, LeftAuthority45842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45919.actual selector witness, LeftAuthority45842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45923

namespace LeftBound45927
def owner : Owner := ⟨.program ⟨214⟩, ⟨17347⟩⟩
def transferEvent : Nat := 45927
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45925 .coefficient, .predecessor 1 45926 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45925 .coefficient)
      LeftBound45923.bound (LeftBound45923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45926 .coefficient)
      LeftAuthority45819.bound (LeftAuthority45819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45819.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45819.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45923.bound, LeftAuthority45819.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45923.bound, LeftAuthority45819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45923.actual selector witness, LeftAuthority45819.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45927

namespace LeftBound45931
def owner : Owner := ⟨.program ⟨214⟩, ⟨17348⟩⟩
def transferEvent : Nat := 45931
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45929 .coefficient, .predecessor 1 45930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45929 .coefficient)
      LeftBound45927.bound (LeftBound45927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45930 .coefficient)
      LeftAuthority45796.bound (LeftAuthority45796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45796.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45796.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45927.bound, LeftAuthority45796.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45927.bound, LeftAuthority45796.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45927.actual selector witness, LeftAuthority45796.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45931

namespace LeftBound45935
def owner : Owner := ⟨.program ⟨214⟩, ⟨17349⟩⟩
def transferEvent : Nat := 45935
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45933 .coefficient, .predecessor 1 45934 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45933 .coefficient)
      LeftBound45931.bound (LeftBound45931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45934 .coefficient)
      LeftAuthority45773.bound (LeftAuthority45773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45773.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45931.bound, LeftAuthority45773.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45931.bound, LeftAuthority45773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45931.actual selector witness, LeftAuthority45773.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45935

namespace LeftBound45939
def owner : Owner := ⟨.program ⟨214⟩, ⟨17350⟩⟩
def transferEvent : Nat := 45939
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45937 .coefficient, .predecessor 1 45938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45937 .coefficient)
      LeftBound45935.bound (LeftBound45935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45938 .coefficient)
      LeftAuthority45750.bound (LeftAuthority45750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45750.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45750.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45935.bound, LeftAuthority45750.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45935.bound, LeftAuthority45750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45935.actual selector witness, LeftAuthority45750.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45939

namespace LeftBound45943
def owner : Owner := ⟨.program ⟨214⟩, ⟨17351⟩⟩
def transferEvent : Nat := 45943
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45941 .coefficient, .predecessor 1 45942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45941 .coefficient)
      LeftBound45939.bound (LeftBound45939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45942 .coefficient)
      LeftAuthority45727.bound (LeftAuthority45727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45939.bound, LeftAuthority45727.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45939.bound, LeftAuthority45727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45939.actual selector witness, LeftAuthority45727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45943

namespace LeftBound45947
def owner : Owner := ⟨.program ⟨214⟩, ⟨18367⟩⟩
def transferEvent : Nat := 45947
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45945 .coefficient, .predecessor 1 45946 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45945 .coefficient)
      LeftBound45943.bound (LeftBound45943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45946 .coefficient)
      LeftAuthority45704.bound (LeftAuthority45704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45704.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45704.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45943.bound, LeftAuthority45704.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45943.bound, LeftAuthority45704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45943.actual selector witness, LeftAuthority45704.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45947

namespace LeftBound45951
def owner : Owner := ⟨.program ⟨214⟩, ⟨18368⟩⟩
def transferEvent : Nat := 45951
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45949 .coefficient, .predecessor 1 45950 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45949 .coefficient)
      LeftBound45947.bound (LeftBound45947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45950 .coefficient)
      LeftAuthority45681.bound (LeftAuthority45681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45681.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45947.bound, LeftAuthority45681.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45947.bound, LeftAuthority45681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45947.actual selector witness, LeftAuthority45681.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45951

namespace LeftBound45955
def owner : Owner := ⟨.program ⟨214⟩, ⟨18369⟩⟩
def transferEvent : Nat := 45955
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45953 .coefficient, .predecessor 1 45954 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45953 .coefficient)
      LeftBound45951.bound (LeftBound45951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45954 .coefficient)
      LeftAuthority45658.bound (LeftAuthority45658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45951.bound, LeftAuthority45658.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45951.bound, LeftAuthority45658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45951.actual selector witness, LeftAuthority45658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45955

namespace LeftBound45959
def owner : Owner := ⟨.program ⟨214⟩, ⟨18370⟩⟩
def transferEvent : Nat := 45959
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45957 .coefficient, .predecessor 1 45958 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45957 .coefficient)
      LeftBound45955.bound (LeftBound45955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45955.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45958 .coefficient)
      LeftAuthority45635.bound (LeftAuthority45635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45635.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45955.bound, LeftAuthority45635.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45955.bound, LeftAuthority45635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45955.actual selector witness, LeftAuthority45635.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45959

namespace LeftBound45963
def owner : Owner := ⟨.program ⟨214⟩, ⟨18371⟩⟩
def transferEvent : Nat := 45963
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45961 .coefficient, .predecessor 1 45962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45961 .coefficient)
      LeftBound45959.bound (LeftBound45959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45962 .coefficient)
      LeftAuthority45612.bound (LeftAuthority45612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45959.bound, LeftAuthority45612.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45959.bound, LeftAuthority45612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45959.actual selector witness, LeftAuthority45612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45963

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
