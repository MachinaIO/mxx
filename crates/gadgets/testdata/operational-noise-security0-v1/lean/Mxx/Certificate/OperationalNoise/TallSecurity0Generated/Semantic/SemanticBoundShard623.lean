import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard573

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound91887
def owner : Owner := ⟨.program ⟨214⟩, ⟨28295⟩⟩
def transferEvent : Nat := 91887
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91885 .coefficient) (.predecessor 1 91886 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91885 .coefficient)
      LeftBound84032.bound (LeftBound84032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91886 .coefficient)
      LeftAuthority91883.bound (LeftAuthority91883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91883.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84032.bound LeftAuthority91883.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84032.bound, LeftAuthority91883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84032.actual selector witness) * (LeftAuthority91883.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91887

namespace LeftBound91888
def owner : Owner := ⟨.program ⟨214⟩, ⟨28295⟩⟩
def transferEvent : Nat := 91888
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩ [⟨.result 91884 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91884 .coefficient)
      LeftAuthority91883.bound (LeftAuthority91883.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28293⟩⟩) (rawTerms := some (Proof.Events358.exact91884RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91883.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91883.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91883.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91888

namespace LeftBound91889
def owner : Owner := ⟨.program ⟨214⟩, ⟨28295⟩⟩
def transferEvent : Nat := 91889
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84036 .summary) (.transfer 91888) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84036 .summary)
      LeftBound84035.bound (LeftBound84035.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26222⟩⟩) (rawTerms := some (Proof.Events328.exact84036RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91888)
      LeftBound91888.bound (LeftBound91888.actual selector witness) := by
  exact .transfer (LeftBound91888.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84035.bound LeftBound91888.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84035.bound, LeftBound91888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84035.actual selector witness) * (LeftBound91888.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91889

namespace LeftBound91900
def owner : Owner := ⟨.program ⟨214⟩, ⟨21618⟩⟩
def transferEvent : Nat := 91900
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 91898 .coefficient) (.value (.predecessor 1 91899 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91898 .coefficient)
      LeftAuthority91896.bound (LeftAuthority91896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91899 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority91896.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91896.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91896.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91900

namespace LeftBound91904
def owner : Owner := ⟨.program ⟨214⟩, ⟨21619⟩⟩
def transferEvent : Nat := 91904
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91902 .coefficient) (.predecessor 1 91903 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91902 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91903 .coefficient)
      LeftBound91900.bound (LeftBound91900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91900.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound91900.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound91900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound91900.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91904

namespace LeftBound91905
def owner : Owner := ⟨.program ⟨214⟩, ⟨21619⟩⟩
def transferEvent : Nat := 91905
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩ [⟨.result 91897 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91897 .coefficient)
      LeftAuthority91896.bound (LeftAuthority91896.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21616⟩⟩) (rawTerms := some (Proof.Events358.exact91897RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91896.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91896.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91896.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91905

namespace LeftBound91906
def owner : Owner := ⟨.program ⟨214⟩, ⟨21619⟩⟩
def transferEvent : Nat := 91906
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 91905) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91905)
      LeftBound91905.bound (LeftBound91905.actual selector witness) := by
  exact .transfer (LeftBound91905.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound91905.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound91905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound91905.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91906

namespace LeftBound92001
def owner : Owner := ⟨.program ⟨214⟩, ⟨16179⟩⟩
def transferEvent : Nat := 92001
def frameStart : Nat := 91962
def rule : BoundRule := .identity (.predecessor 0 92000 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92000 .coefficient)
      LeftAuthority91998.bound (LeftAuthority91998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact91999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91998.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91998.derived selector witness)

def rawBound : CoeffClass := LeftAuthority91998.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority91998.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92001

namespace LeftBound92018
def owner : Owner := ⟨.program ⟨214⟩, ⟨16218⟩⟩
def transferEvent : Nat := 92018
def frameStart : Nat := 91962
def rule : BoundRule := .sum [.predecessor 0 92016 .coefficient, .predecessor 1 92017 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92016 .coefficient)
      LeftBound92001.bound (LeftBound92001.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92017 .coefficient)
      LeftAuthority92014.bound (LeftAuthority92014.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority92014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92001.bound, LeftAuthority92014.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92001.bound, LeftAuthority92014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92001.actual selector witness, LeftAuthority92014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92018

namespace LeftBound92021
def owner : Owner := ⟨.program ⟨214⟩, ⟨16219⟩⟩
def transferEvent : Nat := 92021
def frameStart : Nat := 91962
def rule : BoundRule := .identity (.predecessor 0 92020 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92020 .coefficient)
      LeftBound92018.bound (LeftBound92018.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92018.derived selector witness)

def rawBound : CoeffClass := LeftBound92018.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound92018.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92021

namespace LeftBound92027
def owner : Owner := ⟨.program ⟨214⟩, ⟨16220⟩⟩
def transferEvent : Nat := 92027
def frameStart : Nat := 91962
def rule : BoundRule := .product (.predecessor 0 92025 .coefficient) (.predecessor 1 92026 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92025 .coefficient)
      LeftAuthority92023.bound (LeftAuthority92023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92026 .coefficient)
      LeftBound92021.bound (LeftBound92021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority92023.bound LeftBound92021.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92023.bound, LeftBound92021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority92023.actual selector witness) * (LeftBound92021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92027

namespace LeftBound92035
def owner : Owner := ⟨.program ⟨214⟩, ⟨16221⟩⟩
def transferEvent : Nat := 92035
def frameStart : Nat := 91962
def rule : BoundRule := .sum [.predecessor 0 92033 .coefficient, .predecessor 1 92034 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92033 .coefficient)
      LeftAuthority92031.bound (LeftAuthority92031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92034 .coefficient)
      LeftBound92027.bound (LeftBound92027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92027.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92031.bound, LeftBound92027.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92031.bound, LeftBound92027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92031.actual selector witness, LeftBound92027.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92035

namespace LeftBound92039
def owner : Owner := ⟨.program ⟨214⟩, ⟨28294⟩⟩
def transferEvent : Nat := 92039
def frameStart : Nat := 91962
def rule : BoundRule := .product (.predecessor 0 92037 .coefficient) (.predecessor 1 92038 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92037 .coefficient)
      LeftBound92035.bound (LeftBound92035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92038 .coefficient)
      LeftAuthority92012.bound (LeftAuthority92012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92035.bound LeftAuthority92012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92035.bound, LeftAuthority92012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92035.actual selector witness) * (LeftAuthority92012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92039

namespace LeftBound92050
def owner : Owner := ⟨.program ⟨214⟩, ⟨17664⟩⟩
def transferEvent : Nat := 92050
def frameStart : Nat := 91962
def rule : BoundRule := .product (.predecessor 0 92048 .coefficient) (.predecessor 1 92049 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92048 .coefficient)
      LeftAuthority92023.bound (LeftAuthority92023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92049 .coefficient)
      LeftAuthority92046.bound (LeftAuthority92046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92046.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92046.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority92023.bound LeftAuthority92046.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92023.bound, LeftAuthority92046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority92023.actual selector witness) * (LeftAuthority92046.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92050

namespace LeftBound92058
def owner : Owner := ⟨.program ⟨214⟩, ⟨17665⟩⟩
def transferEvent : Nat := 92058
def frameStart : Nat := 91962
def rule : BoundRule := .sum [.predecessor 0 92056 .coefficient, .predecessor 1 92057 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92056 .coefficient)
      LeftAuthority92054.bound (LeftAuthority92054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92054.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92057 .coefficient)
      LeftBound92050.bound (LeftBound92050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92054.bound, LeftBound92050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92054.bound, LeftBound92050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92054.actual selector witness, LeftBound92050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92058

namespace LeftBound92062
def owner : Owner := ⟨.program ⟨214⟩, ⟨28299⟩⟩
def transferEvent : Nat := 92062
def frameStart : Nat := 91962
def rule : BoundRule := .sum [.predecessor 0 92060 .coefficient, .predecessor 1 92061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92060 .coefficient)
      LeftBound92058.bound (LeftBound92058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92061 .coefficient)
      LeftBound92039.bound (LeftBound92039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92039.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92058.bound, LeftBound92039.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92058.bound, LeftBound92039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92058.actual selector witness, LeftBound92039.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92062

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
