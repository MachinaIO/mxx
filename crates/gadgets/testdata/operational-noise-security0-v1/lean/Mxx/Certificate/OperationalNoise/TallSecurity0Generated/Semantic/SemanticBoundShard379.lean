import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard378

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound56006
def owner : Owner := ⟨.program ⟨214⟩, ⟨14002⟩⟩
def transferEvent : Nat := 56006
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56004 .coefficient) (.predecessor 1 56005 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56004 .coefficient)
      LeftBound56000.bound (LeftBound56000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56000.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56005 .coefficient)
      LeftAuthority2593.bound (LeftAuthority2593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2593.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2593.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound56000.bound LeftAuthority2593.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56000.bound, LeftAuthority2593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound56000.actual selector witness) * (LeftAuthority2593.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56006

namespace LeftBound56007
def owner : Owner := ⟨.program ⟨214⟩, ⟨14002⟩⟩
def transferEvent : Nat := 56007
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩ [⟨.result 2594 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2594 .coefficient)
      LeftAuthority2593.bound (LeftAuthority2593.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13999⟩⟩) (rawTerms := some (Proof.Events010.exact2594RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2593.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2593.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2593.bound []
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2593.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56007

namespace LeftBound56008
def owner : Owner := ⟨.program ⟨214⟩, ⟨14002⟩⟩
def transferEvent : Nat := 56008
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56003 .summary) (.transfer 56007) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56003 .summary)
      LeftBound56001.bound (LeftBound56001.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11392⟩⟩) (rawTerms := some (Proof.Events218.exact56003RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56007)
      LeftBound56007.bound (LeftBound56007.actual selector witness) := by
  exact .transfer (LeftBound56007.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound56001.bound LeftBound56007.bound
def bound : CoeffClass := .finite ⟨13312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56001.bound, LeftBound56007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound56001.actual selector witness) * (LeftBound56007.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56008

namespace LeftBound56014
def owner : Owner := ⟨.program ⟨214⟩, ⟨14003⟩⟩
def transferEvent : Nat := 56014
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 56012 .coefficient) (.predecessor 1 56013 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56012 .coefficient)
      LeftAuthority2593.bound (LeftAuthority2593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2593.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56013 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2593.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2593.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2593.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound56014

namespace LeftBound56019
def owner : Owner := ⟨.program ⟨214⟩, ⟨7252⟩⟩
def transferEvent : Nat := 56019
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56017 .coefficient) (.predecessor 1 56018 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56017 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56018 .coefficient)
      LeftBound12023.bound (LeftBound12023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound12023.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound12023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound12023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56019

namespace LeftBound56024
def owner : Owner := ⟨.program ⟨214⟩, ⟨14004⟩⟩
def transferEvent : Nat := 56024
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 56022 .coefficient, .predecessor 1 56023 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56022 .coefficient)
      LeftBound56019.bound (LeftBound56019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56023 .coefficient)
      LeftBound56014.bound (LeftBound56014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56019.bound, LeftBound56014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56019.bound, LeftBound56014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56019.actual selector witness, LeftBound56014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56024

namespace LeftBound56028
def owner : Owner := ⟨.program ⟨214⟩, ⟨14005⟩⟩
def transferEvent : Nat := 56028
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 56026 .coefficient, .predecessor 1 56027 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56026 .coefficient)
      LeftBound56024.bound (LeftBound56024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56024.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56027 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56024.bound, LeftBound12015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56024.bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56024.actual selector witness, LeftBound12015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56028

namespace LeftBound56029
def owner : Owner := ⟨.program ⟨214⟩, ⟨14005⟩⟩
def transferEvent : Nat := 56029
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩ [⟨.result 12016 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12016 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨72⟩⟩) (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12015.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12015.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56029

namespace LeftBound56034
def owner : Owner := ⟨.program ⟨214⟩, ⟨14006⟩⟩
def transferEvent : Nat := 56034
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56032 .coefficient) (.predecessor 1 56033 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56032 .coefficient)
      LeftBound56028.bound (LeftBound56028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56033 .coefficient)
      LeftBound12012.bound (LeftBound12012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56028.bound LeftBound12012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56028.bound, LeftBound12012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56028.actual selector witness) * (LeftBound12012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56034

namespace LeftBound56035
def owner : Owner := ⟨.program ⟨214⟩, ⟨14006⟩⟩
def transferEvent : Nat := 56035
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩ [⟨.result 12009 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12009 .coefficient)
      LeftAuthority12008.bound (LeftAuthority12008.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7849⟩⟩) (rawTerms := some (Proof.Events046.exact12009RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12008.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12008.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12008.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56035

namespace LeftBound56036
def owner : Owner := ⟨.program ⟨214⟩, ⟨14006⟩⟩
def transferEvent : Nat := 56036
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56031 .summary) (.transfer 56035) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56031 .summary)
      LeftBound56029.bound (LeftBound56029.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14005⟩⟩) (rawTerms := some (Proof.Events218.exact56031RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56035)
      LeftBound56035.bound (LeftBound56035.actual selector witness) := by
  exact .transfer (LeftBound56035.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56029.bound LeftBound56035.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56029.bound, LeftBound56035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56029.actual selector witness) * (LeftBound56035.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56036

namespace LeftBound56044
def owner : Owner := ⟨.program ⟨214⟩, ⟨14007⟩⟩
def transferEvent : Nat := 56044
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 56042 .coefficient, .predecessor 1 56043 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56042 .coefficient)
      LeftBound56034.bound (LeftBound56034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56034.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56043 .coefficient)
      LeftBound56006.bound (LeftBound56006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56034.bound, LeftBound56006.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56034.bound, LeftBound56006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56034.actual selector witness, LeftBound56006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56044

namespace LeftBound56046
def owner : Owner := ⟨.program ⟨214⟩, ⟨14007⟩⟩
def transferEvent : Nat := 56046
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 56041 .summary, .result 56011 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56041 .summary)
      LeftBound56036.bound (LeftBound56036.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14006⟩⟩) (rawTerms := some (Proof.Events218.exact56041RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56011 .summary)
      LeftBound56008.bound (LeftBound56008.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14002⟩⟩) (rawTerms := some (Proof.Events218.exact56011RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56008.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56036.bound, LeftBound56008.bound]
def bound : CoeffClass := .finite ⟨95433728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56036.bound, LeftBound56008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56036.actual selector witness, LeftBound56008.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56046

namespace LeftBound56050
def owner : Owner := ⟨.program ⟨214⟩, ⟨25995⟩⟩
def transferEvent : Nat := 56050
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56048 .coefficient) (.predecessor 1 56049 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56048 .coefficient)
      LeftBound56044.bound (LeftBound56044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56044.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56049 .coefficient)
      LeftAuthority55982.bound (LeftAuthority55982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55982.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55982.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56044.bound LeftAuthority55982.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56044.bound, LeftAuthority55982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56044.actual selector witness) * (LeftAuthority55982.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56050

namespace LeftBound56051
def owner : Owner := ⟨.program ⟨214⟩, ⟨25995⟩⟩
def transferEvent : Nat := 56051
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩ [⟨.result 55983 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55983 .coefficient)
      LeftAuthority55982.bound (LeftAuthority55982.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25994⟩⟩) (rawTerms := some (Proof.Events218.exact55983RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55982.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55982.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55982.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55982.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56051

namespace LeftBound56052
def owner : Owner := ⟨.program ⟨214⟩, ⟨25995⟩⟩
def transferEvent : Nat := 56052
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56047 .summary) (.transfer 56051) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56047 .summary)
      LeftBound56046.bound (LeftBound56046.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14007⟩⟩) (rawTerms := some (Proof.Events218.exact56047RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56051)
      LeftBound56051.bound (LeftBound56051.actual selector witness) := by
  exact .transfer (LeftBound56051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56046.bound LeftBound56051.bound
def bound : CoeffClass := .finite ⟨350243308699648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56046.bound, LeftBound56051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56046.actual selector witness) * (LeftBound56051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56052

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
