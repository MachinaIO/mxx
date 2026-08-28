import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard651

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound95736
def owner : Owner := ⟨.program ⟨214⟩, ⟨10019⟩⟩
def transferEvent : Nat := 95736
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95731 .summary) (.transfer 95735) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95731 .summary)
      LeftBound95729.bound (LeftBound95729.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10018⟩⟩) (rawTerms := some (Proof.Events373.exact95731RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95735)
      LeftBound95735.bound (LeftBound95735.actual selector witness) := by
  exact .transfer (LeftBound95735.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95729.bound LeftBound95735.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95729.bound, LeftBound95735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95729.actual selector witness) * (LeftBound95735.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95736

namespace LeftBound95744
def owner : Owner := ⟨.program ⟨214⟩, ⟨12745⟩⟩
def transferEvent : Nat := 95744
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95742 .coefficient, .predecessor 1 95743 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95742 .coefficient)
      LeftBound95734.bound (LeftBound95734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95734.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95743 .coefficient)
      LeftBound95706.bound (LeftBound95706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95706.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95734.bound, LeftBound95706.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95734.bound, LeftBound95706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95734.actual selector witness, LeftBound95706.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95744

namespace LeftBound95746
def owner : Owner := ⟨.program ⟨214⟩, ⟨12745⟩⟩
def transferEvent : Nat := 95746
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 95741 .summary, .result 95711 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95741 .summary)
      LeftBound95736.bound (LeftBound95736.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10019⟩⟩) (rawTerms := some (Proof.Events373.exact95741RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95711 .summary)
      LeftBound95708.bound (LeftBound95708.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12744⟩⟩) (rawTerms := some (Proof.Events373.exact95711RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95708.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95736.bound, LeftBound95708.bound]
def bound : CoeffClass := .finite ⟨95458688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95736.bound, LeftBound95708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95736.actual selector witness, LeftBound95708.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95746

namespace LeftBound95750
def owner : Owner := ⟨.program ⟨214⟩, ⟨25515⟩⟩
def transferEvent : Nat := 95750
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95748 .coefficient) (.predecessor 1 95749 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95748 .coefficient)
      LeftBound95744.bound (LeftBound95744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95749 .coefficient)
      LeftAuthority95682.bound (LeftAuthority95682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95744.bound LeftAuthority95682.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95744.bound, LeftAuthority95682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95744.actual selector witness) * (LeftAuthority95682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95750

namespace LeftBound95751
def owner : Owner := ⟨.program ⟨214⟩, ⟨25515⟩⟩
def transferEvent : Nat := 95751
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩ [⟨.result 95683 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95683 .coefficient)
      LeftAuthority95682.bound (LeftAuthority95682.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25514⟩⟩) (rawTerms := some (Proof.Events373.exact95683RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95682.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95682.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95682.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95751

namespace LeftBound95752
def owner : Owner := ⟨.program ⟨214⟩, ⟨25515⟩⟩
def transferEvent : Nat := 95752
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95747 .summary) (.transfer 95751) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95747 .summary)
      LeftBound95746.bound (LeftBound95746.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12745⟩⟩) (rawTerms := some (Proof.Events374.exact95747RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95751)
      LeftBound95751.bound (LeftBound95751.actual selector witness) := by
  exact .transfer (LeftBound95751.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95746.bound LeftBound95751.bound
def bound : CoeffClass := .finite ⟨350334912299008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95746.bound, LeftBound95751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95746.actual selector witness) * (LeftBound95751.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95752

namespace LeftBound95763
def owner : Owner := ⟨.program ⟨214⟩, ⟨20023⟩⟩
def transferEvent : Nat := 95763
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 95761 .coefficient) (.value (.predecessor 1 95762 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95761 .coefficient)
      LeftAuthority95759.bound (LeftAuthority95759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95762 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority95759.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95759.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95759.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95763

namespace LeftBound95767
def owner : Owner := ⟨.program ⟨214⟩, ⟨20024⟩⟩
def transferEvent : Nat := 95767
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95765 .coefficient) (.predecessor 1 95766 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95765 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95766 .coefficient)
      LeftBound95763.bound (LeftBound95763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95763.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound95763.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound95763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound95763.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95767

namespace LeftBound95768
def owner : Owner := ⟨.program ⟨214⟩, ⟨20024⟩⟩
def transferEvent : Nat := 95768
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩ [⟨.result 95760 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95760 .coefficient)
      LeftAuthority95759.bound (LeftAuthority95759.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20021⟩⟩) (rawTerms := some (Proof.Events374.exact95760RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95759.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95759.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95759.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95768

namespace LeftBound95769
def owner : Owner := ⟨.program ⟨214⟩, ⟨20024⟩⟩
def transferEvent : Nat := 95769
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 95768) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95768)
      LeftBound95768.bound (LeftBound95768.actual selector witness) := by
  exact .transfer (LeftBound95768.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound95768.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound95768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound95768.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95769

namespace LeftBound95824
def owner : Owner := ⟨.program ⟨214⟩, ⟨12739⟩⟩
def transferEvent : Nat := 95824
def frameStart : Nat := 95807
def rule : BoundRule := .product (.predecessor 0 95822 .coefficient) (.predecessor 1 95823 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95822 .coefficient)
      LeftAuthority95820.bound (LeftAuthority95820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95820.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95823 .coefficient)
      LeftAuthority95817.bound (LeftAuthority95817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95817.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority95820.bound LeftAuthority95817.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95820.bound, LeftAuthority95817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority95820.actual selector witness) * (LeftAuthority95817.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95824

namespace LeftBound95828
def owner : Owner := ⟨.program ⟨214⟩, ⟨12740⟩⟩
def transferEvent : Nat := 95828
def frameStart : Nat := 95807
def rule : BoundRule := .identity (.predecessor 0 95827 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95827 .coefficient)
      LeftBound95824.bound (LeftBound95824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95824.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95824.derived selector witness)

def rawBound : CoeffClass := LeftBound95824.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound95824.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95828

namespace LeftBound95845
def owner : Owner := ⟨.program ⟨214⟩, ⟨12850⟩⟩
def transferEvent : Nat := 95845
def frameStart : Nat := 95807
def rule : BoundRule := .sum [.predecessor 0 95843 .coefficient, .predecessor 1 95844 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95843 .coefficient)
      LeftBound95828.bound (LeftBound95828.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound95828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95844 .coefficient)
      LeftAuthority95841.bound (LeftAuthority95841.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority95841.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95828.bound, LeftAuthority95841.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95828.bound, LeftAuthority95841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95828.actual selector witness, LeftAuthority95841.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95845

namespace LeftBound95848
def owner : Owner := ⟨.program ⟨214⟩, ⟨12851⟩⟩
def transferEvent : Nat := 95848
def frameStart : Nat := 95807
def rule : BoundRule := .identity (.predecessor 0 95847 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95847 .coefficient)
      LeftBound95845.bound (LeftBound95845.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound95845.derived selector witness)

def rawBound : CoeffClass := LeftBound95845.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound95845.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95848

namespace LeftBound95854
def owner : Owner := ⟨.program ⟨214⟩, ⟨12852⟩⟩
def transferEvent : Nat := 95854
def frameStart : Nat := 95807
def rule : BoundRule := .product (.predecessor 0 95852 .coefficient) (.predecessor 1 95853 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95852 .coefficient)
      LeftAuthority95850.bound (LeftAuthority95850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95853 .coefficient)
      LeftBound95848.bound (LeftBound95848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95848.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority95850.bound LeftBound95848.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95850.bound, LeftBound95848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority95850.actual selector witness) * (LeftBound95848.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95854

namespace LeftBound95870
def owner : Owner := ⟨.program ⟨214⟩, ⟨7874⟩⟩
def transferEvent : Nat := 95870
def frameStart : Nat := 95807
def rule : BoundRule := .scale (.predecessor 0 95868 .coefficient) (.value (.predecessor 1 95869 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95868 .coefficient)
      LeftAuthority95866.bound (LeftAuthority95866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95869 .coefficient)
      LeftAuthority95857.bound (LeftAuthority95857.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority95857.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority95866.bound LeftAuthority95857.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95866.bound, LeftAuthority95857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95866.actual selector witness) * (LeftAuthority95857.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95870

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
