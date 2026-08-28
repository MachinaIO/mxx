import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound83851
def owner : Owner := ⟨.program ⟨214⟩, ⟨19674⟩⟩
def transferEvent : Nat := 83851
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 83849 .coefficient) (.value (.predecessor 1 83850 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83849 .coefficient)
      LeftAuthority83847.bound (LeftAuthority83847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83850 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority83847.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83847.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83847.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83851

namespace LeftBound83855
def owner : Owner := ⟨.program ⟨214⟩, ⟨19675⟩⟩
def transferEvent : Nat := 83855
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83853 .coefficient) (.predecessor 1 83854 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83853 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83854 .coefficient)
      LeftBound83851.bound (LeftBound83851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83851.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound83851.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound83851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound83851.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83855

namespace LeftBound83856
def owner : Owner := ⟨.program ⟨214⟩, ⟨19675⟩⟩
def transferEvent : Nat := 83856
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩ [⟨.result 83848 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83848 .coefficient)
      LeftAuthority83847.bound (LeftAuthority83847.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19672⟩⟩) (rawTerms := some (Proof.Events327.exact83848RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83847.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority83847.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83847.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83856

namespace LeftBound83857
def owner : Owner := ⟨.program ⟨214⟩, ⟨19675⟩⟩
def transferEvent : Nat := 83857
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 83856) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83856)
      LeftBound83856.bound (LeftBound83856.actual selector witness) := by
  exact .transfer (LeftBound83856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound83856.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound83856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound83856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83857

namespace LeftBound83936
def owner : Owner := ⟨.program ⟨214⟩, ⟨14642⟩⟩
def transferEvent : Nat := 83936
def frameStart : Nat := 83907
def rule : BoundRule := .product (.predecessor 0 83934 .coefficient) (.predecessor 1 83935 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83934 .coefficient)
      LeftAuthority83932.bound (LeftAuthority83932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83935 .coefficient)
      LeftAuthority83929.bound (LeftAuthority83929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83932.bound LeftAuthority83929.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83932.bound, LeftAuthority83929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority83932.actual selector witness) * (LeftAuthority83929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83936

namespace LeftBound83940
def owner : Owner := ⟨.program ⟨214⟩, ⟨14643⟩⟩
def transferEvent : Nat := 83940
def frameStart : Nat := 83907
def rule : BoundRule := .identity (.predecessor 0 83939 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83939 .coefficient)
      LeftBound83936.bound (LeftBound83936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83936.derived selector witness)

def rawBound : CoeffClass := LeftBound83936.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound83936.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83940

namespace LeftBound83957
def owner : Owner := ⟨.program ⟨214⟩, ⟨14748⟩⟩
def transferEvent : Nat := 83957
def frameStart : Nat := 83907
def rule : BoundRule := .sum [.predecessor 0 83955 .coefficient, .predecessor 1 83956 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83955 .coefficient)
      LeftBound83940.bound (LeftBound83940.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83956 .coefficient)
      LeftAuthority83953.bound (LeftAuthority83953.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority83953.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83940.bound, LeftAuthority83953.bound]
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83940.bound, LeftAuthority83953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83940.actual selector witness, LeftAuthority83953.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83957

namespace LeftBound83960
def owner : Owner := ⟨.program ⟨214⟩, ⟨14749⟩⟩
def transferEvent : Nat := 83960
def frameStart : Nat := 83907
def rule : BoundRule := .identity (.predecessor 0 83959 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83959 .coefficient)
      LeftBound83957.bound (LeftBound83957.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83957.derived selector witness)

def rawBound : CoeffClass := LeftBound83957.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound83957.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83960

namespace LeftBound83966
def owner : Owner := ⟨.program ⟨214⟩, ⟨14750⟩⟩
def transferEvent : Nat := 83966
def frameStart : Nat := 83907
def rule : BoundRule := .product (.predecessor 0 83964 .coefficient) (.predecessor 1 83965 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83964 .coefficient)
      LeftAuthority83962.bound (LeftAuthority83962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83962.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83965 .coefficient)
      LeftBound83960.bound (LeftBound83960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority83962.bound LeftBound83960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83962.bound, LeftBound83960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority83962.actual selector witness) * (LeftBound83960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83966

namespace LeftBound83980
def owner : Owner := ⟨.program ⟨214⟩, ⟨7859⟩⟩
def transferEvent : Nat := 83980
def frameStart : Nat := 83907
def rule : BoundRule := .scale (.predecessor 0 83978 .coefficient) (.value (.predecessor 1 83979 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83978 .coefficient)
      LeftAuthority83976.bound (LeftAuthority83976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact83977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83979 .coefficient)
      LeftAuthority83910.bound (LeftAuthority83910.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority83910.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority83976.bound LeftAuthority83910.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83976.bound, LeftAuthority83910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83976.actual selector witness) * (LeftAuthority83910.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83980

namespace LeftBound83983
def owner : Owner := ⟨.program ⟨214⟩, ⟨6762⟩⟩
def transferEvent : Nat := 83983
def frameStart : Nat := 83907
def rule : BoundRule := .identity (.predecessor 0 83982 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83982 .coefficient)
      LeftAuthority83970.bound (LeftAuthority83970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact83971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83970.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83970.derived selector witness)

def rawBound : CoeffClass := LeftAuthority83970.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority83970.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83983

namespace LeftBound83987
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def transferEvent : Nat := 83987
def frameStart : Nat := 83907
def rule : BoundRule := .product (.predecessor 0 83985 .coefficient) (.predecessor 1 83986 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83985 .coefficient)
      LeftBound83983.bound (LeftBound83983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact83984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83986 .coefficient)
      LeftBound83980.bound (LeftBound83980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact83981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83983.bound LeftBound83980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83983.bound, LeftBound83980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83983.actual selector witness) * (LeftBound83980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83987

namespace LeftBound83992
def owner : Owner := ⟨.program ⟨214⟩, ⟨14751⟩⟩
def transferEvent : Nat := 83992
def frameStart : Nat := 83907
def rule : BoundRule := .sum [.predecessor 0 83990 .coefficient, .predecessor 1 83991 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83990 .coefficient)
      LeftBound83987.bound (LeftBound83987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact83989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83991 .coefficient)
      LeftBound83966.bound (LeftBound83966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact83968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83987.bound, LeftBound83966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83987.bound, LeftBound83966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83987.actual selector witness, LeftBound83966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83992

namespace LeftBound83996
def owner : Owner := ⟨.program ⟨214⟩, ⟨26223⟩⟩
def transferEvent : Nat := 83996
def frameStart : Nat := 83907
def rule : BoundRule := .product (.predecessor 0 83994 .coefficient) (.predecessor 1 83995 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83994 .coefficient)
      LeftBound83992.bound (LeftBound83992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact83993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83995 .coefficient)
      LeftAuthority83951.bound (LeftAuthority83951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83951.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83951.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83992.bound LeftAuthority83951.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83992.bound, LeftAuthority83951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83992.actual selector witness) * (LeftAuthority83951.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83996

namespace LeftBound84007
def owner : Owner := ⟨.program ⟨214⟩, ⟨16180⟩⟩
def transferEvent : Nat := 84007
def frameStart : Nat := 83907
def rule : BoundRule := .product (.predecessor 0 84005 .coefficient) (.predecessor 1 84006 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84005 .coefficient)
      LeftAuthority83962.bound (LeftAuthority83962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83962.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84006 .coefficient)
      LeftAuthority84003.bound (LeftAuthority84003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84003.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83962.bound LeftAuthority84003.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83962.bound, LeftAuthority84003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority83962.actual selector witness) * (LeftAuthority84003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84007

namespace LeftBound84015
def owner : Owner := ⟨.program ⟨214⟩, ⟨16181⟩⟩
def transferEvent : Nat := 84015
def frameStart : Nat := 83907
def rule : BoundRule := .sum [.predecessor 0 84013 .coefficient, .predecessor 1 84014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84013 .coefficient)
      LeftAuthority84011.bound (LeftAuthority84011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84011.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84014 .coefficient)
      LeftBound84007.bound (LeftBound84007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84011.bound, LeftBound84007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84011.bound, LeftBound84007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority84011.actual selector witness, LeftBound84007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84015

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
