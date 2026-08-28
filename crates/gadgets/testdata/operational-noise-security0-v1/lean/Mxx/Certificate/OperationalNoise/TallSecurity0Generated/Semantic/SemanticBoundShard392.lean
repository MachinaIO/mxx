import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard390
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard391

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound57695
def owner : Owner := ⟨.program ⟨214⟩, ⟨25303⟩⟩
def transferEvent : Nat := 57695
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 57689 .summary, .result 57503 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57689 .summary)
      LeftBound57515.bound (LeftBound57515.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19247⟩⟩) (rawTerms := some (Proof.Events225.exact57689RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57503 .summary)
      LeftBound57498.bound (LeftBound57498.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25302⟩⟩) (rawTerms := some (Proof.Events224.exact57503RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57515.bound, LeftBound57498.bound]
def bound : CoeffClass := .finite ⟨352024077676544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57515.bound, LeftBound57498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57515.actual selector witness, LeftBound57498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57695

namespace LeftBound57699
def owner : Owner := ⟨.program ⟨214⟩, ⟨27013⟩⟩
def transferEvent : Nat := 57699
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57697 .coefficient) (.predecessor 1 57698 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57697 .coefficient)
      LeftBound57692.bound (LeftBound57692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57698 .coefficient)
      LeftAuthority57418.bound (LeftAuthority57418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57418.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57692.bound LeftAuthority57418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57692.bound, LeftAuthority57418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57692.actual selector witness) * (LeftAuthority57418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57699

namespace LeftBound57700
def owner : Owner := ⟨.program ⟨214⟩, ⟨27013⟩⟩
def transferEvent : Nat := 57700
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27011⟩⟩]⟩ [⟨.result 57419 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57419 .coefficient)
      LeftAuthority57418.bound (LeftAuthority57418.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27011⟩⟩) (rawTerms := some (Proof.Events224.exact57419RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57418.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority57418.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57418.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57700

namespace LeftBound57701
def owner : Owner := ⟨.program ⟨214⟩, ⟨27013⟩⟩
def transferEvent : Nat := 57701
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 57696 .summary) (.transfer 57700) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57696 .summary)
      LeftBound57695.bound (LeftBound57695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25303⟩⟩) (rawTerms := some (Proof.Events225.exact57696RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 57700)
      LeftBound57700.bound (LeftBound57700.actual selector witness) := by
  exact .transfer (LeftBound57700.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57695.bound LeftBound57700.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57695.bound, LeftBound57700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57695.actual selector witness) * (LeftBound57700.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57701

namespace LeftBound57712
def owner : Owner := ⟨.program ⟨214⟩, ⟨20830⟩⟩
def transferEvent : Nat := 57712
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 57710 .coefficient) (.value (.predecessor 1 57711 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57710 .coefficient)
      LeftAuthority57708.bound (LeftAuthority57708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57711 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority57708.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57708.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57708.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound57712

namespace LeftBound57716
def owner : Owner := ⟨.program ⟨214⟩, ⟨20831⟩⟩
def transferEvent : Nat := 57716
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57714 .coefficient) (.predecessor 1 57715 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57714 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57715 .coefficient)
      LeftBound57712.bound (LeftBound57712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound57712.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound57712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound57712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57716

namespace LeftBound57717
def owner : Owner := ⟨.program ⟨214⟩, ⟨20831⟩⟩
def transferEvent : Nat := 57717
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20828⟩⟩]⟩ [⟨.result 57709 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57709 .coefficient)
      LeftAuthority57708.bound (LeftAuthority57708.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20828⟩⟩) (rawTerms := some (Proof.Events225.exact57709RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57708.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority57708.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57708.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57717

namespace LeftBound57718
def owner : Owner := ⟨.program ⟨214⟩, ⟨20831⟩⟩
def transferEvent : Nat := 57718
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 57717) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 57717)
      LeftBound57717.bound (LeftBound57717.actual selector witness) := by
  exact .transfer (LeftBound57717.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound57717.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound57717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound57717.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57718

namespace LeftBound57813
def owner : Owner := ⟨.program ⟨214⟩, ⟨15427⟩⟩
def transferEvent : Nat := 57813
def frameStart : Nat := 57774
def rule : BoundRule := .identity (.predecessor 0 57812 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57812 .coefficient)
      LeftAuthority57810.bound (LeftAuthority57810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57810.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57810.derived selector witness)

def rawBound : CoeffClass := LeftAuthority57810.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority57810.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57813

namespace LeftBound57830
def owner : Owner := ⟨.program ⟨214⟩, ⟨15466⟩⟩
def transferEvent : Nat := 57830
def frameStart : Nat := 57774
def rule : BoundRule := .sum [.predecessor 0 57828 .coefficient, .predecessor 1 57829 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57828 .coefficient)
      LeftBound57813.bound (LeftBound57813.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57829 .coefficient)
      LeftAuthority57826.bound (LeftAuthority57826.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority57826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57813.bound, LeftAuthority57826.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57813.bound, LeftAuthority57826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57813.actual selector witness, LeftAuthority57826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57830

namespace LeftBound57833
def owner : Owner := ⟨.program ⟨214⟩, ⟨15467⟩⟩
def transferEvent : Nat := 57833
def frameStart : Nat := 57774
def rule : BoundRule := .identity (.predecessor 0 57832 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57832 .coefficient)
      LeftBound57830.bound (LeftBound57830.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57830.derived selector witness)

def rawBound : CoeffClass := LeftBound57830.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound57830.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57833

namespace LeftBound57839
def owner : Owner := ⟨.program ⟨214⟩, ⟨15468⟩⟩
def transferEvent : Nat := 57839
def frameStart : Nat := 57774
def rule : BoundRule := .product (.predecessor 0 57837 .coefficient) (.predecessor 1 57838 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57837 .coefficient)
      LeftAuthority57835.bound (LeftAuthority57835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57838 .coefficient)
      LeftBound57833.bound (LeftBound57833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57833.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority57835.bound LeftBound57833.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57835.bound, LeftBound57833.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority57835.actual selector witness) * (LeftBound57833.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57839

namespace LeftBound57847
def owner : Owner := ⟨.program ⟨214⟩, ⟨15469⟩⟩
def transferEvent : Nat := 57847
def frameStart : Nat := 57774
def rule : BoundRule := .sum [.predecessor 0 57845 .coefficient, .predecessor 1 57846 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57845 .coefficient)
      LeftAuthority57843.bound (LeftAuthority57843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57846 .coefficient)
      LeftBound57839.bound (LeftBound57839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57839.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57839.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57843.bound, LeftBound57839.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57843.bound, LeftBound57839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority57843.actual selector witness, LeftBound57839.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57847

namespace LeftBound57851
def owner : Owner := ⟨.program ⟨214⟩, ⟨27012⟩⟩
def transferEvent : Nat := 57851
def frameStart : Nat := 57774
def rule : BoundRule := .product (.predecessor 0 57849 .coefficient) (.predecessor 1 57850 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57849 .coefficient)
      LeftBound57847.bound (LeftBound57847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57850 .coefficient)
      LeftAuthority57824.bound (LeftAuthority57824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57824.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57824.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57847.bound LeftAuthority57824.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57847.bound, LeftAuthority57824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57847.actual selector witness) * (LeftAuthority57824.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57851

namespace LeftBound57862
def owner : Owner := ⟨.program ⟨214⟩, ⟨17343⟩⟩
def transferEvent : Nat := 57862
def frameStart : Nat := 57774
def rule : BoundRule := .product (.predecessor 0 57860 .coefficient) (.predecessor 1 57861 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57860 .coefficient)
      LeftAuthority57835.bound (LeftAuthority57835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57861 .coefficient)
      LeftAuthority57858.bound (LeftAuthority57858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority57835.bound LeftAuthority57858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57835.bound, LeftAuthority57858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority57835.actual selector witness) * (LeftAuthority57858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57862

namespace LeftBound57870
def owner : Owner := ⟨.program ⟨214⟩, ⟨17344⟩⟩
def transferEvent : Nat := 57870
def frameStart : Nat := 57774
def rule : BoundRule := .sum [.predecessor 0 57868 .coefficient, .predecessor 1 57869 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57868 .coefficient)
      LeftAuthority57866.bound (LeftAuthority57866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57869 .coefficient)
      LeftBound57862.bound (LeftBound57862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57866.bound, LeftBound57862.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57866.bound, LeftBound57862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority57866.actual selector witness, LeftBound57862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57870

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
