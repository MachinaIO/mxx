import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard176
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard177

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26971
def owner : Owner := ⟨.program ⟨214⟩, ⟨15835⟩⟩
def transferEvent : Nat := 26971
def frameStart : Nat := 26869
def rule : BoundRule := .product (.predecessor 0 26969 .coefficient) (.predecessor 1 26970 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26969 .coefficient)
      LeftAuthority26924.bound (LeftAuthority26924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26970 .coefficient)
      LeftAuthority26967.bound (LeftAuthority26967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority26924.bound LeftAuthority26967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26924.bound, LeftAuthority26967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority26924.actual selector witness) * (LeftAuthority26967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26971

namespace LeftBound26979
def owner : Owner := ⟨.program ⟨214⟩, ⟨15836⟩⟩
def transferEvent : Nat := 26979
def frameStart : Nat := 26869
def rule : BoundRule := .sum [.predecessor 0 26977 .coefficient, .predecessor 1 26978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26977 .coefficient)
      LeftAuthority26975.bound (LeftAuthority26975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26975.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26978 .coefficient)
      LeftBound26971.bound (LeftBound26971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority26975.bound, LeftBound26971.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26975.bound, LeftBound26971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority26975.actual selector witness, LeftBound26971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26979

namespace LeftBound26983
def owner : Owner := ⟨.program ⟨214⟩, ⟨26008⟩⟩
def transferEvent : Nat := 26983
def frameStart : Nat := 26869
def rule : BoundRule := .sum [.predecessor 0 26981 .coefficient, .predecessor 1 26982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26981 .coefficient)
      LeftBound26979.bound (LeftBound26979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26982 .coefficient)
      LeftBound26960.bound (LeftBound26960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26979.bound, LeftBound26960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26979.bound, LeftBound26960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26979.actual selector witness, LeftBound26960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26983

namespace LeftBound26996
def owner : Owner := ⟨.program ⟨214⟩, ⟨26006⟩⟩
def transferEvent : Nat := 26996
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26994 .coefficient, .predecessor 1 26995 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26994 .coefficient)
      LeftBound26817.bound (LeftBound26817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26995 .coefficient)
      LeftBound26800.bound (LeftBound26800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26817.bound, LeftBound26800.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26817.bound, LeftBound26800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26817.actual selector witness, LeftBound26800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26996

namespace LeftBound26999
def owner : Owner := ⟨.program ⟨214⟩, ⟨26006⟩⟩
def transferEvent : Nat := 26999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26993 .summary, .result 26807 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26993 .summary)
      LeftBound26819.bound (LeftBound26819.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19471⟩⟩) (rawTerms := some (Proof.Events105.exact26993RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26807 .summary)
      LeftBound26802.bound (LeftBound26802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26005⟩⟩) (rawTerms := some (Proof.Events104.exact26807RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26819.bound, LeftBound26802.bound]
def bound : CoeffClass := .finite ⟨352054612209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26819.bound, LeftBound26802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26819.actual selector witness, LeftBound26802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26999

namespace LeftBound27003
def owner : Owner := ⟨.program ⟨214⟩, ⟨27690⟩⟩
def transferEvent : Nat := 27003
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27001 .coefficient) (.predecessor 1 27002 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27001 .coefficient)
      LeftBound26996.bound (LeftBound26996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27002 .coefficient)
      LeftAuthority26722.bound (LeftAuthority26722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26722.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26996.bound LeftAuthority26722.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26996.bound, LeftAuthority26722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26996.actual selector witness) * (LeftAuthority26722.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27003

namespace LeftBound27004
def owner : Owner := ⟨.program ⟨214⟩, ⟨27690⟩⟩
def transferEvent : Nat := 27004
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩ [⟨.result 26723 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26723 .coefficient)
      LeftAuthority26722.bound (LeftAuthority26722.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27688⟩⟩) (rawTerms := some (Proof.Events104.exact26723RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26722.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority26722.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26722.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27004

namespace LeftBound27005
def owner : Owner := ⟨.program ⟨214⟩, ⟨27690⟩⟩
def transferEvent : Nat := 27005
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27000 .summary) (.transfer 27004) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27000 .summary)
      LeftBound26999.bound (LeftBound26999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26006⟩⟩) (rawTerms := some (Proof.Events105.exact27000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27004)
      LeftBound27004.bound (LeftBound27004.actual selector witness) := by
  exact .transfer (LeftBound27004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26999.bound LeftBound27004.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26999.bound, LeftBound27004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26999.actual selector witness) * (LeftBound27004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27005

namespace LeftBound27016
def owner : Owner := ⟨.program ⟨214⟩, ⟨21270⟩⟩
def transferEvent : Nat := 27016
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 27014 .coefficient) (.value (.predecessor 1 27015 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27014 .coefficient)
      LeftAuthority27012.bound (LeftAuthority27012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27015 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority27012.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27012.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27012.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27016

namespace LeftBound27020
def owner : Owner := ⟨.program ⟨214⟩, ⟨21271⟩⟩
def transferEvent : Nat := 27020
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27018 .coefficient) (.predecessor 1 27019 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27018 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27019 .coefficient)
      LeftBound27016.bound (LeftBound27016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound27016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound27016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound27016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27020

namespace LeftBound27021
def owner : Owner := ⟨.program ⟨214⟩, ⟨21271⟩⟩
def transferEvent : Nat := 27021
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩ [⟨.result 27013 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27013 .coefficient)
      LeftAuthority27012.bound (LeftAuthority27012.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21268⟩⟩) (rawTerms := some (Proof.Events105.exact27013RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27012.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27012.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27012.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27021

namespace LeftBound27022
def owner : Owner := ⟨.program ⟨214⟩, ⟨21271⟩⟩
def transferEvent : Nat := 27022
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 27021) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27021)
      LeftBound27021.bound (LeftBound27021.actual selector witness) := by
  exact .transfer (LeftBound27021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound27021.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound27021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound27021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27022

namespace LeftBound27117
def owner : Owner := ⟨.program ⟨214⟩, ⟨15834⟩⟩
def transferEvent : Nat := 27117
def frameStart : Nat := 27078
def rule : BoundRule := .identity (.predecessor 0 27116 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27116 .coefficient)
      LeftAuthority27114.bound (LeftAuthority27114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27114.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27114.derived selector witness)

def rawBound : CoeffClass := LeftAuthority27114.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority27114.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27117

namespace LeftBound27134
def owner : Owner := ⟨.program ⟨214⟩, ⟨15908⟩⟩
def transferEvent : Nat := 27134
def frameStart : Nat := 27078
def rule : BoundRule := .sum [.predecessor 0 27132 .coefficient, .predecessor 1 27133 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27132 .coefficient)
      LeftBound27117.bound (LeftBound27117.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27133 .coefficient)
      LeftAuthority27130.bound (LeftAuthority27130.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority27130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27117.bound, LeftAuthority27130.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27117.bound, LeftAuthority27130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27117.actual selector witness, LeftAuthority27130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27134

namespace LeftBound27137
def owner : Owner := ⟨.program ⟨214⟩, ⟨15909⟩⟩
def transferEvent : Nat := 27137
def frameStart : Nat := 27078
def rule : BoundRule := .identity (.predecessor 0 27136 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27136 .coefficient)
      LeftBound27134.bound (LeftBound27134.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27134.derived selector witness)

def rawBound : CoeffClass := LeftBound27134.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound27134.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27137

namespace LeftBound27143
def owner : Owner := ⟨.program ⟨214⟩, ⟨15910⟩⟩
def transferEvent : Nat := 27143
def frameStart : Nat := 27078
def rule : BoundRule := .product (.predecessor 0 27141 .coefficient) (.predecessor 1 27142 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27141 .coefficient)
      LeftAuthority27139.bound (LeftAuthority27139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27142 .coefficient)
      LeftBound27137.bound (LeftBound27137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27137.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27137.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority27139.bound LeftBound27137.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27139.bound, LeftBound27137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority27139.actual selector witness) * (LeftBound27137.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27143

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
