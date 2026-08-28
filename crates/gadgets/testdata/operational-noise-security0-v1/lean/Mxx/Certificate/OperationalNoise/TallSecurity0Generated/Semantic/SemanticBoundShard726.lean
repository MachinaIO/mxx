import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard686
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard725

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound105967
def owner : Owner := ⟨.program ⟨214⟩, ⟨15889⟩⟩
def transferEvent : Nat := 105967
def frameStart : Nat := 105920
def rule : BoundRule := .identity (.predecessor 0 105966 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105966 .coefficient)
      LeftBound105964.bound (LeftBound105964.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105964.derived selector witness)

def rawBound : CoeffClass := LeftBound105964.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound105964.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105967

namespace LeftBound105973
def owner : Owner := ⟨.program ⟨214⟩, ⟨15890⟩⟩
def transferEvent : Nat := 105973
def frameStart : Nat := 105920
def rule : BoundRule := .product (.predecessor 0 105971 .coefficient) (.predecessor 1 105972 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105971 .coefficient)
      LeftAuthority105969.bound (LeftAuthority105969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105972 .coefficient)
      LeftBound105967.bound (LeftBound105967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority105969.bound LeftBound105967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105969.bound, LeftBound105967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority105969.actual selector witness) * (LeftBound105967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105973

namespace LeftBound105981
def owner : Owner := ⟨.program ⟨214⟩, ⟨15891⟩⟩
def transferEvent : Nat := 105981
def frameStart : Nat := 105920
def rule : BoundRule := .sum [.predecessor 0 105979 .coefficient, .predecessor 1 105980 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105979 .coefficient)
      LeftAuthority105977.bound (LeftAuthority105977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105980 .coefficient)
      LeftBound105973.bound (LeftBound105973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105973.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105977.bound, LeftBound105973.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105977.bound, LeftBound105973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105977.actual selector witness, LeftBound105973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105981

namespace LeftBound105985
def owner : Owner := ⟨.program ⟨214⟩, ⟨27608⟩⟩
def transferEvent : Nat := 105985
def frameStart : Nat := 105920
def rule : BoundRule := .product (.predecessor 0 105983 .coefficient) (.predecessor 1 105984 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105983 .coefficient)
      LeftBound105981.bound (LeftBound105981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105984 .coefficient)
      LeftAuthority105958.bound (LeftAuthority105958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105958.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105958.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105981.bound LeftAuthority105958.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105981.bound, LeftAuthority105958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105981.actual selector witness) * (LeftAuthority105958.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105985

namespace LeftBound105996
def owner : Owner := ⟨.program ⟨214⟩, ⟨17213⟩⟩
def transferEvent : Nat := 105996
def frameStart : Nat := 105920
def rule : BoundRule := .product (.predecessor 0 105994 .coefficient) (.predecessor 1 105995 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105994 .coefficient)
      LeftAuthority105969.bound (LeftAuthority105969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105995 .coefficient)
      LeftAuthority105992.bound (LeftAuthority105992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact105993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105992.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105992.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority105969.bound LeftAuthority105992.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105969.bound, LeftAuthority105992.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority105969.actual selector witness) * (LeftAuthority105992.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105996

namespace LeftBound106004
def owner : Owner := ⟨.program ⟨214⟩, ⟨17214⟩⟩
def transferEvent : Nat := 106004
def frameStart : Nat := 105920
def rule : BoundRule := .sum [.predecessor 0 106002 .coefficient, .predecessor 1 106003 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106002 .coefficient)
      LeftAuthority106000.bound (LeftAuthority106000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106000.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106000.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106003 .coefficient)
      LeftBound105996.bound (LeftBound105996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact105998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106000.bound, LeftBound105996.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106000.bound, LeftBound105996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106000.actual selector witness, LeftBound105996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106004

namespace LeftBound106008
def owner : Owner := ⟨.program ⟨214⟩, ⟨27613⟩⟩
def transferEvent : Nat := 106008
def frameStart : Nat := 105920
def rule : BoundRule := .sum [.predecessor 0 106006 .coefficient, .predecessor 1 106007 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106006 .coefficient)
      LeftBound106004.bound (LeftBound106004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106007 .coefficient)
      LeftBound105985.bound (LeftBound105985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact105990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105985.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105985.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106004.bound, LeftBound105985.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106004.bound, LeftBound105985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106004.actual selector witness, LeftBound105985.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106008

namespace LeftBound106021
def owner : Owner := ⟨.program ⟨214⟩, ⟨27610⟩⟩
def transferEvent : Nat := 106021
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 106019 .coefficient, .predecessor 1 106020 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106019 .coefficient)
      LeftBound105874.bound (LeftBound105874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106020 .coefficient)
      LeftBound105857.bound (LeftBound105857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105857.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105874.bound, LeftBound105857.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105874.bound, LeftBound105857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105874.actual selector witness, LeftBound105857.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106021

namespace LeftBound106024
def owner : Owner := ⟨.program ⟨214⟩, ⟨27610⟩⟩
def transferEvent : Nat := 106024
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 106018 .summary, .result 105864 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106018 .summary)
      LeftBound105876.bound (LeftBound105876.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21176⟩⟩) (rawTerms := some (Proof.Events414.exact106018RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105864 .summary)
      LeftBound105859.bound (LeftBound105859.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27609⟩⟩) (rawTerms := some (Proof.Events413.exact105864RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105859.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105876.bound, LeftBound105859.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105876.bound, LeftBound105859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105876.actual selector witness, LeftBound105859.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106024

namespace LeftBound106028
def owner : Owner := ⟨.program ⟨214⟩, ⟨27611⟩⟩
def transferEvent : Nat := 106028
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106026 .coefficient) (.predecessor 1 106027 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106026 .coefficient)
      LeftBound106021.bound (LeftBound106021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106021.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106027 .coefficient)
      LeftBound5738.bound (LeftBound5738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106021.bound LeftBound5738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106021.bound, LeftBound5738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106021.actual selector witness) * (LeftBound5738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106028

namespace LeftBound106029
def owner : Owner := ⟨.program ⟨214⟩, ⟨27611⟩⟩
def transferEvent : Nat := 106029
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩ [⟨.result 5735 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5735 .coefficient)
      LeftAuthority5734.bound (LeftAuthority5734.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6643⟩⟩) (rawTerms := some (Proof.Events022.exact5735RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5734.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5734.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5734.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106029

namespace LeftBound106030
def owner : Owner := ⟨.program ⟨214⟩, ⟨27611⟩⟩
def transferEvent : Nat := 106030
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 106025 .summary) (.transfer 106029) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106025 .summary)
      LeftBound106024.bound (LeftBound106024.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27610⟩⟩) (rawTerms := some (Proof.Events414.exact106025RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106029)
      LeftBound106029.bound (LeftBound106029.actual selector witness) := by
  exact .transfer (LeftBound106029.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106024.bound LeftBound106029.bound
def bound : CoeffClass := .finite ⟨4741829718422040195880714240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106024.bound, LeftBound106029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106024.actual selector witness) * (LeftBound106029.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106030

namespace LeftBound106045
def owner : Owner := ⟨.program ⟨214⟩, ⟨27392⟩⟩
def transferEvent : Nat := 106045
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106043 .coefficient) (.predecessor 1 106044 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106043 .coefficient)
      LeftBound99828.bound (LeftBound99828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106044 .coefficient)
      LeftAuthority106041.bound (LeftAuthority106041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106041.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106041.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99828.bound LeftAuthority106041.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99828.bound, LeftAuthority106041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99828.actual selector witness) * (LeftAuthority106041.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106045

namespace LeftBound106046
def owner : Owner := ⟨.program ⟨214⟩, ⟨27392⟩⟩
def transferEvent : Nat := 106046
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩ [⟨.result 106042 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106042 .coefficient)
      LeftAuthority106041.bound (LeftAuthority106041.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27390⟩⟩) (rawTerms := some (Proof.Events414.exact106042RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106041.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106041.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106041.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106041.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106046

namespace LeftBound106047
def owner : Owner := ⟨.program ⟨214⟩, ⟨27392⟩⟩
def transferEvent : Nat := 106047
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99832 .summary) (.transfer 106046) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99832 .summary)
      LeftBound99831.bound (LeftBound99831.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25901⟩⟩) (rawTerms := some (Proof.Events389.exact99832RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106046)
      LeftBound106046.bound (LeftBound106046.actual selector witness) := by
  exact .transfer (LeftBound106046.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99831.bound LeftBound106046.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99831.bound, LeftBound106046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99831.actual selector witness) * (LeftBound106046.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106047

namespace LeftBound106058
def owner : Owner := ⟨.program ⟨214⟩, ⟨21031⟩⟩
def transferEvent : Nat := 106058
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 106056 .coefficient) (.value (.predecessor 1 106057 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106056 .coefficient)
      LeftAuthority106054.bound (LeftAuthority106054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106054.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106057 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority106054.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106054.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106054.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound106058

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
