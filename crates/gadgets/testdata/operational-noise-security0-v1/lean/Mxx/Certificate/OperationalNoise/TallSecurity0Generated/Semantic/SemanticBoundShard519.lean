import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard464
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard518

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound76875
def owner : Owner := ⟨.program ⟨214⟩, ⟨28716⟩⟩
def transferEvent : Nat := 76875
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩ [⟨.result 76871 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76871 .coefficient)
      LeftAuthority76870.bound (LeftAuthority76870.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28714⟩⟩) (rawTerms := some (Proof.Events300.exact76871RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76870.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76870.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76870.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76875

namespace LeftBound76876
def owner : Owner := ⟨.program ⟨214⟩, ⟨28716⟩⟩
def transferEvent : Nat := 76876
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68465 .summary) (.transfer 76875) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68465 .summary)
      LeftBound68464.bound (LeftBound68464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25216⟩⟩) (rawTerms := some (Proof.Events267.exact68465RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76875)
      LeftBound76875.bound (LeftBound76875.actual selector witness) := by
  exact .transfer (LeftBound76875.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68464.bound LeftBound76875.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68464.bound, LeftBound76875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68464.actual selector witness) * (LeftBound76875.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76876

namespace LeftBound76887
def owner : Owner := ⟨.program ⟨214⟩, ⟨21902⟩⟩
def transferEvent : Nat := 76887
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 76885 .coefficient) (.value (.predecessor 1 76886 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76885 .coefficient)
      LeftAuthority76883.bound (LeftAuthority76883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76886 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority76883.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76883.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76883.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound76887

namespace LeftBound76891
def owner : Owner := ⟨.program ⟨214⟩, ⟨21903⟩⟩
def transferEvent : Nat := 76891
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76889 .coefficient) (.predecessor 1 76890 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76889 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76890 .coefficient)
      LeftBound76887.bound (LeftBound76887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76887.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound76887.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound76887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound76887.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76891

namespace LeftBound76892
def owner : Owner := ⟨.program ⟨214⟩, ⟨21903⟩⟩
def transferEvent : Nat := 76892
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩ [⟨.result 76884 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76884 .coefficient)
      LeftAuthority76883.bound (LeftAuthority76883.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21900⟩⟩) (rawTerms := some (Proof.Events300.exact76884RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76883.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76883.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76883.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76892

namespace LeftBound76893
def owner : Owner := ⟨.program ⟨214⟩, ⟨21903⟩⟩
def transferEvent : Nat := 76893
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 76892) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76892)
      LeftBound76892.bound (LeftBound76892.actual selector witness) := by
  exact .transfer (LeftBound76892.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound76892.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound76892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound76892.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76893

namespace LeftBound76988
def owner : Owner := ⟨.program ⟨214⟩, ⟨16378⟩⟩
def transferEvent : Nat := 76988
def frameStart : Nat := 76949
def rule : BoundRule := .identity (.predecessor 0 76987 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76987 .coefficient)
      LeftAuthority76985.bound (LeftAuthority76985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76985.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76985.derived selector witness)

def rawBound : CoeffClass := LeftAuthority76985.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority76985.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76988

namespace LeftBound77005
def owner : Owner := ⟨.program ⟨214⟩, ⟨16417⟩⟩
def transferEvent : Nat := 77005
def frameStart : Nat := 76949
def rule : BoundRule := .sum [.predecessor 0 77003 .coefficient, .predecessor 1 77004 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77003 .coefficient)
      LeftBound76988.bound (LeftBound76988.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76988.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77004 .coefficient)
      LeftAuthority77001.bound (LeftAuthority77001.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority77001.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76988.bound, LeftAuthority77001.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76988.bound, LeftAuthority77001.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76988.actual selector witness, LeftAuthority77001.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77005

namespace LeftBound77008
def owner : Owner := ⟨.program ⟨214⟩, ⟨16418⟩⟩
def transferEvent : Nat := 77008
def frameStart : Nat := 76949
def rule : BoundRule := .identity (.predecessor 0 77007 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77007 .coefficient)
      LeftBound77005.bound (LeftBound77005.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77005.derived selector witness)

def rawBound : CoeffClass := LeftBound77005.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound77005.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77008

namespace LeftBound77014
def owner : Owner := ⟨.program ⟨214⟩, ⟨16419⟩⟩
def transferEvent : Nat := 77014
def frameStart : Nat := 76949
def rule : BoundRule := .product (.predecessor 0 77012 .coefficient) (.predecessor 1 77013 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77012 .coefficient)
      LeftAuthority77010.bound (LeftAuthority77010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77013 .coefficient)
      LeftBound77008.bound (LeftBound77008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority77010.bound LeftBound77008.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77010.bound, LeftBound77008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority77010.actual selector witness) * (LeftBound77008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77014

namespace LeftBound77022
def owner : Owner := ⟨.program ⟨214⟩, ⟨16420⟩⟩
def transferEvent : Nat := 77022
def frameStart : Nat := 76949
def rule : BoundRule := .sum [.predecessor 0 77020 .coefficient, .predecessor 1 77021 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77020 .coefficient)
      LeftAuthority77018.bound (LeftAuthority77018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77018.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77021 .coefficient)
      LeftBound77014.bound (LeftBound77014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77018.bound, LeftBound77014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77018.bound, LeftBound77014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77018.actual selector witness, LeftBound77014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77022

namespace LeftBound77026
def owner : Owner := ⟨.program ⟨214⟩, ⟨28715⟩⟩
def transferEvent : Nat := 77026
def frameStart : Nat := 76949
def rule : BoundRule := .product (.predecessor 0 77024 .coefficient) (.predecessor 1 77025 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77024 .coefficient)
      LeftBound77022.bound (LeftBound77022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77025 .coefficient)
      LeftAuthority76999.bound (LeftAuthority76999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76999.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76999.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77022.bound LeftAuthority76999.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77022.bound, LeftAuthority76999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77022.actual selector witness) * (LeftAuthority76999.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77026

namespace LeftBound77037
def owner : Owner := ⟨.program ⟨214⟩, ⟨18827⟩⟩
def transferEvent : Nat := 77037
def frameStart : Nat := 76949
def rule : BoundRule := .product (.predecessor 0 77035 .coefficient) (.predecessor 1 77036 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77035 .coefficient)
      LeftAuthority77010.bound (LeftAuthority77010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77036 .coefficient)
      LeftAuthority77033.bound (LeftAuthority77033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77033.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority77010.bound LeftAuthority77033.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77010.bound, LeftAuthority77033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority77010.actual selector witness) * (LeftAuthority77033.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77037

namespace LeftBound77045
def owner : Owner := ⟨.program ⟨214⟩, ⟨18831⟩⟩
def transferEvent : Nat := 77045
def frameStart : Nat := 76949
def rule : BoundRule := .sum [.predecessor 0 77043 .coefficient, .predecessor 1 77044 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77043 .coefficient)
      LeftAuthority77041.bound (LeftAuthority77041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77041.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77044 .coefficient)
      LeftBound77037.bound (LeftBound77037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77037.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77037.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77041.bound, LeftBound77037.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77041.bound, LeftBound77037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77041.actual selector witness, LeftBound77037.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77045

namespace LeftBound77049
def owner : Owner := ⟨.program ⟨214⟩, ⟨28720⟩⟩
def transferEvent : Nat := 77049
def frameStart : Nat := 76949
def rule : BoundRule := .sum [.predecessor 0 77047 .coefficient, .predecessor 1 77048 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77047 .coefficient)
      LeftBound77045.bound (LeftBound77045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77048 .coefficient)
      LeftBound77026.bound (LeftBound77026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact77031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77045.bound, LeftBound77026.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77045.bound, LeftBound77026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77045.actual selector witness, LeftBound77026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77049

namespace LeftBound77062
def owner : Owner := ⟨.program ⟨214⟩, ⟨28717⟩⟩
def transferEvent : Nat := 77062
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77060 .coefficient, .predecessor 1 77061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77060 .coefficient)
      LeftBound76891.bound (LeftBound76891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77061 .coefficient)
      LeftBound76874.bound (LeftBound76874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76891.bound, LeftBound76874.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76891.bound, LeftBound76874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76891.actual selector witness, LeftBound76874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77062

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
