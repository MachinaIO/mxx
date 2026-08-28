import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard071

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound11948
def owner : Owner := ⟨.program ⟨214⟩, ⟨27921⟩⟩
def transferEvent : Nat := 11948
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11946 .coefficient, .predecessor 1 11947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11946 .coefficient)
      LeftBound11777.bound (LeftBound11777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11947 .coefficient)
      LeftBound11760.bound (LeftBound11760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11760.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11777.bound, LeftBound11760.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11777.bound, LeftBound11760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11777.actual selector witness, LeftBound11760.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11948

namespace LeftBound11951
def owner : Owner := ⟨.program ⟨214⟩, ⟨27921⟩⟩
def transferEvent : Nat := 11951
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 11945 .summary, .result 11767 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11945 .summary)
      LeftBound11779.bound (LeftBound11779.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21419⟩⟩) (rawTerms := some (Proof.Events046.exact11945RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11767 .summary)
      LeftBound11762.bound (LeftBound11762.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27920⟩⟩) (rawTerms := some (Proof.Events045.exact11767RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11762.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11779.bound, LeftBound11762.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11779.bound, LeftBound11762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11779.actual selector witness, LeftBound11762.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11951

namespace LeftBound11974
def owner : Owner := ⟨.program ⟨214⟩, ⟨92⟩⟩
def transferEvent : Nat := 11974
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 11973 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11973 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11974

namespace LeftBound11978
def owner : Owner := ⟨.program ⟨214⟩, ⟨11402⟩⟩
def transferEvent : Nat := 11978
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 11976 .coefficient) (.predecessor 1 11977 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11976 .coefficient)
      LeftAuthority303.bound (LeftAuthority303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11977 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority303.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority303.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority303.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound11978

namespace LeftBound11982
def owner : Owner := ⟨.program ⟨214⟩, ⟨6778⟩⟩
def transferEvent : Nat := 11982
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 11981 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11981 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11982

namespace LeftBound11986
def owner : Owner := ⟨.program ⟨214⟩, ⟨7386⟩⟩
def transferEvent : Nat := 11986
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11984 .coefficient) (.predecessor 1 11985 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11984 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11985 .coefficient)
      LeftBound11982.bound (LeftBound11982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11982.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound11982.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound11982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound11982.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11986

namespace LeftBound11991
def owner : Owner := ⟨.program ⟨214⟩, ⟨11403⟩⟩
def transferEvent : Nat := 11991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11989 .coefficient, .predecessor 1 11990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11989 .coefficient)
      LeftBound11986.bound (LeftBound11986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11990 .coefficient)
      LeftBound11978.bound (LeftBound11978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11986.bound, LeftBound11978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11986.bound, LeftBound11978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11986.actual selector witness, LeftBound11978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11991

namespace LeftBound11995
def owner : Owner := ⟨.program ⟨214⟩, ⟨11404⟩⟩
def transferEvent : Nat := 11995
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11993 .coefficient, .predecessor 1 11994 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11993 .coefficient)
      LeftBound11991.bound (LeftBound11991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11994 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11991.bound, LeftBound11974.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11991.bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11991.actual selector witness, LeftBound11974.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11995

namespace LeftBound11996
def owner : Owner := ⟨.program ⟨214⟩, ⟨11404⟩⟩
def transferEvent : Nat := 11996
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩ [⟨.result 11975 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11975 .coefficient)
      LeftBound11974.bound (LeftBound11974.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨92⟩⟩) (rawTerms := some (Proof.Events046.exact11975RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11974.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11974.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound11996

namespace LeftBound12001
def owner : Owner := ⟨.program ⟨214⟩, ⟨14029⟩⟩
def transferEvent : Nat := 12001
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11999 .coefficient) (.predecessor 1 12000 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11999 .coefficient)
      LeftBound11995.bound (LeftBound11995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12000 .coefficient)
      LeftAuthority306.bound (LeftAuthority306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound11995.bound LeftAuthority306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11995.bound, LeftAuthority306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound11995.actual selector witness) * (LeftAuthority306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12001

namespace LeftBound12002
def owner : Owner := ⟨.program ⟨214⟩, ⟨14029⟩⟩
def transferEvent : Nat := 12002
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩ [⟨.result 307 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 307 .coefficient)
      LeftAuthority306.bound (LeftAuthority306.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14026⟩⟩) (rawTerms := some (Proof.Events001.exact307RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority306.bound []
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority306.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12002

namespace LeftBound12003
def owner : Owner := ⟨.program ⟨214⟩, ⟨14029⟩⟩
def transferEvent : Nat := 12003
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 11998 .summary) (.transfer 12002) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11998 .summary)
      LeftBound11996.bound (LeftBound11996.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11404⟩⟩) (rawTerms := some (Proof.Events046.exact11998RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 12002)
      LeftBound12002.bound (LeftBound12002.actual selector witness) := by
  exact .transfer (LeftBound12002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound11996.bound LeftBound12002.bound
def bound : CoeffClass := .finite ⟨13312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11996.bound, LeftBound12002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound11996.actual selector witness) * (LeftBound12002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12003

namespace LeftBound12012
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def transferEvent : Nat := 12012
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 12010 .coefficient) (.value (.predecessor 1 12011 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12010 .coefficient)
      LeftAuthority12008.bound (LeftAuthority12008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12008.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12011 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority12008.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12008.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12008.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound12012

namespace LeftBound12015
def owner : Owner := ⟨.program ⟨214⟩, ⟨72⟩⟩
def transferEvent : Nat := 12015
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 12014 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12014 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12015

namespace LeftBound12019
def owner : Owner := ⟨.program ⟨214⟩, ⟨14030⟩⟩
def transferEvent : Nat := 12019
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 12017 .coefficient) (.predecessor 1 12018 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12017 .coefficient)
      LeftAuthority306.bound (LeftAuthority306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12018 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority306.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority306.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority306.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound12019

namespace LeftBound12023
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def transferEvent : Nat := 12023
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 12022 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12022 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12023

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
