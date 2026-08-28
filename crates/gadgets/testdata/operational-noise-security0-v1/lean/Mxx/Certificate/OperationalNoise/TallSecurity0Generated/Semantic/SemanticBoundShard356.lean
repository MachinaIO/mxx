import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard355

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound52892
def owner : Owner := ⟨.program ⟨214⟩, ⟨22270⟩⟩
def transferEvent : Nat := 52892
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 52890 .coefficient) (.value (.predecessor 1 52891 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52890 .coefficient)
      LeftAuthority52888.bound (LeftAuthority52888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52891 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority52888.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52888.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52888.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52892

namespace LeftBound52896
def owner : Owner := ⟨.program ⟨214⟩, ⟨22271⟩⟩
def transferEvent : Nat := 52896
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52894 .coefficient) (.predecessor 1 52895 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52894 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52895 .coefficient)
      LeftBound52892.bound (LeftBound52892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52892.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound52892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound52892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound52892.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52896

namespace LeftBound52897
def owner : Owner := ⟨.program ⟨214⟩, ⟨22271⟩⟩
def transferEvent : Nat := 52897
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩ [⟨.result 52889 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52889 .coefficient)
      LeftAuthority52888.bound (LeftAuthority52888.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22268⟩⟩) (rawTerms := some (Proof.Events206.exact52889RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52888.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority52888.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52888.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound52897

namespace LeftBound52898
def owner : Owner := ⟨.program ⟨214⟩, ⟨22271⟩⟩
def transferEvent : Nat := 52898
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 52897) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 52897)
      LeftBound52897.bound (LeftBound52897.actual selector witness) := by
  exact .transfer (LeftBound52897.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound52897.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound52897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound52897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52898

namespace LeftBound52993
def owner : Owner := ⟨.program ⟨214⟩, ⟨16554⟩⟩
def transferEvent : Nat := 52993
def frameStart : Nat := 52954
def rule : BoundRule := .identity (.predecessor 0 52992 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52992 .coefficient)
      LeftAuthority52990.bound (LeftAuthority52990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52990.derived selector witness)

def rawBound : CoeffClass := LeftAuthority52990.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority52990.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52993

namespace LeftBound53010
def owner : Owner := ⟨.program ⟨214⟩, ⟨16593⟩⟩
def transferEvent : Nat := 53010
def frameStart : Nat := 52954
def rule : BoundRule := .sum [.predecessor 0 53008 .coefficient, .predecessor 1 53009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53008 .coefficient)
      LeftBound52993.bound (LeftBound52993.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53009 .coefficient)
      LeftAuthority53006.bound (LeftAuthority53006.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52993.bound, LeftAuthority53006.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52993.bound, LeftAuthority53006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52993.actual selector witness, LeftAuthority53006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53010

namespace LeftBound53013
def owner : Owner := ⟨.program ⟨214⟩, ⟨16594⟩⟩
def transferEvent : Nat := 53013
def frameStart : Nat := 52954
def rule : BoundRule := .identity (.predecessor 0 53012 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53012 .coefficient)
      LeftBound53010.bound (LeftBound53010.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53010.derived selector witness)

def rawBound : CoeffClass := LeftBound53010.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound53010.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53013

namespace LeftBound53019
def owner : Owner := ⟨.program ⟨214⟩, ⟨16595⟩⟩
def transferEvent : Nat := 53019
def frameStart : Nat := 52954
def rule : BoundRule := .product (.predecessor 0 53017 .coefficient) (.predecessor 1 53018 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53017 .coefficient)
      LeftAuthority53015.bound (LeftAuthority53015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53018 .coefficient)
      LeftBound53013.bound (LeftBound53013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority53015.bound LeftBound53013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53015.bound, LeftBound53013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority53015.actual selector witness) * (LeftBound53013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53019

namespace LeftBound53027
def owner : Owner := ⟨.program ⟨214⟩, ⟨16596⟩⟩
def transferEvent : Nat := 53027
def frameStart : Nat := 52954
def rule : BoundRule := .sum [.predecessor 0 53025 .coefficient, .predecessor 1 53026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53025 .coefficient)
      LeftAuthority53023.bound (LeftAuthority53023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53026 .coefficient)
      LeftBound53019.bound (LeftBound53019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority53023.bound, LeftBound53019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53023.bound, LeftBound53019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority53023.actual selector witness, LeftBound53019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53027

namespace LeftBound53031
def owner : Owner := ⟨.program ⟨214⟩, ⟨29182⟩⟩
def transferEvent : Nat := 53031
def frameStart : Nat := 52954
def rule : BoundRule := .product (.predecessor 0 53029 .coefficient) (.predecessor 1 53030 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53029 .coefficient)
      LeftBound53027.bound (LeftBound53027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53030 .coefficient)
      LeftAuthority53004.bound (LeftAuthority53004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53027.bound LeftAuthority53004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53027.bound, LeftAuthority53004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53027.actual selector witness) * (LeftAuthority53004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53031

namespace LeftBound53042
def owner : Owner := ⟨.program ⟨214⟩, ⟨18209⟩⟩
def transferEvent : Nat := 53042
def frameStart : Nat := 52954
def rule : BoundRule := .product (.predecessor 0 53040 .coefficient) (.predecessor 1 53041 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53040 .coefficient)
      LeftAuthority53015.bound (LeftAuthority53015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53041 .coefficient)
      LeftAuthority53038.bound (LeftAuthority53038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53038.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53015.bound LeftAuthority53038.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53015.bound, LeftAuthority53038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority53015.actual selector witness) * (LeftAuthority53038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53042

namespace LeftBound53050
def owner : Owner := ⟨.program ⟨214⟩, ⟨18210⟩⟩
def transferEvent : Nat := 53050
def frameStart : Nat := 52954
def rule : BoundRule := .sum [.predecessor 0 53048 .coefficient, .predecessor 1 53049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53048 .coefficient)
      LeftAuthority53046.bound (LeftAuthority53046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53046.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53049 .coefficient)
      LeftBound53042.bound (LeftBound53042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority53046.bound, LeftBound53042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53046.bound, LeftBound53042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority53046.actual selector witness, LeftBound53042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53050

namespace LeftBound53054
def owner : Owner := ⟨.program ⟨214⟩, ⟨29186⟩⟩
def transferEvent : Nat := 53054
def frameStart : Nat := 52954
def rule : BoundRule := .sum [.predecessor 0 53052 .coefficient, .predecessor 1 53053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53052 .coefficient)
      LeftBound53050.bound (LeftBound53050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53053 .coefficient)
      LeftBound53031.bound (LeftBound53031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53050.bound, LeftBound53031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53050.bound, LeftBound53031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53050.actual selector witness, LeftBound53031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53054

namespace LeftBound53067
def owner : Owner := ⟨.program ⟨214⟩, ⟨29184⟩⟩
def transferEvent : Nat := 53067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53065 .coefficient, .predecessor 1 53066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53065 .coefficient)
      LeftBound52896.bound (LeftBound52896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53066 .coefficient)
      LeftBound52879.bound (LeftBound52879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52896.bound, LeftBound52879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52896.bound, LeftBound52879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52896.actual selector witness, LeftBound52879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53067

namespace LeftBound53070
def owner : Owner := ⟨.program ⟨214⟩, ⟨29184⟩⟩
def transferEvent : Nat := 53070
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 53064 .summary, .result 52886 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53064 .summary)
      LeftBound52898.bound (LeftBound52898.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22271⟩⟩) (rawTerms := some (Proof.Events207.exact53064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52886 .summary)
      LeftBound52881.bound (LeftBound52881.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29183⟩⟩) (rawTerms := some (Proof.Events206.exact52886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52898.bound, LeftBound52881.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52898.bound, LeftBound52881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52898.actual selector witness, LeftBound52881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53070

namespace LeftBound53094
def owner : Owner := ⟨.program ⟨214⟩, ⟨12381⟩⟩
def transferEvent : Nat := 53094
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 53092 .coefficient) (.predecessor 1 53093 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53092 .coefficient)
      LeftAuthority2452.bound (LeftAuthority2452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2452.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53093 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2452.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2452.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2452.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53094

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
