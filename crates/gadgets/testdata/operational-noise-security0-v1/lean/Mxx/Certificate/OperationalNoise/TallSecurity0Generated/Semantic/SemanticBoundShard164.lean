import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard163

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound25088
def owner : Owner := ⟨.program ⟨214⟩, ⟨21846⟩⟩
def transferEvent : Nat := 25088
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 25086 .coefficient) (.value (.predecessor 1 25087 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25086 .coefficient)
      LeftAuthority25084.bound (LeftAuthority25084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25084.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25087 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority25084.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25084.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25084.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25088

namespace LeftBound25092
def owner : Owner := ⟨.program ⟨214⟩, ⟨21847⟩⟩
def transferEvent : Nat := 25092
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25090 .coefficient) (.predecessor 1 25091 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25090 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25091 .coefficient)
      LeftBound25088.bound (LeftBound25088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25088.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25088.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound25088.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound25088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound25088.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25092

namespace LeftBound25093
def owner : Owner := ⟨.program ⟨214⟩, ⟨21847⟩⟩
def transferEvent : Nat := 25093
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21844⟩⟩]⟩ [⟨.result 25085 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25085 .coefficient)
      LeftAuthority25084.bound (LeftAuthority25084.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21844⟩⟩) (rawTerms := some (Proof.Events097.exact25085RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25084.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25084.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25084.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25084.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25084.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25093

namespace LeftBound25094
def owner : Owner := ⟨.program ⟨214⟩, ⟨21847⟩⟩
def transferEvent : Nat := 25094
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 25093) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25093)
      LeftBound25093.bound (LeftBound25093.actual selector witness) := by
  exact .transfer (LeftBound25093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound25093.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound25093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound25093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25094

namespace LeftBound25189
def owner : Owner := ⟨.program ⟨214⟩, ⟨16275⟩⟩
def transferEvent : Nat := 25189
def frameStart : Nat := 25150
def rule : BoundRule := .identity (.predecessor 0 25188 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25188 .coefficient)
      LeftAuthority25186.bound (LeftAuthority25186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25186.derived selector witness)

def rawBound : CoeffClass := LeftAuthority25186.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority25186.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25189

namespace LeftBound25206
def owner : Owner := ⟨.program ⟨214⟩, ⟨16349⟩⟩
def transferEvent : Nat := 25206
def frameStart : Nat := 25150
def rule : BoundRule := .sum [.predecessor 0 25204 .coefficient, .predecessor 1 25205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25204 .coefficient)
      LeftBound25189.bound (LeftBound25189.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25205 .coefficient)
      LeftAuthority25202.bound (LeftAuthority25202.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25189.bound, LeftAuthority25202.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25189.bound, LeftAuthority25202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25189.actual selector witness, LeftAuthority25202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25206

namespace LeftBound25209
def owner : Owner := ⟨.program ⟨214⟩, ⟨16350⟩⟩
def transferEvent : Nat := 25209
def frameStart : Nat := 25150
def rule : BoundRule := .identity (.predecessor 0 25208 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25208 .coefficient)
      LeftBound25206.bound (LeftBound25206.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25206.derived selector witness)

def rawBound : CoeffClass := LeftBound25206.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound25206.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25209

namespace LeftBound25215
def owner : Owner := ⟨.program ⟨214⟩, ⟨16351⟩⟩
def transferEvent : Nat := 25215
def frameStart : Nat := 25150
def rule : BoundRule := .product (.predecessor 0 25213 .coefficient) (.predecessor 1 25214 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25213 .coefficient)
      LeftAuthority25211.bound (LeftAuthority25211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25214 .coefficient)
      LeftBound25209.bound (LeftBound25209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25209.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority25211.bound LeftBound25209.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25211.bound, LeftBound25209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority25211.actual selector witness) * (LeftBound25209.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25215

namespace LeftBound25223
def owner : Owner := ⟨.program ⟨214⟩, ⟨16352⟩⟩
def transferEvent : Nat := 25223
def frameStart : Nat := 25150
def rule : BoundRule := .sum [.predecessor 0 25221 .coefficient, .predecessor 1 25222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25221 .coefficient)
      LeftAuthority25219.bound (LeftAuthority25219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25222 .coefficient)
      LeftBound25215.bound (LeftBound25215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25215.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority25219.bound, LeftBound25215.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25219.bound, LeftBound25215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority25219.actual selector witness, LeftBound25215.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25223

namespace LeftBound25227
def owner : Owner := ⟨.program ⟨214⟩, ⟨28557⟩⟩
def transferEvent : Nat := 25227
def frameStart : Nat := 25150
def rule : BoundRule := .product (.predecessor 0 25225 .coefficient) (.predecessor 1 25226 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25225 .coefficient)
      LeftBound25223.bound (LeftBound25223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25226 .coefficient)
      LeftAuthority25200.bound (LeftAuthority25200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25223.bound LeftAuthority25200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25223.bound, LeftAuthority25200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25223.actual selector witness) * (LeftAuthority25200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25227

namespace LeftBound25238
def owner : Owner := ⟨.program ⟨214⟩, ⟨16318⟩⟩
def transferEvent : Nat := 25238
def frameStart : Nat := 25150
def rule : BoundRule := .product (.predecessor 0 25236 .coefficient) (.predecessor 1 25237 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25236 .coefficient)
      LeftAuthority25211.bound (LeftAuthority25211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25237 .coefficient)
      LeftAuthority25234.bound (LeftAuthority25234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25234.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25234.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority25211.bound LeftAuthority25234.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25211.bound, LeftAuthority25234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority25211.actual selector witness) * (LeftAuthority25234.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25238

namespace LeftBound25246
def owner : Owner := ⟨.program ⟨214⟩, ⟨16319⟩⟩
def transferEvent : Nat := 25246
def frameStart : Nat := 25150
def rule : BoundRule := .sum [.predecessor 0 25244 .coefficient, .predecessor 1 25245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25244 .coefficient)
      LeftAuthority25242.bound (LeftAuthority25242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25242.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25245 .coefficient)
      LeftBound25238.bound (LeftBound25238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25238.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority25242.bound, LeftBound25238.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25242.bound, LeftBound25238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority25242.actual selector witness, LeftBound25238.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25246

namespace LeftBound25250
def owner : Owner := ⟨.program ⟨214⟩, ⟨28561⟩⟩
def transferEvent : Nat := 25250
def frameStart : Nat := 25150
def rule : BoundRule := .sum [.predecessor 0 25248 .coefficient, .predecessor 1 25249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25248 .coefficient)
      LeftBound25246.bound (LeftBound25246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25249 .coefficient)
      LeftBound25227.bound (LeftBound25227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25227.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25246.bound, LeftBound25227.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25246.bound, LeftBound25227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25246.actual selector witness, LeftBound25227.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25250

namespace LeftBound25263
def owner : Owner := ⟨.program ⟨214⟩, ⟨28559⟩⟩
def transferEvent : Nat := 25263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25261 .coefficient, .predecessor 1 25262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25261 .coefficient)
      LeftBound25092.bound (LeftBound25092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25092.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25262 .coefficient)
      LeftBound25075.bound (LeftBound25075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25075.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25092.bound, LeftBound25075.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25092.bound, LeftBound25075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25092.actual selector witness, LeftBound25075.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25263

namespace LeftBound25266
def owner : Owner := ⟨.program ⟨214⟩, ⟨28559⟩⟩
def transferEvent : Nat := 25266
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 25260 .summary, .result 25082 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25260 .summary)
      LeftBound25094.bound (LeftBound25094.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21847⟩⟩) (rawTerms := some (Proof.Events098.exact25260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25082 .summary)
      LeftBound25077.bound (LeftBound25077.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28558⟩⟩) (rawTerms := some (Proof.Events097.exact25082RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25094.bound, LeftBound25077.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25094.bound, LeftBound25077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25094.actual selector witness, LeftBound25077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25266

namespace LeftBound25290
def owner : Owner := ⟨.program ⟨214⟩, ⟨11650⟩⟩
def transferEvent : Nat := 25290
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 25288 .coefficient) (.predecessor 1 25289 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25288 .coefficient)
      LeftAuthority1025.bound (LeftAuthority1025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1025.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1025.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25289 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1025.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1025.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1025.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25290

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
