import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard290

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound43093
def owner : Owner := ⟨.program ⟨214⟩, ⟨20835⟩⟩
def transferEvent : Nat := 43093
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 43092) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43092)
      LeftBound43092.bound (LeftBound43092.actual selector witness) := by
  exact .transfer (LeftBound43092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound43092.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound43092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound43092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43093

namespace LeftBound43188
def owner : Owner := ⟨.program ⟨214⟩, ⟨15431⟩⟩
def transferEvent : Nat := 43188
def frameStart : Nat := 43149
def rule : BoundRule := .identity (.predecessor 0 43187 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43187 .coefficient)
      LeftAuthority43185.bound (LeftAuthority43185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43185.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43185.derived selector witness)

def rawBound : CoeffClass := LeftAuthority43185.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority43185.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43188

namespace LeftBound43205
def owner : Owner := ⟨.program ⟨214⟩, ⟨15470⟩⟩
def transferEvent : Nat := 43205
def frameStart : Nat := 43149
def rule : BoundRule := .sum [.predecessor 0 43203 .coefficient, .predecessor 1 43204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43203 .coefficient)
      LeftBound43188.bound (LeftBound43188.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43204 .coefficient)
      LeftAuthority43201.bound (LeftAuthority43201.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority43201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43188.bound, LeftAuthority43201.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43188.bound, LeftAuthority43201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43188.actual selector witness, LeftAuthority43201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43205

namespace LeftBound43208
def owner : Owner := ⟨.program ⟨214⟩, ⟨15471⟩⟩
def transferEvent : Nat := 43208
def frameStart : Nat := 43149
def rule : BoundRule := .identity (.predecessor 0 43207 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43207 .coefficient)
      LeftBound43205.bound (LeftBound43205.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43205.derived selector witness)

def rawBound : CoeffClass := LeftBound43205.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43205.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound43205.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43208

namespace LeftBound43214
def owner : Owner := ⟨.program ⟨214⟩, ⟨15472⟩⟩
def transferEvent : Nat := 43214
def frameStart : Nat := 43149
def rule : BoundRule := .product (.predecessor 0 43212 .coefficient) (.predecessor 1 43213 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43212 .coefficient)
      LeftAuthority43210.bound (LeftAuthority43210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43213 .coefficient)
      LeftBound43208.bound (LeftBound43208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43208.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority43210.bound LeftBound43208.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43210.bound, LeftBound43208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority43210.actual selector witness) * (LeftBound43208.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43214

namespace LeftBound43222
def owner : Owner := ⟨.program ⟨214⟩, ⟨15473⟩⟩
def transferEvent : Nat := 43222
def frameStart : Nat := 43149
def rule : BoundRule := .sum [.predecessor 0 43220 .coefficient, .predecessor 1 43221 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43220 .coefficient)
      LeftAuthority43218.bound (LeftAuthority43218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43218.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43221 .coefficient)
      LeftBound43214.bound (LeftBound43214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43214.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority43218.bound, LeftBound43214.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43218.bound, LeftBound43214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority43218.actual selector witness, LeftBound43214.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43222

namespace LeftBound43226
def owner : Owner := ⟨.program ⟨214⟩, ⟨27025⟩⟩
def transferEvent : Nat := 43226
def frameStart : Nat := 43149
def rule : BoundRule := .product (.predecessor 0 43224 .coefficient) (.predecessor 1 43225 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43224 .coefficient)
      LeftBound43222.bound (LeftBound43222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43222.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43225 .coefficient)
      LeftAuthority43199.bound (LeftAuthority43199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43199.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43199.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43222.bound LeftAuthority43199.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43222.bound, LeftAuthority43199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43222.actual selector witness) * (LeftAuthority43199.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43226

namespace LeftBound43237
def owner : Owner := ⟨.program ⟨214⟩, ⟨17352⟩⟩
def transferEvent : Nat := 43237
def frameStart : Nat := 43149
def rule : BoundRule := .product (.predecessor 0 43235 .coefficient) (.predecessor 1 43236 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43235 .coefficient)
      LeftAuthority43210.bound (LeftAuthority43210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43236 .coefficient)
      LeftAuthority43233.bound (LeftAuthority43233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43233.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority43210.bound LeftAuthority43233.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43210.bound, LeftAuthority43233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority43210.actual selector witness) * (LeftAuthority43233.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43237

namespace LeftBound43245
def owner : Owner := ⟨.program ⟨214⟩, ⟨17353⟩⟩
def transferEvent : Nat := 43245
def frameStart : Nat := 43149
def rule : BoundRule := .sum [.predecessor 0 43243 .coefficient, .predecessor 1 43244 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43243 .coefficient)
      LeftAuthority43241.bound (LeftAuthority43241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43241.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43244 .coefficient)
      LeftBound43237.bound (LeftBound43237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority43241.bound, LeftBound43237.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43241.bound, LeftBound43237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority43241.actual selector witness, LeftBound43237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43245

namespace LeftBound43249
def owner : Owner := ⟨.program ⟨214⟩, ⟨27029⟩⟩
def transferEvent : Nat := 43249
def frameStart : Nat := 43149
def rule : BoundRule := .sum [.predecessor 0 43247 .coefficient, .predecessor 1 43248 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43247 .coefficient)
      LeftBound43245.bound (LeftBound43245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43248 .coefficient)
      LeftBound43226.bound (LeftBound43226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43226.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43245.bound, LeftBound43226.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43245.bound, LeftBound43226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43245.actual selector witness, LeftBound43226.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43249

namespace LeftBound43262
def owner : Owner := ⟨.program ⟨214⟩, ⟨27027⟩⟩
def transferEvent : Nat := 43262
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43260 .coefficient, .predecessor 1 43261 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43260 .coefficient)
      LeftBound43091.bound (LeftBound43091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43261 .coefficient)
      LeftBound43074.bound (LeftBound43074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43074.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43091.bound, LeftBound43074.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43091.bound, LeftBound43074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43091.actual selector witness, LeftBound43074.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43262

namespace LeftBound43265
def owner : Owner := ⟨.program ⟨214⟩, ⟨27027⟩⟩
def transferEvent : Nat := 43265
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 43259 .summary, .result 43081 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43259 .summary)
      LeftBound43093.bound (LeftBound43093.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20835⟩⟩) (rawTerms := some (Proof.Events168.exact43259RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43093.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43081 .summary)
      LeftBound43076.bound (LeftBound43076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27026⟩⟩) (rawTerms := some (Proof.Events168.exact43081RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43076.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43093.bound, LeftBound43076.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43093.bound, LeftBound43076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43093.actual selector witness, LeftBound43076.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43265

namespace LeftBound43289
def owner : Owner := ⟨.program ⟨214⟩, ⟨10996⟩⟩
def transferEvent : Nat := 43289
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 43287 .coefficient) (.predecessor 1 43288 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43287 .coefficient)
      LeftAuthority1934.bound (LeftAuthority1934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1934.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43288 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1934.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1934.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1934.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43289

namespace LeftBound43294
def owner : Owner := ⟨.program ⟨214⟩, ⟨7306⟩⟩
def transferEvent : Nat := 43294
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43292 .coefficient) (.predecessor 1 43293 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43292 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43293 .coefficient)
      LeftBound13986.bound (LeftBound13986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound13986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound13986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound13986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43294

namespace LeftBound43299
def owner : Owner := ⟨.program ⟨214⟩, ⟨10997⟩⟩
def transferEvent : Nat := 43299
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43297 .coefficient, .predecessor 1 43298 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43297 .coefficient)
      LeftBound43294.bound (LeftBound43294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43298 .coefficient)
      LeftBound43289.bound (LeftBound43289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43289.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43294.bound, LeftBound43289.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43294.bound, LeftBound43289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43294.actual selector witness, LeftBound43289.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43299

namespace LeftBound43303
def owner : Owner := ⟨.program ⟨214⟩, ⟨10998⟩⟩
def transferEvent : Nat := 43303
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43301 .coefficient, .predecessor 1 43302 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43301 .coefficient)
      LeftBound43299.bound (LeftBound43299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43302 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43299.bound, LeftBound13978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43299.bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43299.actual selector witness, LeftBound13978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43303

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
