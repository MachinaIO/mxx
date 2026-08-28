import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard677

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98862
def owner : Owner := ⟨.program ⟨214⟩, ⟨14181⟩⟩
def transferEvent : Nat := 98862
def frameStart : Nat := 98845
def rule : BoundRule := .product (.predecessor 0 98860 .coefficient) (.predecessor 1 98861 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98860 .coefficient)
      LeftAuthority98858.bound (LeftAuthority98858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98861 .coefficient)
      LeftAuthority98855.bound (LeftAuthority98855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98855.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98855.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority98858.bound LeftAuthority98855.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98858.bound, LeftAuthority98855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority98858.actual selector witness) * (LeftAuthority98855.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98862

namespace LeftBound98866
def owner : Owner := ⟨.program ⟨214⟩, ⟨14182⟩⟩
def transferEvent : Nat := 98866
def frameStart : Nat := 98845
def rule : BoundRule := .identity (.predecessor 0 98865 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98865 .coefficient)
      LeftBound98862.bound (LeftBound98862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98862.derived selector witness)

def rawBound : CoeffClass := LeftBound98862.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound98862.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98866

namespace LeftBound98883
def owner : Owner := ⟨.program ⟨214⟩, ⟨14306⟩⟩
def transferEvent : Nat := 98883
def frameStart : Nat := 98845
def rule : BoundRule := .sum [.predecessor 0 98881 .coefficient, .predecessor 1 98882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98881 .coefficient)
      LeftBound98866.bound (LeftBound98866.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98882 .coefficient)
      LeftAuthority98879.bound (LeftAuthority98879.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98866.bound, LeftAuthority98879.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98866.bound, LeftAuthority98879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98866.actual selector witness, LeftAuthority98879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98883

namespace LeftBound98886
def owner : Owner := ⟨.program ⟨214⟩, ⟨14307⟩⟩
def transferEvent : Nat := 98886
def frameStart : Nat := 98845
def rule : BoundRule := .identity (.predecessor 0 98885 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98885 .coefficient)
      LeftBound98883.bound (LeftBound98883.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98883.derived selector witness)

def rawBound : CoeffClass := LeftBound98883.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound98883.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98886

namespace LeftBound98892
def owner : Owner := ⟨.program ⟨214⟩, ⟨14308⟩⟩
def transferEvent : Nat := 98892
def frameStart : Nat := 98845
def rule : BoundRule := .product (.predecessor 0 98890 .coefficient) (.predecessor 1 98891 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98890 .coefficient)
      LeftAuthority98888.bound (LeftAuthority98888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98891 .coefficient)
      LeftBound98886.bound (LeftBound98886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98886.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98886.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority98888.bound LeftBound98886.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98888.bound, LeftBound98886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority98888.actual selector witness) * (LeftBound98886.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98892

namespace LeftBound98908
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def transferEvent : Nat := 98908
def frameStart : Nat := 98845
def rule : BoundRule := .scale (.predecessor 0 98906 .coefficient) (.value (.predecessor 1 98907 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98906 .coefficient)
      LeftAuthority98904.bound (LeftAuthority98904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98907 .coefficient)
      LeftAuthority98895.bound (LeftAuthority98895.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98895.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98904.bound LeftAuthority98895.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98904.bound, LeftAuthority98895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98904.actual selector witness) * (LeftAuthority98895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98908

namespace LeftBound98911
def owner : Owner := ⟨.program ⟨214⟩, ⟨6759⟩⟩
def transferEvent : Nat := 98911
def frameStart : Nat := 98845
def rule : BoundRule := .identity (.predecessor 0 98910 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98910 .coefficient)
      LeftAuthority98898.bound (LeftAuthority98898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98898.derived selector witness)

def rawBound : CoeffClass := LeftAuthority98898.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority98898.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98911

namespace LeftBound98915
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def transferEvent : Nat := 98915
def frameStart : Nat := 98845
def rule : BoundRule := .product (.predecessor 0 98913 .coefficient) (.predecessor 1 98914 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98913 .coefficient)
      LeftBound98911.bound (LeftBound98911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98911.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98914 .coefficient)
      LeftBound98908.bound (LeftBound98908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98908.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98911.bound LeftBound98908.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98911.bound, LeftBound98908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98911.actual selector witness) * (LeftBound98908.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98915

namespace LeftBound98920
def owner : Owner := ⟨.program ⟨214⟩, ⟨14309⟩⟩
def transferEvent : Nat := 98920
def frameStart : Nat := 98845
def rule : BoundRule := .sum [.predecessor 0 98918 .coefficient, .predecessor 1 98919 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98918 .coefficient)
      LeftBound98915.bound (LeftBound98915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98915.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98919 .coefficient)
      LeftBound98892.bound (LeftBound98892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98915.bound, LeftBound98892.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98915.bound, LeftBound98892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98915.actual selector witness, LeftBound98892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98920

namespace LeftBound98924
def owner : Owner := ⟨.program ⟨214⟩, ⟨26056⟩⟩
def transferEvent : Nat := 98924
def frameStart : Nat := 98845
def rule : BoundRule := .product (.predecessor 0 98922 .coefficient) (.predecessor 1 98923 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98922 .coefficient)
      LeftBound98920.bound (LeftBound98920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98923 .coefficient)
      LeftAuthority98877.bound (LeftAuthority98877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98877.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98920.bound LeftAuthority98877.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98920.bound, LeftAuthority98877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98920.actual selector witness) * (LeftAuthority98877.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98924

namespace LeftBound98935
def owner : Owner := ⟨.program ⟨214⟩, ⟨15932⟩⟩
def transferEvent : Nat := 98935
def frameStart : Nat := 98845
def rule : BoundRule := .product (.predecessor 0 98933 .coefficient) (.predecessor 1 98934 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98933 .coefficient)
      LeftAuthority98888.bound (LeftAuthority98888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98934 .coefficient)
      LeftAuthority98931.bound (LeftAuthority98931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98931.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority98888.bound LeftAuthority98931.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98888.bound, LeftAuthority98931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority98888.actual selector witness) * (LeftAuthority98931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98935

namespace LeftBound98943
def owner : Owner := ⟨.program ⟨214⟩, ⟨15933⟩⟩
def transferEvent : Nat := 98943
def frameStart : Nat := 98845
def rule : BoundRule := .sum [.predecessor 0 98941 .coefficient, .predecessor 1 98942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98941 .coefficient)
      LeftAuthority98939.bound (LeftAuthority98939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98942 .coefficient)
      LeftBound98935.bound (LeftBound98935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority98939.bound, LeftBound98935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98939.bound, LeftBound98935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority98939.actual selector witness, LeftBound98935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98943

namespace LeftBound98947
def owner : Owner := ⟨.program ⟨214⟩, ⟨26057⟩⟩
def transferEvent : Nat := 98947
def frameStart : Nat := 98845
def rule : BoundRule := .sum [.predecessor 0 98945 .coefficient, .predecessor 1 98946 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98945 .coefficient)
      LeftBound98943.bound (LeftBound98943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98946 .coefficient)
      LeftBound98924.bound (LeftBound98924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98943.bound, LeftBound98924.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98943.bound, LeftBound98924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98943.actual selector witness, LeftBound98924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98947

namespace LeftBound98960
def owner : Owner := ⟨.program ⟨214⟩, ⟨26055⟩⟩
def transferEvent : Nat := 98960
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98958 .coefficient, .predecessor 1 98959 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98958 .coefficient)
      LeftBound98805.bound (LeftBound98805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98805.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98959 .coefficient)
      LeftBound98788.bound (LeftBound98788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98788.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98788.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98805.bound, LeftBound98788.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98805.bound, LeftBound98788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98805.actual selector witness, LeftBound98788.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98960

namespace LeftBound98963
def owner : Owner := ⟨.program ⟨214⟩, ⟨26055⟩⟩
def transferEvent : Nat := 98963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 98957 .summary, .result 98795 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98957 .summary)
      LeftBound98807.bound (LeftBound98807.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19520⟩⟩) (rawTerms := some (Proof.Events386.exact98957RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98795 .summary)
      LeftBound98790.bound (LeftBound98790.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26054⟩⟩) (rawTerms := some (Proof.Events385.exact98795RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98790.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98807.bound, LeftBound98790.bound]
def bound : CoeffClass := .finite ⟨352060719116288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98807.bound, LeftBound98790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98807.actual selector witness, LeftBound98790.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98963

namespace LeftBound98967
def owner : Owner := ⟨.program ⟨214⟩, ⟨27833⟩⟩
def transferEvent : Nat := 98967
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98965 .coefficient) (.predecessor 1 98966 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98965 .coefficient)
      LeftBound98960.bound (LeftBound98960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98966 .coefficient)
      LeftAuthority98710.bound (LeftAuthority98710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98710.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98960.bound LeftAuthority98710.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98960.bound, LeftAuthority98710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98960.actual selector witness) * (LeftAuthority98710.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98967

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
