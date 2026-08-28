import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard610

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound89935
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def transferEvent : Nat := 89935
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89933 .coefficient, .predecessor 1 89934 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89933 .coefficient)
      LeftBound89931.bound (LeftBound89931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89934 .coefficient)
      LeftAuthority89907.bound (LeftAuthority89907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89907.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89931.bound, LeftAuthority89907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89931.bound, LeftAuthority89907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89931.actual selector witness, LeftAuthority89907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89935

namespace LeftBound89939
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def transferEvent : Nat := 89939
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89937 .coefficient, .predecessor 1 89938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89937 .coefficient)
      LeftBound89935.bound (LeftBound89935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89938 .coefficient)
      LeftAuthority89904.bound (LeftAuthority89904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89904.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89935.bound, LeftAuthority89904.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89935.bound, LeftAuthority89904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89935.actual selector witness, LeftAuthority89904.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89939

namespace LeftBound89943
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def transferEvent : Nat := 89943
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89941 .coefficient, .predecessor 1 89942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89941 .coefficient)
      LeftBound89939.bound (LeftBound89939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89942 .coefficient)
      LeftAuthority89901.bound (LeftAuthority89901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89901.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89939.bound, LeftAuthority89901.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89939.bound, LeftAuthority89901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89939.actual selector witness, LeftAuthority89901.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89943

namespace LeftBound89947
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def transferEvent : Nat := 89947
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89945 .coefficient, .predecessor 1 89946 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89945 .coefficient)
      LeftBound89943.bound (LeftBound89943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89946 .coefficient)
      LeftAuthority89898.bound (LeftAuthority89898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89943.bound, LeftAuthority89898.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89943.bound, LeftAuthority89898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89943.actual selector witness, LeftAuthority89898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89947

namespace LeftBound89951
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def transferEvent : Nat := 89951
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89949 .coefficient, .predecessor 1 89950 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89949 .coefficient)
      LeftBound89947.bound (LeftBound89947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89950 .coefficient)
      LeftAuthority89895.bound (LeftAuthority89895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89895.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89895.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89947.bound, LeftAuthority89895.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89947.bound, LeftAuthority89895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89947.actual selector witness, LeftAuthority89895.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89951

namespace LeftBound89955
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def transferEvent : Nat := 89955
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89953 .coefficient, .predecessor 1 89954 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89953 .coefficient)
      LeftBound89951.bound (LeftBound89951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89954 .coefficient)
      LeftAuthority89892.bound (LeftAuthority89892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89951.bound, LeftAuthority89892.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89951.bound, LeftAuthority89892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89951.actual selector witness, LeftAuthority89892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89955

namespace LeftBound89959
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def transferEvent : Nat := 89959
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89957 .coefficient, .predecessor 1 89958 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89957 .coefficient)
      LeftBound89955.bound (LeftBound89955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89955.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89958 .coefficient)
      LeftAuthority89889.bound (LeftAuthority89889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89955.bound, LeftAuthority89889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89955.bound, LeftAuthority89889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89955.actual selector witness, LeftAuthority89889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89959

namespace LeftBound89963
def owner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def transferEvent : Nat := 89963
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89961 .coefficient, .predecessor 1 89962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89961 .coefficient)
      LeftBound89959.bound (LeftBound89959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89962 .coefficient)
      LeftAuthority89886.bound (LeftAuthority89886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89886.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89959.bound, LeftAuthority89886.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89959.bound, LeftAuthority89886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89959.actual selector witness, LeftAuthority89886.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89963

namespace LeftBound89967
def owner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def transferEvent : Nat := 89967
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89965 .coefficient, .predecessor 1 89966 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89965 .coefficient)
      LeftBound89963.bound (LeftBound89963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89966 .coefficient)
      LeftAuthority89883.bound (LeftAuthority89883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89883.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89963.bound, LeftAuthority89883.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89963.bound, LeftAuthority89883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89963.actual selector witness, LeftAuthority89883.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89967

namespace LeftBound89971
def owner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def transferEvent : Nat := 89971
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89969 .coefficient, .predecessor 1 89970 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89969 .coefficient)
      LeftBound89967.bound (LeftBound89967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89970 .coefficient)
      LeftAuthority89880.bound (LeftAuthority89880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89967.bound, LeftAuthority89880.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89967.bound, LeftAuthority89880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89967.actual selector witness, LeftAuthority89880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89971

namespace LeftBound89975
def owner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def transferEvent : Nat := 89975
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89973 .coefficient, .predecessor 1 89974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89973 .coefficient)
      LeftBound89971.bound (LeftBound89971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89974 .coefficient)
      LeftAuthority89877.bound (LeftAuthority89877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89877.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89971.bound, LeftAuthority89877.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89971.bound, LeftAuthority89877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89971.actual selector witness, LeftAuthority89877.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89975

namespace LeftBound89979
def owner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def transferEvent : Nat := 89979
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89977 .coefficient, .predecessor 1 89978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89977 .coefficient)
      LeftBound89975.bound (LeftBound89975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89978 .coefficient)
      LeftAuthority89874.bound (LeftAuthority89874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89874.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89975.bound, LeftAuthority89874.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89975.bound, LeftAuthority89874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89975.actual selector witness, LeftAuthority89874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89979

namespace LeftBound89983
def owner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def transferEvent : Nat := 89983
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89981 .coefficient, .predecessor 1 89982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89981 .coefficient)
      LeftBound89979.bound (LeftBound89979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89982 .coefficient)
      LeftAuthority89871.bound (LeftAuthority89871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89979.bound, LeftAuthority89871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89979.bound, LeftAuthority89871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89979.actual selector witness, LeftAuthority89871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89983

namespace LeftBound89987
def owner : Owner := ⟨.program ⟨214⟩, ⟨6811⟩⟩
def transferEvent : Nat := 89987
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89985 .coefficient, .predecessor 1 89986 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89985 .coefficient)
      LeftBound89983.bound (LeftBound89983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89986 .coefficient)
      LeftAuthority89868.bound (LeftAuthority89868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89868.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89983.bound, LeftAuthority89868.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89983.bound, LeftAuthority89868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89983.actual selector witness, LeftAuthority89868.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89987

namespace LeftBound89991
def owner : Owner := ⟨.program ⟨214⟩, ⟨18650⟩⟩
def transferEvent : Nat := 89991
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89989 .coefficient, .predecessor 1 89990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89989 .coefficient)
      LeftBound89987.bound (LeftBound89987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89990 .coefficient)
      LeftBound89847.bound (LeftBound89847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89847.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89987.bound, LeftBound89847.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89987.bound, LeftBound89847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89987.actual selector witness, LeftBound89847.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89991

namespace LeftBound89995
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def transferEvent : Nat := 89995
def frameStart : Nat := 89317
def rule : BoundRule := .product (.predecessor 0 89993 .coefficient) (.predecessor 1 89994 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89993 .coefficient)
      LeftBound89991.bound (LeftBound89991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89994 .coefficient)
      LeftAuthority89832.bound (LeftAuthority89832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89832.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89832.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound89991.bound LeftAuthority89832.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89991.bound, LeftAuthority89832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound89991.actual selector witness) * (LeftAuthority89832.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound89995

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
