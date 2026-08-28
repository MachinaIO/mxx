import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard184

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27911
def owner : Owner := ⟨.program ⟨214⟩, ⟨6793⟩⟩
def transferEvent : Nat := 27911
def frameStart : Nat := 27833
def rule : BoundRule := .identity (.predecessor 0 27910 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27910 .coefficient)
      LeftAuthority27898.bound (LeftAuthority27898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27898.derived selector witness)

def rawBound : CoeffClass := LeftAuthority27898.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority27898.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27911

namespace LeftBound27915
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def transferEvent : Nat := 27915
def frameStart : Nat := 27833
def rule : BoundRule := .product (.predecessor 0 27913 .coefficient) (.predecessor 1 27914 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27913 .coefficient)
      LeftBound27911.bound (LeftBound27911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27911.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27914 .coefficient)
      LeftBound27908.bound (LeftBound27908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27908.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27911.bound LeftBound27908.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27911.bound, LeftBound27908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27911.actual selector witness) * (LeftBound27908.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27915

namespace LeftBound27920
def owner : Owner := ⟨.program ⟨214⟩, ⟨13678⟩⟩
def transferEvent : Nat := 27920
def frameStart : Nat := 27833
def rule : BoundRule := .sum [.predecessor 0 27918 .coefficient, .predecessor 1 27919 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27918 .coefficient)
      LeftBound27915.bound (LeftBound27915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27915.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27919 .coefficient)
      LeftBound27892.bound (LeftBound27892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27915.bound, LeftBound27892.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27915.bound, LeftBound27892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27915.actual selector witness, LeftBound27892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27920

namespace LeftBound27924
def owner : Owner := ⟨.program ⟨214⟩, ⟨25853⟩⟩
def transferEvent : Nat := 27924
def frameStart : Nat := 27833
def rule : BoundRule := .product (.predecessor 0 27922 .coefficient) (.predecessor 1 27923 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27922 .coefficient)
      LeftBound27920.bound (LeftBound27920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27923 .coefficient)
      LeftAuthority27877.bound (LeftAuthority27877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27877.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27920.bound LeftAuthority27877.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27920.bound, LeftAuthority27877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27920.actual selector witness) * (LeftAuthority27877.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27924

namespace LeftBound27935
def owner : Owner := ⟨.program ⟨214⟩, ⟨15597⟩⟩
def transferEvent : Nat := 27935
def frameStart : Nat := 27833
def rule : BoundRule := .product (.predecessor 0 27933 .coefficient) (.predecessor 1 27934 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27933 .coefficient)
      LeftAuthority27888.bound (LeftAuthority27888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27934 .coefficient)
      LeftAuthority27931.bound (LeftAuthority27931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27931.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority27888.bound LeftAuthority27931.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27888.bound, LeftAuthority27931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority27888.actual selector witness) * (LeftAuthority27931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27935

namespace LeftBound27943
def owner : Owner := ⟨.program ⟨214⟩, ⟨15598⟩⟩
def transferEvent : Nat := 27943
def frameStart : Nat := 27833
def rule : BoundRule := .sum [.predecessor 0 27941 .coefficient, .predecessor 1 27942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27941 .coefficient)
      LeftAuthority27939.bound (LeftAuthority27939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27942 .coefficient)
      LeftBound27935.bound (LeftBound27935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27939.bound, LeftBound27935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27939.bound, LeftBound27935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority27939.actual selector witness, LeftBound27935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27943

namespace LeftBound27947
def owner : Owner := ⟨.program ⟨214⟩, ⟨25854⟩⟩
def transferEvent : Nat := 27947
def frameStart : Nat := 27833
def rule : BoundRule := .sum [.predecessor 0 27945 .coefficient, .predecessor 1 27946 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27945 .coefficient)
      LeftBound27943.bound (LeftBound27943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27946 .coefficient)
      LeftBound27924.bound (LeftBound27924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27943.bound, LeftBound27924.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27943.bound, LeftBound27924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27943.actual selector witness, LeftBound27924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27947

namespace LeftBound27960
def owner : Owner := ⟨.program ⟨214⟩, ⟨25852⟩⟩
def transferEvent : Nat := 27960
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27958 .coefficient, .predecessor 1 27959 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27958 .coefficient)
      LeftBound27781.bound (LeftBound27781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27959 .coefficient)
      LeftBound27764.bound (LeftBound27764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27781.bound, LeftBound27764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27781.bound, LeftBound27764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27781.actual selector witness, LeftBound27764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27960

namespace LeftBound27963
def owner : Owner := ⟨.program ⟨214⟩, ⟨25852⟩⟩
def transferEvent : Nat := 27963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 27957 .summary, .result 27771 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27957 .summary)
      LeftBound27783.bound (LeftBound27783.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19327⟩⟩) (rawTerms := some (Proof.Events109.exact27957RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27771 .summary)
      LeftBound27766.bound (LeftBound27766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25851⟩⟩) (rawTerms := some (Proof.Events108.exact27771RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27783.bound, LeftBound27766.bound]
def bound : CoeffClass := .finite ⟨352036291489792, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27783.bound, LeftBound27766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27783.actual selector witness, LeftBound27766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27963

namespace LeftBound27967
def owner : Owner := ⟨.program ⟨214⟩, ⟨27256⟩⟩
def transferEvent : Nat := 27967
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27965 .coefficient) (.predecessor 1 27966 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27965 .coefficient)
      LeftBound27960.bound (LeftBound27960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27966 .coefficient)
      LeftAuthority27686.bound (LeftAuthority27686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27686.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27686.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27960.bound LeftAuthority27686.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27960.bound, LeftAuthority27686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27960.actual selector witness) * (LeftAuthority27686.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27967

namespace LeftBound27968
def owner : Owner := ⟨.program ⟨214⟩, ⟨27256⟩⟩
def transferEvent : Nat := 27968
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩ [⟨.result 27687 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27687 .coefficient)
      LeftAuthority27686.bound (LeftAuthority27686.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27254⟩⟩) (rawTerms := some (Proof.Events108.exact27687RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27686.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27686.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27686.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27686.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27968

namespace LeftBound27969
def owner : Owner := ⟨.program ⟨214⟩, ⟨27256⟩⟩
def transferEvent : Nat := 27969
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27964 .summary) (.transfer 27968) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27964 .summary)
      LeftBound27963.bound (LeftBound27963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25852⟩⟩) (rawTerms := some (Proof.Events109.exact27964RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27968)
      LeftBound27968.bound (LeftBound27968.actual selector witness) := by
  exact .transfer (LeftBound27968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27963.bound LeftBound27968.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27963.bound, LeftBound27968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27963.actual selector witness) * (LeftBound27968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27969

namespace LeftBound27980
def owner : Owner := ⟨.program ⟨214⟩, ⟨20982⟩⟩
def transferEvent : Nat := 27980
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 27978 .coefficient) (.value (.predecessor 1 27979 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27978 .coefficient)
      LeftAuthority27976.bound (LeftAuthority27976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27979 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority27976.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27976.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27976.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27980

namespace LeftBound27984
def owner : Owner := ⟨.program ⟨214⟩, ⟨20983⟩⟩
def transferEvent : Nat := 27984
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27982 .coefficient) (.predecessor 1 27983 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27982 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27983 .coefficient)
      LeftBound27980.bound (LeftBound27980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound27980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound27980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound27980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27984

namespace LeftBound27985
def owner : Owner := ⟨.program ⟨214⟩, ⟨20983⟩⟩
def transferEvent : Nat := 27985
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩ [⟨.result 27977 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27977 .coefficient)
      LeftAuthority27976.bound (LeftAuthority27976.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20980⟩⟩) (rawTerms := some (Proof.Events109.exact27977RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27976.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27976.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27976.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27985

namespace LeftBound27986
def owner : Owner := ⟨.program ⟨214⟩, ⟨20983⟩⟩
def transferEvent : Nat := 27986
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 27985) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27985)
      LeftBound27985.bound (LeftBound27985.actual selector witness) := by
  exact .transfer (LeftBound27985.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound27985.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound27985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound27985.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27986

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
