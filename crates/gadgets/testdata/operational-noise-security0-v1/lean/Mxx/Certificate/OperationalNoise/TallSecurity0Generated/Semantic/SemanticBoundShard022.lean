import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard017
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound5905
def owner : Owner := ⟨.program ⟨214⟩, ⟨7657⟩⟩
def transferEvent : Nat := 5905
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5903 .coefficient, .predecessor 1 5904 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5903 .coefficient)
      LeftBound5901.bound (LeftBound5901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5904 .coefficient)
      LeftBound5745.bound (LeftBound5745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5745.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5901.bound, LeftBound5745.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5901.bound, LeftBound5745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5901.actual selector witness, LeftBound5745.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5905

namespace LeftBound5909
def owner : Owner := ⟨.program ⟨214⟩, ⟨7658⟩⟩
def transferEvent : Nat := 5909
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5907 .coefficient, .predecessor 1 5908 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5907 .coefficient)
      LeftBound5905.bound (LeftBound5905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5905.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5908 .coefficient)
      LeftBound5725.bound (LeftBound5725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5725.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5905.bound, LeftBound5725.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5905.bound, LeftBound5725.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5905.actual selector witness, LeftBound5725.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5909

namespace LeftBound5913
def owner : Owner := ⟨.program ⟨214⟩, ⟨7659⟩⟩
def transferEvent : Nat := 5913
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5911 .coefficient, .predecessor 1 5912 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5911 .coefficient)
      LeftBound5909.bound (LeftBound5909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5912 .coefficient)
      LeftBound5705.bound (LeftBound5705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5705.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5909.bound, LeftBound5705.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5909.bound, LeftBound5705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5909.actual selector witness, LeftBound5705.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5913

namespace LeftBound5917
def owner : Owner := ⟨.program ⟨214⟩, ⟨7660⟩⟩
def transferEvent : Nat := 5917
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5915 .coefficient, .predecessor 1 5916 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5915 .coefficient)
      LeftBound5913.bound (LeftBound5913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5916 .coefficient)
      LeftBound5685.bound (LeftBound5685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5685.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5685.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5913.bound, LeftBound5685.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5913.bound, LeftBound5685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5913.actual selector witness, LeftBound5685.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5917

namespace LeftBound5921
def owner : Owner := ⟨.program ⟨214⟩, ⟨7661⟩⟩
def transferEvent : Nat := 5921
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5919 .coefficient, .predecessor 1 5920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5919 .coefficient)
      LeftBound5917.bound (LeftBound5917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5920 .coefficient)
      LeftBound5665.bound (LeftBound5665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5665.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5665.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5917.bound, LeftBound5665.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5917.bound, LeftBound5665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5917.actual selector witness, LeftBound5665.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5921

namespace LeftBound5925
def owner : Owner := ⟨.program ⟨214⟩, ⟨7662⟩⟩
def transferEvent : Nat := 5925
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5923 .coefficient, .predecessor 1 5924 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5923 .coefficient)
      LeftBound5921.bound (LeftBound5921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5924 .coefficient)
      LeftBound5645.bound (LeftBound5645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5645.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5645.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5921.bound, LeftBound5645.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5921.bound, LeftBound5645.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5921.actual selector witness, LeftBound5645.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5925

namespace LeftBound5929
def owner : Owner := ⟨.program ⟨214⟩, ⟨7663⟩⟩
def transferEvent : Nat := 5929
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5927 .coefficient, .predecessor 1 5928 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5927 .coefficient)
      LeftBound5925.bound (LeftBound5925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5928 .coefficient)
      LeftBound5625.bound (LeftBound5625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5625.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5925.bound, LeftBound5625.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5925.bound, LeftBound5625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5925.actual selector witness, LeftBound5625.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5929

namespace LeftBound5933
def owner : Owner := ⟨.program ⟨214⟩, ⟨7664⟩⟩
def transferEvent : Nat := 5933
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5931 .coefficient, .predecessor 1 5932 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5931 .coefficient)
      LeftBound5929.bound (LeftBound5929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5929.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5932 .coefficient)
      LeftBound5605.bound (LeftBound5605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5605.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5929.bound, LeftBound5605.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5929.bound, LeftBound5605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5929.actual selector witness, LeftBound5605.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5933

namespace LeftBound5937
def owner : Owner := ⟨.program ⟨214⟩, ⟨7665⟩⟩
def transferEvent : Nat := 5937
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5935 .coefficient, .predecessor 1 5936 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5935 .coefficient)
      LeftBound5933.bound (LeftBound5933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5933.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5933.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5936 .coefficient)
      LeftBound5585.bound (LeftBound5585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5933.bound, LeftBound5585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5933.bound, LeftBound5585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5933.actual selector witness, LeftBound5585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5937

namespace LeftBound5941
def owner : Owner := ⟨.program ⟨214⟩, ⟨7666⟩⟩
def transferEvent : Nat := 5941
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5939 .coefficient, .predecessor 1 5940 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5939 .coefficient)
      LeftBound5937.bound (LeftBound5937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5937.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5940 .coefficient)
      LeftBound5565.bound (LeftBound5565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5565.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5565.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5937.bound, LeftBound5565.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5937.bound, LeftBound5565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5937.actual selector witness, LeftBound5565.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5941

namespace LeftBound5945
def owner : Owner := ⟨.program ⟨214⟩, ⟨7667⟩⟩
def transferEvent : Nat := 5945
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5943 .coefficient, .predecessor 1 5944 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5943 .coefficient)
      LeftBound5941.bound (LeftBound5941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5944 .coefficient)
      LeftBound5545.bound (LeftBound5545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5545.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5941.bound, LeftBound5545.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5941.bound, LeftBound5545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5941.actual selector witness, LeftBound5545.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5945

namespace LeftBound5949
def owner : Owner := ⟨.program ⟨214⟩, ⟨7668⟩⟩
def transferEvent : Nat := 5949
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5947 .coefficient, .predecessor 1 5948 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5947 .coefficient)
      LeftBound5945.bound (LeftBound5945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5945.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5948 .coefficient)
      LeftBound5525.bound (LeftBound5525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5525.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5525.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5945.bound, LeftBound5525.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5945.bound, LeftBound5525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5945.actual selector witness, LeftBound5525.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5949

namespace LeftBound5953
def owner : Owner := ⟨.program ⟨214⟩, ⟨7795⟩⟩
def transferEvent : Nat := 5953
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5951 .coefficient, .predecessor 1 5952 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5951 .coefficient)
      LeftBound5949.bound (LeftBound5949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5952 .coefficient)
      LeftBound5505.bound (LeftBound5505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5949.bound, LeftBound5505.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5949.bound, LeftBound5505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5949.actual selector witness, LeftBound5505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5953

namespace LeftBound5960
def owner : Owner := ⟨.program ⟨214⟩, ⟨7886⟩⟩
def transferEvent : Nat := 5960
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 5958 .coefficient) (.value (.predecessor 1 5959 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5958 .coefficient)
      LeftAuthority5956.bound (LeftAuthority5956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5959 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority5956.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5956.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5956.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound5960

namespace LeftBound5967
def owner : Owner := ⟨.program ⟨214⟩, ⟨7887⟩⟩
def transferEvent : Nat := 5967
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5965 .coefficient) (.predecessor 1 5966 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5965 .coefficient)
      LeftAuthority5963.bound (LeftAuthority5963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5966 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftAuthority5963.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5963.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftAuthority5963.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5967

namespace LeftBound5972
def owner : Owner := ⟨.program ⟨214⟩, ⟨7911⟩⟩
def transferEvent : Nat := 5972
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5970 .coefficient) (.predecessor 1 5971 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5970 .coefficient)
      LeftBound5967.bound (LeftBound5967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5971 .coefficient)
      LeftBound5486.bound (LeftBound5486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound5967.bound LeftBound5486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5967.bound, LeftBound5486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound5967.actual selector witness) * (LeftBound5486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5972

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
