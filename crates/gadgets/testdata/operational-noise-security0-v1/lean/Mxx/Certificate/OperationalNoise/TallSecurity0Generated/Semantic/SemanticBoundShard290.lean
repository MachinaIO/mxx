import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard289

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound43015
def owner : Owner := ⟨.program ⟨214⟩, ⟨7841⟩⟩
def transferEvent : Nat := 43015
def frameStart : Nat := 42940
def rule : BoundRule := .scale (.predecessor 0 43013 .coefficient) (.value (.predecessor 1 43014 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43013 .coefficient)
      LeftAuthority43011.bound (LeftAuthority43011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43011.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43014 .coefficient)
      LeftAuthority43002.bound (LeftAuthority43002.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority43002.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority43011.bound LeftAuthority43002.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43011.bound, LeftAuthority43002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43011.actual selector witness) * (LeftAuthority43002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43015

namespace LeftBound43018
def owner : Owner := ⟨.program ⟨214⟩, ⟨6792⟩⟩
def transferEvent : Nat := 43018
def frameStart : Nat := 42940
def rule : BoundRule := .identity (.predecessor 0 43017 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43017 .coefficient)
      LeftAuthority43005.bound (LeftAuthority43005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact43006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43005.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43005.derived selector witness)

def rawBound : CoeffClass := LeftAuthority43005.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority43005.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43018

namespace LeftBound43022
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def transferEvent : Nat := 43022
def frameStart : Nat := 42940
def rule : BoundRule := .product (.predecessor 0 43020 .coefficient) (.predecessor 1 43021 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43020 .coefficient)
      LeftBound43018.bound (LeftBound43018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43021 .coefficient)
      LeftBound43015.bound (LeftBound43015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43015.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43018.bound LeftBound43015.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43018.bound, LeftBound43015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43018.actual selector witness) * (LeftBound43015.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43022

namespace LeftBound43027
def owner : Owner := ⟨.program ⟨214⟩, ⟨12281⟩⟩
def transferEvent : Nat := 43027
def frameStart : Nat := 42940
def rule : BoundRule := .sum [.predecessor 0 43025 .coefficient, .predecessor 1 43026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43025 .coefficient)
      LeftBound43022.bound (LeftBound43022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43026 .coefficient)
      LeftBound42999.bound (LeftBound42999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact43001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43022.bound, LeftBound42999.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43022.bound, LeftBound42999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43022.actual selector witness, LeftBound42999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43027

namespace LeftBound43031
def owner : Owner := ⟨.program ⟨214⟩, ⟨25309⟩⟩
def transferEvent : Nat := 43031
def frameStart : Nat := 42940
def rule : BoundRule := .product (.predecessor 0 43029 .coefficient) (.predecessor 1 43030 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43029 .coefficient)
      LeftBound43027.bound (LeftBound43027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43030 .coefficient)
      LeftAuthority42984.bound (LeftAuthority42984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42984.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42984.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43027.bound LeftAuthority42984.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43027.bound, LeftAuthority42984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43027.actual selector witness) * (LeftAuthority42984.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43031

namespace LeftBound43042
def owner : Owner := ⟨.program ⟨214⟩, ⟨15432⟩⟩
def transferEvent : Nat := 43042
def frameStart : Nat := 42940
def rule : BoundRule := .product (.predecessor 0 43040 .coefficient) (.predecessor 1 43041 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43040 .coefficient)
      LeftAuthority42995.bound (LeftAuthority42995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42995.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43041 .coefficient)
      LeftAuthority43038.bound (LeftAuthority43038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43038.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority42995.bound LeftAuthority43038.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42995.bound, LeftAuthority43038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority42995.actual selector witness) * (LeftAuthority43038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43042

namespace LeftBound43050
def owner : Owner := ⟨.program ⟨214⟩, ⟨15433⟩⟩
def transferEvent : Nat := 43050
def frameStart : Nat := 42940
def rule : BoundRule := .sum [.predecessor 0 43048 .coefficient, .predecessor 1 43049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43048 .coefficient)
      LeftAuthority43046.bound (LeftAuthority43046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43046.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43049 .coefficient)
      LeftBound43042.bound (LeftBound43042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority43046.bound, LeftBound43042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43046.bound, LeftBound43042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority43046.actual selector witness, LeftBound43042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43050

namespace LeftBound43054
def owner : Owner := ⟨.program ⟨214⟩, ⟨25310⟩⟩
def transferEvent : Nat := 43054
def frameStart : Nat := 42940
def rule : BoundRule := .sum [.predecessor 0 43052 .coefficient, .predecessor 1 43053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43052 .coefficient)
      LeftBound43050.bound (LeftBound43050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43053 .coefficient)
      LeftBound43031.bound (LeftBound43031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43050.bound, LeftBound43031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43050.bound, LeftBound43031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43050.actual selector witness, LeftBound43031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43054

namespace LeftBound43067
def owner : Owner := ⟨.program ⟨214⟩, ⟨25308⟩⟩
def transferEvent : Nat := 43067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43065 .coefficient, .predecessor 1 43066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43065 .coefficient)
      LeftBound42888.bound (LeftBound42888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43066 .coefficient)
      LeftBound42871.bound (LeftBound42871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42888.bound, LeftBound42871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42888.bound, LeftBound42871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42888.actual selector witness, LeftBound42871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43067

namespace LeftBound43070
def owner : Owner := ⟨.program ⟨214⟩, ⟨25308⟩⟩
def transferEvent : Nat := 43070
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 43064 .summary, .result 42878 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43064 .summary)
      LeftBound42890.bound (LeftBound42890.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19251⟩⟩) (rawTerms := some (Proof.Events168.exact43064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42878 .summary)
      LeftBound42873.bound (LeftBound42873.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25307⟩⟩) (rawTerms := some (Proof.Events167.exact42878RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42890.bound, LeftBound42873.bound]
def bound : CoeffClass := .finite ⟨352024077676544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42890.bound, LeftBound42873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42890.actual selector witness, LeftBound42873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43070

namespace LeftBound43074
def owner : Owner := ⟨.program ⟨214⟩, ⟨27026⟩⟩
def transferEvent : Nat := 43074
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43072 .coefficient) (.predecessor 1 43073 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43072 .coefficient)
      LeftBound43067.bound (LeftBound43067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43073 .coefficient)
      LeftAuthority42793.bound (LeftAuthority42793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42793.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42793.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43067.bound LeftAuthority42793.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43067.bound, LeftAuthority42793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43067.actual selector witness) * (LeftAuthority42793.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43074

namespace LeftBound43075
def owner : Owner := ⟨.program ⟨214⟩, ⟨27026⟩⟩
def transferEvent : Nat := 43075
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27024⟩⟩]⟩ [⟨.result 42794 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42794 .coefficient)
      LeftAuthority42793.bound (LeftAuthority42793.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27024⟩⟩) (rawTerms := some (Proof.Events167.exact42794RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42793.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42793.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority42793.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42793.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43075

namespace LeftBound43076
def owner : Owner := ⟨.program ⟨214⟩, ⟨27026⟩⟩
def transferEvent : Nat := 43076
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43071 .summary) (.transfer 43075) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43071 .summary)
      LeftBound43070.bound (LeftBound43070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25308⟩⟩) (rawTerms := some (Proof.Events168.exact43071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43075)
      LeftBound43075.bound (LeftBound43075.actual selector witness) := by
  exact .transfer (LeftBound43075.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43070.bound LeftBound43075.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43070.bound, LeftBound43075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43070.actual selector witness) * (LeftBound43075.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43076

namespace LeftBound43087
def owner : Owner := ⟨.program ⟨214⟩, ⟨20834⟩⟩
def transferEvent : Nat := 43087
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 43085 .coefficient) (.value (.predecessor 1 43086 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43085 .coefficient)
      LeftAuthority43083.bound (LeftAuthority43083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43086 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority43083.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43083.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43083.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43087

namespace LeftBound43091
def owner : Owner := ⟨.program ⟨214⟩, ⟨20835⟩⟩
def transferEvent : Nat := 43091
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43089 .coefficient) (.predecessor 1 43090 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43089 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43090 .coefficient)
      LeftBound43087.bound (LeftBound43087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound43087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound43087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound43087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43091

namespace LeftBound43092
def owner : Owner := ⟨.program ⟨214⟩, ⟨20835⟩⟩
def transferEvent : Nat := 43092
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20832⟩⟩]⟩ [⟨.result 43084 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43084 .coefficient)
      LeftAuthority43083.bound (LeftAuthority43083.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20832⟩⟩) (rawTerms := some (Proof.Events168.exact43084RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43083.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43083.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43083.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43092

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
