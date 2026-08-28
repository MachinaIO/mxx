import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard419
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard421
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard422
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard423
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard424
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard425
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard426
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard427
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard429
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard430
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard433

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64824
def owner : Owner := ⟨.program ⟨214⟩, ⟨26792⟩⟩
def transferEvent : Nat := 64824
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64820 .summary, .result 64359 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64820 .summary)
      LeftBound64819.bound (LeftBound64819.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26575⟩⟩) (rawTerms := some (Proof.Events253.exact64820RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64359 .summary)
      LeftBound64354.bound (LeftBound64354.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26791⟩⟩) (rawTerms := some (Proof.Events251.exact64359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64819.bound, LeftBound64354.bound]
def bound : CoeffClass := .finite ⟨14223885201645539505274355764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64819.bound, LeftBound64354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64819.actual selector witness, LeftBound64354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64824

namespace LeftBound64828
def owner : Owner := ⟨.program ⟨214⟩, ⟨27009⟩⟩
def transferEvent : Nat := 64828
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64826 .coefficient, .predecessor 1 64827 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64826 .coefficient)
      LeftBound64823.bound (LeftBound64823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64827 .coefficient)
      LeftBound64140.bound (LeftBound64140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64140.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64823.bound, LeftBound64140.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64823.bound, LeftBound64140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64823.actual selector witness, LeftBound64140.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64828

namespace LeftBound64829
def owner : Owner := ⟨.program ⟨214⟩, ⟨27009⟩⟩
def transferEvent : Nat := 64829
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64825 .summary, .result 64147 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64825 .summary)
      LeftBound64824.bound (LeftBound64824.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26792⟩⟩) (rawTerms := some (Proof.Events253.exact64825RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64147 .summary)
      LeftBound64142.bound (LeftBound64142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27008⟩⟩) (rawTerms := some (Proof.Events250.exact64147RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64824.bound, LeftBound64142.bound]
def bound : CoeffClass := .finite ⟨18965303649908456346701791284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64824.bound, LeftBound64142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64824.actual selector witness, LeftBound64142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64829

namespace LeftBound64833
def owner : Owner := ⟨.program ⟨214⟩, ⟨27226⟩⟩
def transferEvent : Nat := 64833
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64831 .coefficient, .predecessor 1 64832 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64831 .coefficient)
      LeftBound64828.bound (LeftBound64828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64832 .coefficient)
      LeftBound63928.bound (LeftBound63928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63928.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64828.bound, LeftBound63928.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64828.bound, LeftBound63928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64828.actual selector witness, LeftBound63928.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64833

namespace LeftBound64834
def owner : Owner := ⟨.program ⟨214⟩, ⟨27226⟩⟩
def transferEvent : Nat := 64834
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64830 .summary, .result 63935 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64830 .summary)
      LeftBound64829.bound (LeftBound64829.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27009⟩⟩) (rawTerms := some (Proof.Events253.exact64830RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63935 .summary)
      LeftBound63930.bound (LeftBound63930.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27225⟩⟩) (rawTerms := some (Proof.Events249.exact63935RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64829.bound, LeftBound63930.bound]
def bound : CoeffClass := .finite ⟨23706886606235022529910538292, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64829.bound, LeftBound63930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64829.actual selector witness, LeftBound63930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64834

namespace LeftBound64838
def owner : Owner := ⟨.program ⟨214⟩, ⟨27443⟩⟩
def transferEvent : Nat := 64838
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64836 .coefficient, .predecessor 1 64837 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64836 .coefficient)
      LeftBound64833.bound (LeftBound64833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64837 .coefficient)
      LeftBound63716.bound (LeftBound63716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63716.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64833.bound, LeftBound63716.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64833.bound, LeftBound63716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64833.actual selector witness, LeftBound63716.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64838

namespace LeftBound64839
def owner : Owner := ⟨.program ⟨214⟩, ⟨27443⟩⟩
def transferEvent : Nat := 64839
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64835 .summary, .result 63723 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64835 .summary)
      LeftBound64834.bound (LeftBound64834.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27226⟩⟩) (rawTerms := some (Proof.Events253.exact64835RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64834.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63723 .summary)
      LeftBound63718.bound (LeftBound63718.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27442⟩⟩) (rawTerms := some (Proof.Events248.exact63723RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64834.bound, LeftBound63718.bound]
def bound : CoeffClass := .finite ⟨28448551816593413384009941044, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64834.bound, LeftBound63718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64834.actual selector witness, LeftBound63718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64839

namespace LeftBound64843
def owner : Owner := ⟨.program ⟨214⟩, ⟨27660⟩⟩
def transferEvent : Nat := 64843
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64841 .coefficient, .predecessor 1 64842 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64841 .coefficient)
      LeftBound64838.bound (LeftBound64838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64842 .coefficient)
      LeftBound63504.bound (LeftBound63504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63504.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63504.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64838.bound, LeftBound63504.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64838.bound, LeftBound63504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64838.actual selector witness, LeftBound63504.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64843

namespace LeftBound64844
def owner : Owner := ⟨.program ⟨214⟩, ⟨27660⟩⟩
def transferEvent : Nat := 64844
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64840 .summary, .result 63511 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64840 .summary)
      LeftBound64839.bound (LeftBound64839.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27443⟩⟩) (rawTerms := some (Proof.Events253.exact64840RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63511 .summary)
      LeftBound63506.bound (LeftBound63506.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27659⟩⟩) (rawTerms := some (Proof.Events248.exact63511RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64839.bound, LeftBound63506.bound]
def bound : CoeffClass := .finite ⟨33190381535015453579890655284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64839.bound, LeftBound63506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64839.actual selector witness, LeftBound63506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64844

namespace LeftBound64848
def owner : Owner := ⟨.program ⟨214⟩, ⟨27877⟩⟩
def transferEvent : Nat := 64848
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64846 .coefficient, .predecessor 1 64847 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64846 .coefficient)
      LeftBound64843.bound (LeftBound64843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64847 .coefficient)
      LeftBound63292.bound (LeftBound63292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63292.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63292.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64843.bound, LeftBound63292.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64843.bound, LeftBound63292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64843.actual selector witness, LeftBound63292.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64848

namespace LeftBound64849
def owner : Owner := ⟨.program ⟨214⟩, ⟨27877⟩⟩
def transferEvent : Nat := 64849
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64845 .summary, .result 63299 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64845 .summary)
      LeftBound64844.bound (LeftBound64844.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27660⟩⟩) (rawTerms := some (Proof.Events253.exact64845RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63299 .summary)
      LeftBound63294.bound (LeftBound63294.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27876⟩⟩) (rawTerms := some (Proof.Events247.exact63299RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63294.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64844.bound, LeftBound63294.bound]
def bound : CoeffClass := .finite ⟨37932293507469318446662025268, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64844.bound, LeftBound63294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64844.actual selector witness, LeftBound63294.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64849

namespace LeftBound64853
def owner : Owner := ⟨.program ⟨214⟩, ⟨28094⟩⟩
def transferEvent : Nat := 64853
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64851 .coefficient, .predecessor 1 64852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64851 .coefficient)
      LeftBound64848.bound (LeftBound64848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64848.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64852 .coefficient)
      LeftBound63080.bound (LeftBound63080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63080.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64848.bound, LeftBound63080.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64848.bound, LeftBound63080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64848.actual selector witness, LeftBound63080.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64853

namespace LeftBound64854
def owner : Owner := ⟨.program ⟨214⟩, ⟨28094⟩⟩
def transferEvent : Nat := 64854
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64850 .summary, .result 63087 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64850 .summary)
      LeftBound64849.bound (LeftBound64849.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27877⟩⟩) (rawTerms := some (Proof.Events253.exact64850RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63087 .summary)
      LeftBound63082.bound (LeftBound63082.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28093⟩⟩) (rawTerms := some (Proof.Events246.exact63087RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63082.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64849.bound, LeftBound63082.bound]
def bound : CoeffClass := .finite ⟨42674369987986832655214706740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64849.bound, LeftBound63082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64849.actual selector witness, LeftBound63082.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64854

namespace LeftBound64858
def owner : Owner := ⟨.program ⟨214⟩, ⟨28311⟩⟩
def transferEvent : Nat := 64858
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64856 .coefficient, .predecessor 1 64857 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64856 .coefficient)
      LeftBound64853.bound (LeftBound64853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64857 .coefficient)
      LeftBound62868.bound (LeftBound62868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62868.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64853.bound, LeftBound62868.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64853.bound, LeftBound62868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64853.actual selector witness, LeftBound62868.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64858

namespace LeftBound64859
def owner : Owner := ⟨.program ⟨214⟩, ⟨28311⟩⟩
def transferEvent : Nat := 64859
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64855 .summary, .result 62875 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64855 .summary)
      LeftBound64854.bound (LeftBound64854.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28094⟩⟩) (rawTerms := some (Proof.Events253.exact64855RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62875 .summary)
      LeftBound62870.bound (LeftBound62870.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28310⟩⟩) (rawTerms := some (Proof.Events245.exact62875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62870.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64854.bound, LeftBound62870.bound]
def bound : CoeffClass := .finite ⟨47416693230599820876439355444, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64854.bound, LeftBound62870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64854.actual selector witness, LeftBound62870.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64859

namespace LeftBound64863
def owner : Owner := ⟨.program ⟨214⟩, ⟨28528⟩⟩
def transferEvent : Nat := 64863
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64861 .coefficient, .predecessor 1 64862 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64861 .coefficient)
      LeftBound64858.bound (LeftBound64858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64862 .coefficient)
      LeftBound62656.bound (LeftBound62656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64858.bound, LeftBound62656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64858.bound, LeftBound62656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64858.actual selector witness, LeftBound62656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64863

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
