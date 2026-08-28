import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard262
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard266
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard269
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard273
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard277
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard280
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard284
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard288
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard302

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44730
def owner : Owner := ⟨.program ⟨214⟩, ⟨27245⟩⟩
def transferEvent : Nat := 44730
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44728 .coefficient, .predecessor 1 44729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44728 .coefficient)
      LeftBound44725.bound (LeftBound44725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44725.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44729 .coefficient)
      LeftBound42780.bound (LeftBound42780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42780.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42780.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44725.bound, LeftBound42780.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44725.bound, LeftBound42780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44725.actual selector witness, LeftBound42780.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44730

namespace LeftBound44731
def owner : Owner := ⟨.program ⟨214⟩, ⟨27245⟩⟩
def transferEvent : Nat := 44731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44727 .summary, .result 42784 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44727 .summary)
      LeftBound44726.bound (LeftBound44726.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27028⟩⟩) (rawTerms := some (Proof.Events174.exact44727RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42784 .summary)
      LeftBound42783.bound (LeftBound42783.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27244⟩⟩) (rawTerms := some (Proof.Events167.exact42784RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42783.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44726.bound, LeftBound42783.bound]
def bound : CoeffClass := .finite ⟨6459613965234762608640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44726.bound, LeftBound42783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44726.actual selector witness, LeftBound42783.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44731

namespace LeftBound44735
def owner : Owner := ⟨.program ⟨214⟩, ⟨27462⟩⟩
def transferEvent : Nat := 44735
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44733 .coefficient, .predecessor 1 44734 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44733 .coefficient)
      LeftBound44730.bound (LeftBound44730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44734 .coefficient)
      LeftBound42298.bound (LeftBound42298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42298.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44730.bound, LeftBound42298.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44730.bound, LeftBound42298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44730.actual selector witness, LeftBound42298.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44735

namespace LeftBound44736
def owner : Owner := ⟨.program ⟨214⟩, ⟨27462⟩⟩
def transferEvent : Nat := 44736
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44732 .summary, .result 42302 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44732 .summary)
      LeftBound44731.bound (LeftBound44731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27245⟩⟩) (rawTerms := some (Proof.Events174.exact44732RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42302 .summary)
      LeftBound42301.bound (LeftBound42301.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27461⟩⟩) (rawTerms := some (Proof.Events165.exact42302RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42301.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44731.bound, LeftBound42301.bound]
def bound : CoeffClass := .finite ⟨7751615201839287181312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44731.bound, LeftBound42301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44731.actual selector witness, LeftBound42301.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44736

namespace LeftBound44740
def owner : Owner := ⟨.program ⟨214⟩, ⟨27679⟩⟩
def transferEvent : Nat := 44740
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44738 .coefficient, .predecessor 1 44739 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44738 .coefficient)
      LeftBound44735.bound (LeftBound44735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44739 .coefficient)
      LeftBound41816.bound (LeftBound41816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41816.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44735.bound, LeftBound41816.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44735.bound, LeftBound41816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44735.actual selector witness, LeftBound41816.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44740

namespace LeftBound44741
def owner : Owner := ⟨.program ⟨214⟩, ⟨27679⟩⟩
def transferEvent : Nat := 44741
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44737 .summary, .result 41820 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44737 .summary)
      LeftBound44736.bound (LeftBound44736.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27462⟩⟩) (rawTerms := some (Proof.Events174.exact44737RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41820 .summary)
      LeftBound41819.bound (LeftBound41819.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27678⟩⟩) (rawTerms := some (Proof.Events163.exact41820RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41819.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44736.bound, LeftBound41819.bound]
def bound : CoeffClass := .finite ⟨9043661263333852925952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44736.bound, LeftBound41819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44736.actual selector witness, LeftBound41819.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44741

namespace LeftBound44745
def owner : Owner := ⟨.program ⟨214⟩, ⟨27896⟩⟩
def transferEvent : Nat := 44745
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44743 .coefficient, .predecessor 1 44744 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44743 .coefficient)
      LeftBound44740.bound (LeftBound44740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44740.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44740.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44744 .coefficient)
      LeftBound41334.bound (LeftBound41334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41334.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44740.bound, LeftBound41334.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44740.bound, LeftBound41334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44740.actual selector witness, LeftBound41334.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44745

namespace LeftBound44746
def owner : Owner := ⟨.program ⟨214⟩, ⟨27896⟩⟩
def transferEvent : Nat := 44746
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44742 .summary, .result 41338 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44742 .summary)
      LeftBound44741.bound (LeftBound44741.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27679⟩⟩) (rawTerms := some (Proof.Events174.exact44742RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41338 .summary)
      LeftBound41337.bound (LeftBound41337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27895⟩⟩) (rawTerms := some (Proof.Events161.exact41338RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41337.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44741.bound, LeftBound41337.bound]
def bound : CoeffClass := .finite ⟨10335729737273439256576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44741.bound, LeftBound41337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44741.actual selector witness, LeftBound41337.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44746

namespace LeftBound44750
def owner : Owner := ⟨.program ⟨214⟩, ⟨28113⟩⟩
def transferEvent : Nat := 44750
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44748 .coefficient, .predecessor 1 44749 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44748 .coefficient)
      LeftBound44745.bound (LeftBound44745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44749 .coefficient)
      LeftBound40852.bound (LeftBound40852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40852.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44745.bound, LeftBound40852.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44745.bound, LeftBound40852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44745.actual selector witness, LeftBound40852.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44750

namespace LeftBound44751
def owner : Owner := ⟨.program ⟨214⟩, ⟨28113⟩⟩
def transferEvent : Nat := 44751
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44747 .summary, .result 40856 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44747 .summary)
      LeftBound44746.bound (LeftBound44746.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27896⟩⟩) (rawTerms := some (Proof.Events174.exact44747RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40856 .summary)
      LeftBound40855.bound (LeftBound40855.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28112⟩⟩) (rawTerms := some (Proof.Events159.exact40856RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44746.bound, LeftBound40855.bound]
def bound : CoeffClass := .finite ⟨11627843036103066759168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44746.bound, LeftBound40855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44746.actual selector witness, LeftBound40855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44751

namespace LeftBound44755
def owner : Owner := ⟨.program ⟨214⟩, ⟨28330⟩⟩
def transferEvent : Nat := 44755
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44753 .coefficient, .predecessor 1 44754 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44753 .coefficient)
      LeftBound44750.bound (LeftBound44750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44754 .coefficient)
      LeftBound40370.bound (LeftBound40370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44750.bound, LeftBound40370.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44750.bound, LeftBound40370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44750.actual selector witness, LeftBound40370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44755

namespace LeftBound44756
def owner : Owner := ⟨.program ⟨214⟩, ⟨28330⟩⟩
def transferEvent : Nat := 44756
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44752 .summary, .result 40374 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44752 .summary)
      LeftBound44751.bound (LeftBound44751.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28113⟩⟩) (rawTerms := some (Proof.Events174.exact44752RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40374 .summary)
      LeftBound40373.bound (LeftBound40373.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28329⟩⟩) (rawTerms := some (Proof.Events157.exact40374RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40373.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44751.bound, LeftBound40373.bound]
def bound : CoeffClass := .finite ⟨12920023572267756019712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44751.bound, LeftBound40373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44751.actual selector witness, LeftBound40373.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44756

namespace LeftBound44760
def owner : Owner := ⟨.program ⟨214⟩, ⟨28547⟩⟩
def transferEvent : Nat := 44760
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44758 .coefficient, .predecessor 1 44759 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44758 .coefficient)
      LeftBound44755.bound (LeftBound44755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44759 .coefficient)
      LeftBound39888.bound (LeftBound39888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39888.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44755.bound, LeftBound39888.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44755.bound, LeftBound39888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44755.actual selector witness, LeftBound39888.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44760

namespace LeftBound44761
def owner : Owner := ⟨.program ⟨214⟩, ⟨28547⟩⟩
def transferEvent : Nat := 44761
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44757 .summary, .result 39892 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44757 .summary)
      LeftBound44756.bound (LeftBound44756.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28330⟩⟩) (rawTerms := some (Proof.Events174.exact44757RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39892 .summary)
      LeftBound39891.bound (LeftBound39891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28546⟩⟩) (rawTerms := some (Proof.Events155.exact39892RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44756.bound, LeftBound39891.bound]
def bound : CoeffClass := .finite ⟨14212226520877465866240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44756.bound, LeftBound39891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44756.actual selector witness, LeftBound39891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44761

namespace LeftBound44765
def owner : Owner := ⟨.program ⟨214⟩, ⟨28764⟩⟩
def transferEvent : Nat := 44765
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44763 .coefficient, .predecessor 1 44764 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44763 .coefficient)
      LeftBound44760.bound (LeftBound44760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44764 .coefficient)
      LeftBound39406.bound (LeftBound39406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44760.bound, LeftBound39406.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44760.bound, LeftBound39406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44760.actual selector witness, LeftBound39406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44765

namespace LeftBound44766
def owner : Owner := ⟨.program ⟨214⟩, ⟨28764⟩⟩
def transferEvent : Nat := 44766
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44762 .summary, .result 39410 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44762 .summary)
      LeftBound44761.bound (LeftBound44761.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28547⟩⟩) (rawTerms := some (Proof.Events174.exact44762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39410 .summary)
      LeftBound39409.bound (LeftBound39409.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28763⟩⟩) (rawTerms := some (Proof.Events153.exact39410RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39409.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44761.bound, LeftBound39409.bound]
def bound : CoeffClass := .finite ⟨15504496706822237470720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44761.bound, LeftBound39409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44761.actual selector witness, LeftBound39409.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44766

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
