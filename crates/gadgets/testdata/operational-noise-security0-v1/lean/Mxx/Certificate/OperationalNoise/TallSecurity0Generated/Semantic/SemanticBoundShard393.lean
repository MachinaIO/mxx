import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard392

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound57874
def owner : Owner := ⟨.program ⟨214⟩, ⟨27016⟩⟩
def transferEvent : Nat := 57874
def frameStart : Nat := 57774
def rule : BoundRule := .sum [.predecessor 0 57872 .coefficient, .predecessor 1 57873 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57872 .coefficient)
      LeftBound57870.bound (LeftBound57870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57873 .coefficient)
      LeftBound57851.bound (LeftBound57851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57851.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57870.bound, LeftBound57851.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57870.bound, LeftBound57851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57870.actual selector witness, LeftBound57851.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57874

namespace LeftBound57887
def owner : Owner := ⟨.program ⟨214⟩, ⟨27014⟩⟩
def transferEvent : Nat := 57887
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57885 .coefficient, .predecessor 1 57886 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57885 .coefficient)
      LeftBound57716.bound (LeftBound57716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57886 .coefficient)
      LeftBound57699.bound (LeftBound57699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57699.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57716.bound, LeftBound57699.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57716.bound, LeftBound57699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57716.actual selector witness, LeftBound57699.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57887

namespace LeftBound57890
def owner : Owner := ⟨.program ⟨214⟩, ⟨27014⟩⟩
def transferEvent : Nat := 57890
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 57884 .summary, .result 57706 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57884 .summary)
      LeftBound57718.bound (LeftBound57718.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20831⟩⟩) (rawTerms := some (Proof.Events226.exact57884RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57706 .summary)
      LeftBound57701.bound (LeftBound57701.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27013⟩⟩) (rawTerms := some (Proof.Events225.exact57706RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57701.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57718.bound, LeftBound57701.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57718.bound, LeftBound57701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57718.actual selector witness, LeftBound57701.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57890

namespace LeftBound57914
def owner : Owner := ⟨.program ⟨214⟩, ⟨10988⟩⟩
def transferEvent : Nat := 57914
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 57912 .coefficient) (.predecessor 1 57913 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57912 .coefficient)
      LeftAuthority2682.bound (LeftAuthority2682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57913 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2682.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2682.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2682.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound57914

namespace LeftBound57919
def owner : Owner := ⟨.program ⟨214⟩, ⟨7268⟩⟩
def transferEvent : Nat := 57919
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57917 .coefficient) (.predecessor 1 57918 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57917 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57918 .coefficient)
      LeftBound13986.bound (LeftBound13986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound13986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound13986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound13986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57919

namespace LeftBound57924
def owner : Owner := ⟨.program ⟨214⟩, ⟨10989⟩⟩
def transferEvent : Nat := 57924
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57922 .coefficient, .predecessor 1 57923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57922 .coefficient)
      LeftBound57919.bound (LeftBound57919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57923 .coefficient)
      LeftBound57914.bound (LeftBound57914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57919.bound, LeftBound57914.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57919.bound, LeftBound57914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57919.actual selector witness, LeftBound57914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57924

namespace LeftBound57928
def owner : Owner := ⟨.program ⟨214⟩, ⟨10990⟩⟩
def transferEvent : Nat := 57928
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57926 .coefficient, .predecessor 1 57927 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57926 .coefficient)
      LeftBound57924.bound (LeftBound57924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57927 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57924.bound, LeftBound13978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57924.bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57924.actual selector witness, LeftBound13978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57928

namespace LeftBound57929
def owner : Owner := ⟨.program ⟨214⟩, ⟨10990⟩⟩
def transferEvent : Nat := 57929
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨88⟩⟩]⟩ [⟨.result 13979 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13979 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨88⟩⟩) (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13978.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13978.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57929

namespace LeftBound57934
def owner : Owner := ⟨.program ⟨214⟩, ⟨10991⟩⟩
def transferEvent : Nat := 57934
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57932 .coefficient) (.predecessor 1 57933 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57932 .coefficient)
      LeftBound57928.bound (LeftBound57928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57928.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57933 .coefficient)
      LeftAuthority2685.bound (LeftAuthority2685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2685.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound57928.bound LeftAuthority2685.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57928.bound, LeftAuthority2685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound57928.actual selector witness) * (LeftAuthority2685.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57934

namespace LeftBound57935
def owner : Owner := ⟨.program ⟨214⟩, ⟨10991⟩⟩
def transferEvent : Nat := 57935
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩], []⟩ [⟨.result 2686 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2686 .coefficient)
      LeftAuthority2685.bound (LeftAuthority2685.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10847⟩⟩) (rawTerms := some (Proof.Events010.exact2686RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2685.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2685.bound []
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2685.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57935

namespace LeftBound57936
def owner : Owner := ⟨.program ⟨214⟩, ⟨10991⟩⟩
def transferEvent : Nat := 57936
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 57931 .summary) (.transfer 57935) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57931 .summary)
      LeftBound57929.bound (LeftBound57929.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10990⟩⟩) (rawTerms := some (Proof.Events226.exact57931RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 57935)
      LeftBound57935.bound (LeftBound57935.actual selector witness) := by
  exact .transfer (LeftBound57935.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound57929.bound LeftBound57935.bound
def bound : CoeffClass := .finite ⟨3328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57929.bound, LeftBound57935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound57929.actual selector witness) * (LeftBound57935.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57936

namespace LeftBound57942
def owner : Owner := ⟨.program ⟨214⟩, ⟨10848⟩⟩
def transferEvent : Nat := 57942
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 57940 .coefficient) (.predecessor 1 57941 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57940 .coefficient)
      LeftAuthority2685.bound (LeftAuthority2685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2685.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57941 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2685.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2685.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2685.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound57942

namespace LeftBound57947
def owner : Owner := ⟨.program ⟨214⟩, ⟨7285⟩⟩
def transferEvent : Nat := 57947
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57945 .coefficient) (.predecessor 1 57946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57945 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57946 .coefficient)
      LeftBound14027.bound (LeftBound14027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound14027.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound14027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound14027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57947

namespace LeftBound57952
def owner : Owner := ⟨.program ⟨214⟩, ⟨10849⟩⟩
def transferEvent : Nat := 57952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57950 .coefficient, .predecessor 1 57951 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57950 .coefficient)
      LeftBound57947.bound (LeftBound57947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57951 .coefficient)
      LeftBound57942.bound (LeftBound57942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57942.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57947.bound, LeftBound57942.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57947.bound, LeftBound57942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57947.actual selector witness, LeftBound57942.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57952

namespace LeftBound57956
def owner : Owner := ⟨.program ⟨214⟩, ⟨10850⟩⟩
def transferEvent : Nat := 57956
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57954 .coefficient, .predecessor 1 57955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57954 .coefficient)
      LeftBound57952.bound (LeftBound57952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57955 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57952.bound, LeftBound14019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57952.bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57952.actual selector witness, LeftBound14019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57956

namespace LeftBound57957
def owner : Owner := ⟨.program ⟨214⟩, ⟨10850⟩⟩
def transferEvent : Nat := 57957
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨105⟩⟩]⟩ [⟨.result 14020 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14020 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨105⟩⟩) (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14019.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14019.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57957

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
