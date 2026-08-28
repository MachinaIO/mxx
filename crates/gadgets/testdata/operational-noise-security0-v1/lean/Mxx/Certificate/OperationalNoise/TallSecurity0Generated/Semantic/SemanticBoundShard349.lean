import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard040
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard348

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound52029
def owner : Owner := ⟨.program ⟨214⟩, ⟨16757⟩⟩
def transferEvent : Nat := 52029
def frameStart : Nat := 51990
def rule : BoundRule := .identity (.predecessor 0 52028 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52028 .coefficient)
      LeftAuthority52026.bound (LeftAuthority52026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52026.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52026.derived selector witness)

def rawBound : CoeffClass := LeftAuthority52026.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority52026.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52029

namespace LeftBound52046
def owner : Owner := ⟨.program ⟨214⟩, ⟨16831⟩⟩
def transferEvent : Nat := 52046
def frameStart : Nat := 51990
def rule : BoundRule := .sum [.predecessor 0 52044 .coefficient, .predecessor 1 52045 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52044 .coefficient)
      LeftBound52029.bound (LeftBound52029.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52045 .coefficient)
      LeftAuthority52042.bound (LeftAuthority52042.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority52042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52029.bound, LeftAuthority52042.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52029.bound, LeftAuthority52042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52029.actual selector witness, LeftAuthority52042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52046

namespace LeftBound52049
def owner : Owner := ⟨.program ⟨214⟩, ⟨16832⟩⟩
def transferEvent : Nat := 52049
def frameStart : Nat := 51990
def rule : BoundRule := .identity (.predecessor 0 52048 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52048 .coefficient)
      LeftBound52046.bound (LeftBound52046.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52046.derived selector witness)

def rawBound : CoeffClass := LeftBound52046.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound52046.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52049

namespace LeftBound52055
def owner : Owner := ⟨.program ⟨214⟩, ⟨16833⟩⟩
def transferEvent : Nat := 52055
def frameStart : Nat := 51990
def rule : BoundRule := .product (.predecessor 0 52053 .coefficient) (.predecessor 1 52054 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52053 .coefficient)
      LeftAuthority52051.bound (LeftAuthority52051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52051.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52054 .coefficient)
      LeftBound52049.bound (LeftBound52049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52049.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority52051.bound LeftBound52049.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52051.bound, LeftBound52049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority52051.actual selector witness) * (LeftBound52049.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52055

namespace LeftBound52063
def owner : Owner := ⟨.program ⟨214⟩, ⟨16834⟩⟩
def transferEvent : Nat := 52063
def frameStart : Nat := 51990
def rule : BoundRule := .sum [.predecessor 0 52061 .coefficient, .predecessor 1 52062 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52061 .coefficient)
      LeftAuthority52059.bound (LeftAuthority52059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52062 .coefficient)
      LeftBound52055.bound (LeftBound52055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52055.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority52059.bound, LeftBound52055.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52059.bound, LeftBound52055.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority52059.actual selector witness, LeftBound52055.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52063

namespace LeftBound52067
def owner : Owner := ⟨.program ⟨214⟩, ⟨29616⟩⟩
def transferEvent : Nat := 52067
def frameStart : Nat := 51990
def rule : BoundRule := .product (.predecessor 0 52065 .coefficient) (.predecessor 1 52066 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52065 .coefficient)
      LeftBound52063.bound (LeftBound52063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52066 .coefficient)
      LeftAuthority52040.bound (LeftAuthority52040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52040.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52063.bound LeftAuthority52040.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52063.bound, LeftAuthority52040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52063.actual selector witness) * (LeftAuthority52040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52067

namespace LeftBound52078
def owner : Owner := ⟨.program ⟨214⟩, ⟨16802⟩⟩
def transferEvent : Nat := 52078
def frameStart : Nat := 51990
def rule : BoundRule := .product (.predecessor 0 52076 .coefficient) (.predecessor 1 52077 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52076 .coefficient)
      LeftAuthority52051.bound (LeftAuthority52051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52051.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52077 .coefficient)
      LeftAuthority52074.bound (LeftAuthority52074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52074.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority52051.bound LeftAuthority52074.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52051.bound, LeftAuthority52074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority52051.actual selector witness) * (LeftAuthority52074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52078

namespace LeftBound52086
def owner : Owner := ⟨.program ⟨214⟩, ⟨16803⟩⟩
def transferEvent : Nat := 52086
def frameStart : Nat := 51990
def rule : BoundRule := .sum [.predecessor 0 52084 .coefficient, .predecessor 1 52085 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52084 .coefficient)
      LeftAuthority52082.bound (LeftAuthority52082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52085 .coefficient)
      LeftBound52078.bound (LeftBound52078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority52082.bound, LeftBound52078.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52082.bound, LeftBound52078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority52082.actual selector witness, LeftBound52078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52086

namespace LeftBound52090
def owner : Owner := ⟨.program ⟨214⟩, ⟨29620⟩⟩
def transferEvent : Nat := 52090
def frameStart : Nat := 51990
def rule : BoundRule := .sum [.predecessor 0 52088 .coefficient, .predecessor 1 52089 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52088 .coefficient)
      LeftBound52086.bound (LeftBound52086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52089 .coefficient)
      LeftBound52067.bound (LeftBound52067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52086.bound, LeftBound52067.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52086.bound, LeftBound52067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52086.actual selector witness, LeftBound52067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52090

namespace LeftBound52103
def owner : Owner := ⟨.program ⟨214⟩, ⟨29618⟩⟩
def transferEvent : Nat := 52103
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52101 .coefficient, .predecessor 1 52102 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52101 .coefficient)
      LeftBound51932.bound (LeftBound51932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51932.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52102 .coefficient)
      LeftBound51915.bound (LeftBound51915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51915.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51915.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51932.bound, LeftBound51915.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51932.bound, LeftBound51915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51932.actual selector witness, LeftBound51915.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52103

namespace LeftBound52106
def owner : Owner := ⟨.program ⟨214⟩, ⟨29618⟩⟩
def transferEvent : Nat := 52106
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 52100 .summary, .result 51922 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52100 .summary)
      LeftBound51934.bound (LeftBound51934.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22559⟩⟩) (rawTerms := some (Proof.Events203.exact52100RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51922 .summary)
      LeftBound51917.bound (LeftBound51917.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29617⟩⟩) (rawTerms := some (Proof.Events202.exact51922RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51934.bound, LeftBound51917.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51934.bound, LeftBound51917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51934.actual selector witness, LeftBound51917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52106

namespace LeftBound52130
def owner : Owner := ⟨.program ⟨214⟩, ⟨12773⟩⟩
def transferEvent : Nat := 52130
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 52128 .coefficient) (.predecessor 1 52129 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52128 .coefficient)
      LeftAuthority2406.bound (LeftAuthority2406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52129 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2406.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2406.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2406.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52130

namespace LeftBound52135
def owner : Owner := ⟨.program ⟨214⟩, ⟨7281⟩⟩
def transferEvent : Nat := 52135
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52133 .coefficient) (.predecessor 1 52134 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52133 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52134 .coefficient)
      LeftBound7974.bound (LeftBound7974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7974.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound7974.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound7974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound7974.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52135

namespace LeftBound52140
def owner : Owner := ⟨.program ⟨214⟩, ⟨12774⟩⟩
def transferEvent : Nat := 52140
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52138 .coefficient, .predecessor 1 52139 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52138 .coefficient)
      LeftBound52135.bound (LeftBound52135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52139 .coefficient)
      LeftBound52130.bound (LeftBound52130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52135.bound, LeftBound52130.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52135.bound, LeftBound52130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52135.actual selector witness, LeftBound52130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52140

namespace LeftBound52144
def owner : Owner := ⟨.program ⟨214⟩, ⟨12775⟩⟩
def transferEvent : Nat := 52144
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52142 .coefficient, .predecessor 1 52143 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52142 .coefficient)
      LeftBound52140.bound (LeftBound52140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52143 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52140.bound, LeftBound7966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52140.bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52140.actual selector witness, LeftBound7966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52144

namespace LeftBound52145
def owner : Owner := ⟨.program ⟨214⟩, ⟨12775⟩⟩
def transferEvent : Nat := 52145
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨101⟩⟩]⟩ [⟨.result 7967 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7967 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨101⟩⟩) (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7966.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7966.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound52145

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
