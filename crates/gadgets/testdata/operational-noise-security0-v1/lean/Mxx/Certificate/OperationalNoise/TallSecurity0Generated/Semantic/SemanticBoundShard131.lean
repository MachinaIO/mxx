import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard105
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard106
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard107
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard110
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard111
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard130

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21003
def owner : Owner := ⟨.program ⟨214⟩, ⟨29218⟩⟩
def transferEvent : Nat := 21003
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21001 .coefficient, .predecessor 1 21002 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21001 .coefficient)
      LeftBound20998.bound (LeftBound20998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21002 .coefficient)
      LeftBound18142.bound (LeftBound18142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20998.bound, LeftBound18142.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20998.bound, LeftBound18142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20998.actual selector witness, LeftBound18142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21003

namespace LeftBound21004
def owner : Owner := ⟨.program ⟨214⟩, ⟨29218⟩⟩
def transferEvent : Nat := 21004
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21000 .summary, .result 18149 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21000 .summary)
      LeftBound20999.bound (LeftBound20999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29001⟩⟩) (rawTerms := some (Proof.Events082.exact21000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18149 .summary)
      LeftBound18144.bound (LeftBound18144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29217⟩⟩) (rawTerms := some (Proof.Events070.exact18149RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18144.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20999.bound, LeftBound18144.bound]
def bound : CoeffClass := .finite ⟨66387466773624617837369753652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20999.bound, LeftBound18144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20999.actual selector witness, LeftBound18144.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21004

namespace LeftBound21008
def owner : Owner := ⟨.program ⟨214⟩, ⟨29435⟩⟩
def transferEvent : Nat := 21008
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21006 .coefficient, .predecessor 1 21007 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21006 .coefficient)
      LeftBound21003.bound (LeftBound21003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21007 .coefficient)
      LeftBound17930.bound (LeftBound17930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact17937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21003.bound, LeftBound17930.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21003.bound, LeftBound17930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21003.actual selector witness, LeftBound17930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21008

namespace LeftBound21009
def owner : Owner := ⟨.program ⟨214⟩, ⟨29435⟩⟩
def transferEvent : Nat := 21009
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21005 .summary, .result 17937 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21005 .summary)
      LeftBound21004.bound (LeftBound21004.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29218⟩⟩) (rawTerms := some (Proof.Events082.exact21005RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17937 .summary)
      LeftBound17932.bound (LeftBound17932.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29434⟩⟩) (rawTerms := some (Proof.Events070.exact17937RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17932.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21004.bound, LeftBound17932.bound]
def bound : CoeffClass := .finite ⟨71130530302524028096610304052, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21004.bound, LeftBound17932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21004.actual selector witness, LeftBound17932.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21009

namespace LeftBound21013
def owner : Owner := ⟨.program ⟨214⟩, ⟨29652⟩⟩
def transferEvent : Nat := 21013
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21011 .coefficient, .predecessor 1 21012 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21011 .coefficient)
      LeftBound21008.bound (LeftBound21008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21008.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21012 .coefficient)
      LeftBound17718.bound (LeftBound17718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21008.bound, LeftBound17718.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21008.bound, LeftBound17718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21008.actual selector witness, LeftBound17718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21013

namespace LeftBound21014
def owner : Owner := ⟨.program ⟨214⟩, ⟨29652⟩⟩
def transferEvent : Nat := 21014
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21010 .summary, .result 17725 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21010 .summary)
      LeftBound21009.bound (LeftBound21009.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29435⟩⟩) (rawTerms := some (Proof.Events082.exact21010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17725 .summary)
      LeftBound17720.bound (LeftBound17720.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29651⟩⟩) (rawTerms := some (Proof.Events069.exact17725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21009.bound, LeftBound17720.bound]
def bound : CoeffClass := .finite ⟨75873840593518912368522821684, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21009.bound, LeftBound17720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21009.actual selector witness, LeftBound17720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21014

namespace LeftBound21018
def owner : Owner := ⟨.program ⟨214⟩, ⟨29869⟩⟩
def transferEvent : Nat := 21018
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21016 .coefficient, .predecessor 1 21017 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21016 .coefficient)
      LeftBound21013.bound (LeftBound21013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21013.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21017 .coefficient)
      LeftBound17506.bound (LeftBound17506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21013.bound, LeftBound17506.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21013.bound, LeftBound17506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21013.actual selector witness, LeftBound17506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21018

namespace LeftBound21019
def owner : Owner := ⟨.program ⟨214⟩, ⟨29869⟩⟩
def transferEvent : Nat := 21019
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21015 .summary, .result 17513 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21015 .summary)
      LeftBound21014.bound (LeftBound21014.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29652⟩⟩) (rawTerms := some (Proof.Events082.exact21015RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17513 .summary)
      LeftBound17508.bound (LeftBound17508.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29868⟩⟩) (rawTerms := some (Proof.Events068.exact17513RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21014.bound, LeftBound17508.bound]
def bound : CoeffClass := .finite ⟨80617397646609270653107306548, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21014.bound, LeftBound17508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21014.actual selector witness, LeftBound17508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21019

namespace LeftBound21023
def owner : Owner := ⟨.program ⟨214⟩, ⟨30203⟩⟩
def transferEvent : Nat := 21023
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21021 .coefficient, .predecessor 1 21022 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21021 .coefficient)
      LeftBound21018.bound (LeftBound21018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21022 .coefficient)
      LeftBound17294.bound (LeftBound17294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17294.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21018.bound, LeftBound17294.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21018.bound, LeftBound17294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21018.actual selector witness, LeftBound17294.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21023

namespace LeftBound21024
def owner : Owner := ⟨.program ⟨214⟩, ⟨30203⟩⟩
def transferEvent : Nat := 21024
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21020 .summary, .result 17301 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21020 .summary)
      LeftBound21019.bound (LeftBound21019.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29869⟩⟩) (rawTerms := some (Proof.Events082.exact21020RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17301 .summary)
      LeftBound17296.bound (LeftBound17296.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30202⟩⟩) (rawTerms := some (Proof.Events067.exact17301RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17296.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21019.bound, LeftBound17296.bound]
def bound : CoeffClass := .finite ⟨85361036953731453608582447156, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21019.bound, LeftBound17296.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21019.actual selector witness, LeftBound17296.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21024

namespace LeftBound21028
def owner : Owner := ⟨.program ⟨214⟩, ⟨30214⟩⟩
def transferEvent : Nat := 21028
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21026 .coefficient, .predecessor 1 21027 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21026 .coefficient)
      LeftBound21023.bound (LeftBound21023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21027 .coefficient)
      LeftBound17082.bound (LeftBound17082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17082.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17082.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21023.bound, LeftBound17082.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21023.bound, LeftBound17082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21023.actual selector witness, LeftBound17082.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21028

namespace LeftBound21029
def owner : Owner := ⟨.program ⟨214⟩, ⟨30214⟩⟩
def transferEvent : Nat := 21029
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21025 .summary, .result 17089 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21025 .summary)
      LeftBound21024.bound (LeftBound21024.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30203⟩⟩) (rawTerms := some (Proof.Events082.exact21025RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17089 .summary)
      LeftBound17084.bound (LeftBound17084.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30212⟩⟩) (rawTerms := some (Proof.Events066.exact17089RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17084.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21024.bound, LeftBound17084.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789388161076, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21024.bound, LeftBound17084.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21024.actual selector witness, LeftBound17084.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21029

namespace LeftBound21035
def owner : Owner := ⟨.program ⟨214⟩, ⟨7089⟩⟩
def transferEvent : Nat := 21035
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21033 .coefficient) (.predecessor 1 21034 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21033 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21034 .coefficient)
      LeftAuthority5963.bound (LeftAuthority5963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftAuthority5963.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority5963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftAuthority5963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21035

namespace LeftBound21040
def owner : Owner := ⟨.program ⟨214⟩, ⟨7719⟩⟩
def transferEvent : Nat := 21040
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21038 .coefficient, .predecessor 1 21039 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21038 .coefficient)
      LeftBound21035.bound (LeftBound21035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21039 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21035.bound, LeftBound6447.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21035.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21035.actual selector witness, LeftBound6447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21040

namespace LeftBound21044
def owner : Owner := ⟨.program ⟨214⟩, ⟨7720⟩⟩
def transferEvent : Nat := 21044
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21042 .coefficient, .predecessor 1 21043 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21042 .coefficient)
      LeftBound21040.bound (LeftBound21040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21043 .coefficient)
      LeftAuthority21031.bound (LeftAuthority21031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21040.bound, LeftAuthority21031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21040.bound, LeftAuthority21031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21040.actual selector witness, LeftAuthority21031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21044

namespace LeftBound21045
def owner : Owner := ⟨.program ⟨214⟩, ⟨7720⟩⟩
def transferEvent : Nat := 21045
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨4⟩⟩]⟩ [⟨.result 21032 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21032 .coefficient)
      LeftAuthority21031.bound (LeftAuthority21031.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨4⟩⟩) (rawTerms := some (Proof.Events082.exact21032RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21031.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21031.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21031.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21045

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
