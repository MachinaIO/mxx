import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard406

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound60564
def owner : Owner := ⟨.program ⟨214⟩, ⟨17341⟩⟩
def transferEvent : Nat := 60564
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60562 .coefficient, .predecessor 1 60563 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60562 .coefficient)
      LeftBound60560.bound (LeftBound60560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60560.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60563 .coefficient)
      LeftAuthority60375.bound (LeftAuthority60375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60375.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60375.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60560.bound, LeftAuthority60375.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60560.bound, LeftAuthority60375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60560.actual selector witness, LeftAuthority60375.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60564

namespace LeftBound60568
def owner : Owner := ⟨.program ⟨214⟩, ⟨17342⟩⟩
def transferEvent : Nat := 60568
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60566 .coefficient, .predecessor 1 60567 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60566 .coefficient)
      LeftBound60564.bound (LeftBound60564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60567 .coefficient)
      LeftAuthority60352.bound (LeftAuthority60352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60352.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60352.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60564.bound, LeftAuthority60352.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60564.bound, LeftAuthority60352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60564.actual selector witness, LeftAuthority60352.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60568

namespace LeftBound60572
def owner : Owner := ⟨.program ⟨214⟩, ⟨18354⟩⟩
def transferEvent : Nat := 60572
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60570 .coefficient, .predecessor 1 60571 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60570 .coefficient)
      LeftBound60568.bound (LeftBound60568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60571 .coefficient)
      LeftAuthority60329.bound (LeftAuthority60329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60329.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60329.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60568.bound, LeftAuthority60329.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60568.bound, LeftAuthority60329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60568.actual selector witness, LeftAuthority60329.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60572

namespace LeftBound60576
def owner : Owner := ⟨.program ⟨214⟩, ⟨18355⟩⟩
def transferEvent : Nat := 60576
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60574 .coefficient, .predecessor 1 60575 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60574 .coefficient)
      LeftBound60572.bound (LeftBound60572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60575 .coefficient)
      LeftAuthority60306.bound (LeftAuthority60306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60306.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60572.bound, LeftAuthority60306.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60572.bound, LeftAuthority60306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60572.actual selector witness, LeftAuthority60306.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60576

namespace LeftBound60580
def owner : Owner := ⟨.program ⟨214⟩, ⟨18356⟩⟩
def transferEvent : Nat := 60580
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60578 .coefficient, .predecessor 1 60579 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60578 .coefficient)
      LeftBound60576.bound (LeftBound60576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60579 .coefficient)
      LeftAuthority60283.bound (LeftAuthority60283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60283.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60576.bound, LeftAuthority60283.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60576.bound, LeftAuthority60283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60576.actual selector witness, LeftAuthority60283.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60580

namespace LeftBound60584
def owner : Owner := ⟨.program ⟨214⟩, ⟨18357⟩⟩
def transferEvent : Nat := 60584
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60582 .coefficient, .predecessor 1 60583 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60582 .coefficient)
      LeftBound60580.bound (LeftBound60580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60583 .coefficient)
      LeftAuthority60260.bound (LeftAuthority60260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60260.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60580.bound, LeftAuthority60260.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60580.bound, LeftAuthority60260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60580.actual selector witness, LeftAuthority60260.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60584

namespace LeftBound60588
def owner : Owner := ⟨.program ⟨214⟩, ⟨18358⟩⟩
def transferEvent : Nat := 60588
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60586 .coefficient, .predecessor 1 60587 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60586 .coefficient)
      LeftBound60584.bound (LeftBound60584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60587 .coefficient)
      LeftAuthority60237.bound (LeftAuthority60237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60237.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60584.bound, LeftAuthority60237.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60584.bound, LeftAuthority60237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60584.actual selector witness, LeftAuthority60237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60588

namespace LeftBound60592
def owner : Owner := ⟨.program ⟨214⟩, ⟨18359⟩⟩
def transferEvent : Nat := 60592
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60590 .coefficient, .predecessor 1 60591 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60590 .coefficient)
      LeftBound60588.bound (LeftBound60588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60591 .coefficient)
      LeftAuthority60214.bound (LeftAuthority60214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60214.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60214.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60588.bound, LeftAuthority60214.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60588.bound, LeftAuthority60214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60588.actual selector witness, LeftAuthority60214.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60592

namespace LeftBound60596
def owner : Owner := ⟨.program ⟨214⟩, ⟨18360⟩⟩
def transferEvent : Nat := 60596
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60594 .coefficient, .predecessor 1 60595 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60594 .coefficient)
      LeftBound60592.bound (LeftBound60592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60595 .coefficient)
      LeftAuthority60191.bound (LeftAuthority60191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60191.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60592.bound, LeftAuthority60191.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60592.bound, LeftAuthority60191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60592.actual selector witness, LeftAuthority60191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60596

namespace LeftBound60600
def owner : Owner := ⟨.program ⟨214⟩, ⟨18361⟩⟩
def transferEvent : Nat := 60600
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60598 .coefficient, .predecessor 1 60599 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60598 .coefficient)
      LeftBound60596.bound (LeftBound60596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60596.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60599 .coefficient)
      LeftAuthority60168.bound (LeftAuthority60168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60168.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60596.bound, LeftAuthority60168.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60596.bound, LeftAuthority60168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60596.actual selector witness, LeftAuthority60168.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60600

namespace LeftBound60604
def owner : Owner := ⟨.program ⟨214⟩, ⟨18362⟩⟩
def transferEvent : Nat := 60604
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60602 .coefficient, .predecessor 1 60603 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60602 .coefficient)
      LeftBound60600.bound (LeftBound60600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60600.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60603 .coefficient)
      LeftAuthority60145.bound (LeftAuthority60145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events234.exact60146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60145.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60600.bound, LeftAuthority60145.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60600.bound, LeftAuthority60145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60600.actual selector witness, LeftAuthority60145.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60604

namespace LeftBound60607
def owner : Owner := ⟨.program ⟨214⟩, ⟨18363⟩⟩
def transferEvent : Nat := 60607
def frameStart : Nat := 60103
def rule : BoundRule := .identity (.predecessor 0 60606 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60606 .coefficient)
      LeftBound60604.bound (LeftBound60604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60604.derived selector witness)

def rawBound : CoeffClass := LeftBound60604.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound60604.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound60607

namespace LeftBound60624
def owner : Owner := ⟨.program ⟨214⟩, ⟨18651⟩⟩
def transferEvent : Nat := 60624
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60622 .coefficient, .predecessor 1 60623 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60622 .coefficient)
      LeftBound60607.bound (LeftBound60607.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound60607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60623 .coefficient)
      LeftAuthority60620.bound (LeftAuthority60620.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority60620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60607.bound, LeftAuthority60620.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60607.bound, LeftAuthority60620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60607.actual selector witness, LeftAuthority60620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60624

namespace LeftBound60627
def owner : Owner := ⟨.program ⟨214⟩, ⟨18652⟩⟩
def transferEvent : Nat := 60627
def frameStart : Nat := 60103
def rule : BoundRule := .identity (.predecessor 0 60626 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60626 .coefficient)
      LeftBound60624.bound (LeftBound60624.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound60624.derived selector witness)

def rawBound : CoeffClass := LeftBound60624.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound60624.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound60627

namespace LeftBound60633
def owner : Owner := ⟨.program ⟨214⟩, ⟨18653⟩⟩
def transferEvent : Nat := 60633
def frameStart : Nat := 60103
def rule : BoundRule := .product (.predecessor 0 60631 .coefficient) (.predecessor 1 60632 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60631 .coefficient)
      LeftAuthority60629.bound (LeftAuthority60629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60629.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60632 .coefficient)
      LeftBound60627.bound (LeftBound60627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60627.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority60629.bound LeftBound60627.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority60629.bound, LeftBound60627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority60629.actual selector witness) * (LeftBound60627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60633

namespace LeftBound60709
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def transferEvent : Nat := 60709
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60707 .coefficient, .predecessor 1 60708 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60707 .coefficient)
      LeftAuthority60705.bound (LeftAuthority60705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60705.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60705.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60708 .coefficient)
      LeftAuthority60702.bound (LeftAuthority60702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60702.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority60705.bound, LeftAuthority60702.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority60705.bound, LeftAuthority60702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority60705.actual selector witness, LeftAuthority60702.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60709

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
