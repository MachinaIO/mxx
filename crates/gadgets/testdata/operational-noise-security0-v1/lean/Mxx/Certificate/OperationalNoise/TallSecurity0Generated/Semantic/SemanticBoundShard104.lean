import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard101
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard102
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard103

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound16859
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def transferEvent : Nat := 16859
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16857 .coefficient, .predecessor 1 16858 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16857 .coefficient)
      LeftBound16855.bound (LeftBound16855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16858 .coefficient)
      LeftAuthority16803.bound (LeftAuthority16803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16803.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16855.bound, LeftAuthority16803.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16855.bound, LeftAuthority16803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16855.actual selector witness, LeftAuthority16803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16859

namespace LeftBound16863
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def transferEvent : Nat := 16863
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16861 .coefficient, .predecessor 1 16862 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16861 .coefficient)
      LeftBound16859.bound (LeftBound16859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16862 .coefficient)
      LeftAuthority16800.bound (LeftAuthority16800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16859.bound, LeftAuthority16800.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16859.bound, LeftAuthority16800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16859.actual selector witness, LeftAuthority16800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16863

namespace LeftBound16867
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def transferEvent : Nat := 16867
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16865 .coefficient, .predecessor 1 16866 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16865 .coefficient)
      LeftBound16863.bound (LeftBound16863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16866 .coefficient)
      LeftAuthority16797.bound (LeftAuthority16797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16797.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16797.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16863.bound, LeftAuthority16797.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16863.bound, LeftAuthority16797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16863.actual selector witness, LeftAuthority16797.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16867

namespace LeftBound16871
def owner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def transferEvent : Nat := 16871
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16869 .coefficient, .predecessor 1 16870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16869 .coefficient)
      LeftBound16867.bound (LeftBound16867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16870 .coefficient)
      LeftAuthority16794.bound (LeftAuthority16794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16794.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16867.bound, LeftAuthority16794.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16867.bound, LeftAuthority16794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16867.actual selector witness, LeftAuthority16794.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16871

namespace LeftBound16875
def owner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def transferEvent : Nat := 16875
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16873 .coefficient, .predecessor 1 16874 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16873 .coefficient)
      LeftBound16871.bound (LeftBound16871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16874 .coefficient)
      LeftAuthority16791.bound (LeftAuthority16791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16871.bound, LeftAuthority16791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16871.bound, LeftAuthority16791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16871.actual selector witness, LeftAuthority16791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16875

namespace LeftBound16879
def owner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def transferEvent : Nat := 16879
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16877 .coefficient, .predecessor 1 16878 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16877 .coefficient)
      LeftBound16875.bound (LeftBound16875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16878 .coefficient)
      LeftAuthority16788.bound (LeftAuthority16788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16788.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16788.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16875.bound, LeftAuthority16788.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16875.bound, LeftAuthority16788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16875.actual selector witness, LeftAuthority16788.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16879

namespace LeftBound16883
def owner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def transferEvent : Nat := 16883
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16881 .coefficient, .predecessor 1 16882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16881 .coefficient)
      LeftBound16879.bound (LeftBound16879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16882 .coefficient)
      LeftAuthority16785.bound (LeftAuthority16785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16785.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16785.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16879.bound, LeftAuthority16785.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16879.bound, LeftAuthority16785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16879.actual selector witness, LeftAuthority16785.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16883

namespace LeftBound16887
def owner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def transferEvent : Nat := 16887
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16885 .coefficient, .predecessor 1 16886 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16885 .coefficient)
      LeftBound16883.bound (LeftBound16883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16886 .coefficient)
      LeftAuthority16782.bound (LeftAuthority16782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16782.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16883.bound, LeftAuthority16782.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16883.bound, LeftAuthority16782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16883.actual selector witness, LeftAuthority16782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16887

namespace LeftBound16891
def owner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def transferEvent : Nat := 16891
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16889 .coefficient, .predecessor 1 16890 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16889 .coefficient)
      LeftBound16887.bound (LeftBound16887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16890 .coefficient)
      LeftAuthority16779.bound (LeftAuthority16779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16779.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16887.bound, LeftAuthority16779.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16887.bound, LeftAuthority16779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16887.actual selector witness, LeftAuthority16779.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16891

namespace LeftBound16895
def owner : Owner := ⟨.program ⟨214⟩, ⟨6811⟩⟩
def transferEvent : Nat := 16895
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16893 .coefficient, .predecessor 1 16894 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16893 .coefficient)
      LeftBound16891.bound (LeftBound16891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16894 .coefficient)
      LeftAuthority16776.bound (LeftAuthority16776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16776.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16891.bound, LeftAuthority16776.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16891.bound, LeftAuthority16776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16891.actual selector witness, LeftAuthority16776.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16895

namespace LeftBound16899
def owner : Owner := ⟨.program ⟨214⟩, ⟨18666⟩⟩
def transferEvent : Nat := 16899
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16897 .coefficient, .predecessor 1 16898 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16897 .coefficient)
      LeftBound16895.bound (LeftBound16895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16898 .coefficient)
      LeftBound16755.bound (LeftBound16755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16895.bound, LeftBound16755.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16895.bound, LeftBound16755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16895.actual selector witness, LeftBound16755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16899

namespace LeftBound16903
def owner : Owner := ⟨.program ⟨214⟩, ⟨18694⟩⟩
def transferEvent : Nat := 16903
def frameStart : Nat := 16225
def rule : BoundRule := .product (.predecessor 0 16901 .coefficient) (.predecessor 1 16902 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16901 .coefficient)
      LeftBound16899.bound (LeftBound16899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16902 .coefficient)
      LeftAuthority16740.bound (LeftAuthority16740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16740.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16740.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound16899.bound LeftAuthority16740.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16899.bound, LeftAuthority16740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound16899.actual selector witness) * (LeftAuthority16740.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound16903

namespace LeftBound16982
def owner : Owner := ⟨.program ⟨214⟩, ⟨18513⟩⟩
def transferEvent : Nat := 16982
def frameStart : Nat := 16225
def rule : BoundRule := .product (.predecessor 0 16980 .coefficient) (.predecessor 1 16981 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16980 .coefficient)
      LeftAuthority16751.bound (LeftAuthority16751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16751.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16981 .coefficient)
      LeftAuthority16978.bound (LeftAuthority16978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16978.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority16751.bound LeftAuthority16978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16751.bound, LeftAuthority16978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority16751.actual selector witness) * (LeftAuthority16978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound16982

namespace LeftBound16990
def owner : Owner := ⟨.program ⟨214⟩, ⟨18514⟩⟩
def transferEvent : Nat := 16990
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16988 .coefficient, .predecessor 1 16989 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16988 .coefficient)
      LeftAuthority16986.bound (LeftAuthority16986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16989 .coefficient)
      LeftBound16982.bound (LeftBound16982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority16986.bound, LeftBound16982.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16986.bound, LeftBound16982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority16986.actual selector witness, LeftBound16982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16990

namespace LeftBound16994
def owner : Owner := ⟨.program ⟨214⟩, ⟨18695⟩⟩
def transferEvent : Nat := 16994
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16992 .coefficient, .predecessor 1 16993 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16992 .coefficient)
      LeftBound16990.bound (LeftBound16990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16993 .coefficient)
      LeftBound16903.bound (LeftBound16903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16903.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16903.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16990.bound, LeftBound16903.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16990.bound, LeftBound16903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16990.actual selector witness, LeftBound16903.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16994

namespace LeftBound17041
def owner : Owner := ⟨.program ⟨214⟩, ⟨30211⟩⟩
def transferEvent : Nat := 17041
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 17039 .coefficient, .predecessor 1 17040 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17039 .coefficient)
      LeftBound15632.bound (LeftBound15632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15632.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17040 .coefficient)
      LeftBound15547.bound (LeftBound15547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15632.bound, LeftBound15547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15632.bound, LeftBound15547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15632.actual selector witness, LeftBound15547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17041

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
