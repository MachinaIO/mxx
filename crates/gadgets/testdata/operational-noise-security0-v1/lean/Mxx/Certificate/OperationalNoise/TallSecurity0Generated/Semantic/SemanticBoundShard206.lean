import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard138
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard203
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard205

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31519
def owner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def transferEvent : Nat := 31519
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31517 .coefficient, .predecessor 1 31518 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31517 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31518 .coefficient)
      LeftAuthority31407.bound (LeftAuthority31407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31407.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31515.bound, LeftAuthority31407.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31515.bound, LeftAuthority31407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31515.actual selector witness, LeftAuthority31407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31519

namespace LeftBound31523
def owner : Owner := ⟨.program ⟨214⟩, ⟨6811⟩⟩
def transferEvent : Nat := 31523
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31521 .coefficient, .predecessor 1 31522 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31521 .coefficient)
      LeftBound31519.bound (LeftBound31519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31522 .coefficient)
      LeftAuthority31404.bound (LeftAuthority31404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31404.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31519.bound, LeftAuthority31404.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31519.bound, LeftAuthority31404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31519.actual selector witness, LeftAuthority31404.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31523

namespace LeftBound31527
def owner : Owner := ⟨.program ⟨214⟩, ⟨18662⟩⟩
def transferEvent : Nat := 31527
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31525 .coefficient, .predecessor 1 31526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31525 .coefficient)
      LeftBound31523.bound (LeftBound31523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31526 .coefficient)
      LeftBound31383.bound (LeftBound31383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31383.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31383.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31523.bound, LeftBound31383.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31523.bound, LeftBound31383.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31523.actual selector witness, LeftBound31383.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31527

namespace LeftBound31531
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def transferEvent : Nat := 31531
def frameStart : Nat := 30853
def rule : BoundRule := .product (.predecessor 0 31529 .coefficient) (.predecessor 1 31530 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31529 .coefficient)
      LeftBound31527.bound (LeftBound31527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31530 .coefficient)
      LeftAuthority31368.bound (LeftAuthority31368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31368.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound31527.bound LeftAuthority31368.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31527.bound, LeftAuthority31368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound31527.actual selector witness) * (LeftAuthority31368.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31531

namespace LeftBound31610
def owner : Owner := ⟨.program ⟨214⟩, ⟨18509⟩⟩
def transferEvent : Nat := 31610
def frameStart : Nat := 30853
def rule : BoundRule := .product (.predecessor 0 31608 .coefficient) (.predecessor 1 31609 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31608 .coefficient)
      LeftAuthority31379.bound (LeftAuthority31379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31379.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31609 .coefficient)
      LeftAuthority31606.bound (LeftAuthority31606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31606.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31606.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority31379.bound LeftAuthority31606.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31379.bound, LeftAuthority31606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority31379.actual selector witness) * (LeftAuthority31606.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31610

namespace LeftBound31618
def owner : Owner := ⟨.program ⟨214⟩, ⟨18510⟩⟩
def transferEvent : Nat := 31618
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31616 .coefficient, .predecessor 1 31617 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31616 .coefficient)
      LeftAuthority31614.bound (LeftAuthority31614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31617 .coefficient)
      LeftBound31610.bound (LeftBound31610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31610.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31614.bound, LeftBound31610.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31614.bound, LeftBound31610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority31614.actual selector witness, LeftBound31610.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31618

namespace LeftBound31622
def owner : Owner := ⟨.program ⟨214⟩, ⟨18692⟩⟩
def transferEvent : Nat := 31622
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31620 .coefficient, .predecessor 1 31621 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31620 .coefficient)
      LeftBound31618.bound (LeftBound31618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31621 .coefficient)
      LeftBound31531.bound (LeftBound31531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31618.bound, LeftBound31531.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31618.bound, LeftBound31531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31618.actual selector witness, LeftBound31531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31622

namespace LeftBound31669
def owner : Owner := ⟨.program ⟨214⟩, ⟨30189⟩⟩
def transferEvent : Nat := 31669
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31667 .coefficient, .predecessor 1 31668 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31667 .coefficient)
      LeftBound30260.bound (LeftBound30260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31668 .coefficient)
      LeftBound30175.bound (LeftBound30175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events118.exact30250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30260.bound, LeftBound30175.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30260.bound, LeftBound30175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30260.actual selector witness, LeftBound30175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31669

namespace LeftBound31706
def owner : Owner := ⟨.program ⟨214⟩, ⟨30189⟩⟩
def transferEvent : Nat := 31706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31666 .summary, .result 30250 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31666 .summary)
      LeftBound30262.bound (LeftBound30262.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18574⟩⟩) (rawTerms := some (Proof.Events123.exact31666RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30250 .summary)
      LeftBound30177.bound (LeftBound30177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30188⟩⟩) (rawTerms := some (Proof.Events118.exact30250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30262.bound, LeftBound30177.bound]
def bound : CoeffClass := .finite ⟨85361036953731455419885957120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30262.bound, LeftBound30177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30262.actual selector witness, LeftBound30177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31706

namespace LeftBound31710
def owner : Owner := ⟨.program ⟨214⟩, ⟨30190⟩⟩
def transferEvent : Nat := 31710
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31708 .coefficient) (.predecessor 1 31709 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31708 .coefficient)
      LeftBound31669.bound (LeftBound31669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31709 .coefficient)
      LeftBound5498.bound (LeftBound5498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound31669.bound LeftBound5498.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31669.bound, LeftBound5498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound31669.actual selector witness) * (LeftBound5498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31710

namespace LeftBound31711
def owner : Owner := ⟨.program ⟨214⟩, ⟨30190⟩⟩
def transferEvent : Nat := 31711
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩ [⟨.result 5495 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5495 .coefficient)
      LeftAuthority5494.bound (LeftAuthority5494.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6651⟩⟩) (rawTerms := some (Proof.Events021.exact5495RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5494.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5494.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5494.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31711

namespace LeftBound31712
def owner : Owner := ⟨.program ⟨214⟩, ⟨30190⟩⟩
def transferEvent : Nat := 31712
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 31707 .summary) (.transfer 31711) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31707 .summary)
      LeftBound31706.bound (LeftBound31706.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30189⟩⟩) (rawTerms := some (Proof.Events123.exact31707RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 31711)
      LeftBound31711.bound (LeftBound31711.actual selector witness) := by
  exact .transfer (LeftBound31711.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound31706.bound LeftBound31711.bound
def bound : CoeffClass := .finite ⟨313276371396785701094268180805713920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31706.bound, LeftBound31711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound31706.actual selector witness) * (LeftBound31711.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31712

namespace LeftBound31727
def owner : Owner := ⟨.program ⟨214⟩, ⟨30178⟩⟩
def transferEvent : Nat := 31727
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31725 .coefficient) (.predecessor 1 31726 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31725 .coefficient)
      LeftBound21694.bound (LeftBound21694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31726 .coefficient)
      LeftAuthority31723.bound (LeftAuthority31723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31723.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31723.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21694.bound LeftAuthority31723.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21694.bound, LeftAuthority31723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21694.actual selector witness) * (LeftAuthority31723.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31727

namespace LeftBound31728
def owner : Owner := ⟨.program ⟨214⟩, ⟨30178⟩⟩
def transferEvent : Nat := 31728
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30176⟩⟩]⟩ [⟨.result 31724 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31724 .coefficient)
      LeftAuthority31723.bound (LeftAuthority31723.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30176⟩⟩) (rawTerms := some (Proof.Events123.exact31724RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31723.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31723.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority31723.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority31723.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31728

namespace LeftBound31729
def owner : Owner := ⟨.program ⟨214⟩, ⟨30178⟩⟩
def transferEvent : Nat := 31729
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21698 .summary) (.transfer 31728) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21698 .summary)
      LeftBound21697.bound (LeftBound21697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25775⟩⟩) (rawTerms := some (Proof.Events084.exact21698RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 31728)
      LeftBound31728.bound (LeftBound31728.actual selector witness) := by
  exact .transfer (LeftBound31728.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21697.bound LeftBound31728.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21697.bound, LeftBound31728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21697.actual selector witness) * (LeftBound31728.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31729

namespace LeftBound31740
def owner : Owner := ⟨.program ⟨214⟩, ⟨22782⟩⟩
def transferEvent : Nat := 31740
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 31738 .coefficient) (.value (.predecessor 1 31739 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31738 .coefficient)
      LeftAuthority31736.bound (LeftAuthority31736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31736.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31739 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority31736.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31736.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority31736.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound31740

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
