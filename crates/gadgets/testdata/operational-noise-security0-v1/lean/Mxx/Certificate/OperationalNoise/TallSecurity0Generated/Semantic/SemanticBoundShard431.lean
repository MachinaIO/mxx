import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard402
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard430

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64388
def owner : Owner := ⟨.program ⟨214⟩, ⟨20471⟩⟩
def transferEvent : Nat := 64388
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 64387) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64387)
      LeftBound64387.bound (LeftBound64387.actual selector witness) := by
  exact .transfer (LeftBound64387.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound64387.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound64387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound64387.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64388

namespace LeftBound64483
def owner : Owner := ⟨.program ⟨214⟩, ⟨14958⟩⟩
def transferEvent : Nat := 64483
def frameStart : Nat := 64444
def rule : BoundRule := .identity (.predecessor 0 64482 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64482 .coefficient)
      LeftAuthority64480.bound (LeftAuthority64480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64480.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64480.derived selector witness)

def rawBound : CoeffClass := LeftAuthority64480.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority64480.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64483

namespace LeftBound64500
def owner : Owner := ⟨.program ⟨214⟩, ⟨14997⟩⟩
def transferEvent : Nat := 64500
def frameStart : Nat := 64444
def rule : BoundRule := .sum [.predecessor 0 64498 .coefficient, .predecessor 1 64499 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64498 .coefficient)
      LeftBound64483.bound (LeftBound64483.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64499 .coefficient)
      LeftAuthority64496.bound (LeftAuthority64496.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority64496.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64483.bound, LeftAuthority64496.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64483.bound, LeftAuthority64496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64483.actual selector witness, LeftAuthority64496.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64500

namespace LeftBound64503
def owner : Owner := ⟨.program ⟨214⟩, ⟨14998⟩⟩
def transferEvent : Nat := 64503
def frameStart : Nat := 64444
def rule : BoundRule := .identity (.predecessor 0 64502 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64502 .coefficient)
      LeftBound64500.bound (LeftBound64500.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64500.derived selector witness)

def rawBound : CoeffClass := LeftBound64500.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound64500.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64503

namespace LeftBound64509
def owner : Owner := ⟨.program ⟨214⟩, ⟨14999⟩⟩
def transferEvent : Nat := 64509
def frameStart : Nat := 64444
def rule : BoundRule := .product (.predecessor 0 64507 .coefficient) (.predecessor 1 64508 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64507 .coefficient)
      LeftAuthority64505.bound (LeftAuthority64505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64508 .coefficient)
      LeftBound64503.bound (LeftBound64503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority64505.bound LeftBound64503.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64505.bound, LeftBound64503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority64505.actual selector witness) * (LeftBound64503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64509

namespace LeftBound64517
def owner : Owner := ⟨.program ⟨214⟩, ⟨15000⟩⟩
def transferEvent : Nat := 64517
def frameStart : Nat := 64444
def rule : BoundRule := .sum [.predecessor 0 64515 .coefficient, .predecessor 1 64516 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64515 .coefficient)
      LeftAuthority64513.bound (LeftAuthority64513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64516 .coefficient)
      LeftBound64509.bound (LeftBound64509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64509.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64513.bound, LeftBound64509.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64513.bound, LeftBound64509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority64513.actual selector witness, LeftBound64509.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64517

namespace LeftBound64521
def owner : Owner := ⟨.program ⟨214⟩, ⟨26571⟩⟩
def transferEvent : Nat := 64521
def frameStart : Nat := 64444
def rule : BoundRule := .product (.predecessor 0 64519 .coefficient) (.predecessor 1 64520 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64519 .coefficient)
      LeftBound64517.bound (LeftBound64517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64520 .coefficient)
      LeftAuthority64494.bound (LeftAuthority64494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64494.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64517.bound LeftAuthority64494.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64517.bound, LeftAuthority64494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64517.actual selector witness) * (LeftAuthority64494.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64521

namespace LeftBound64532
def owner : Owner := ⟨.program ⟨214⟩, ⟨15055⟩⟩
def transferEvent : Nat := 64532
def frameStart : Nat := 64444
def rule : BoundRule := .product (.predecessor 0 64530 .coefficient) (.predecessor 1 64531 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64530 .coefficient)
      LeftAuthority64505.bound (LeftAuthority64505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64531 .coefficient)
      LeftAuthority64528.bound (LeftAuthority64528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64528.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority64505.bound LeftAuthority64528.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64505.bound, LeftAuthority64528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority64505.actual selector witness) * (LeftAuthority64528.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64532

namespace LeftBound64540
def owner : Owner := ⟨.program ⟨214⟩, ⟨15056⟩⟩
def transferEvent : Nat := 64540
def frameStart : Nat := 64444
def rule : BoundRule := .sum [.predecessor 0 64538 .coefficient, .predecessor 1 64539 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64538 .coefficient)
      LeftAuthority64536.bound (LeftAuthority64536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64536.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64539 .coefficient)
      LeftBound64532.bound (LeftBound64532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64536.bound, LeftBound64532.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64536.bound, LeftBound64532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority64536.actual selector witness, LeftBound64532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64540

namespace LeftBound64544
def owner : Owner := ⟨.program ⟨214⟩, ⟨26576⟩⟩
def transferEvent : Nat := 64544
def frameStart : Nat := 64444
def rule : BoundRule := .sum [.predecessor 0 64542 .coefficient, .predecessor 1 64543 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64542 .coefficient)
      LeftBound64540.bound (LeftBound64540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64543 .coefficient)
      LeftBound64521.bound (LeftBound64521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64521.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64540.bound, LeftBound64521.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64540.bound, LeftBound64521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64540.actual selector witness, LeftBound64521.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64544

namespace LeftBound64557
def owner : Owner := ⟨.program ⟨214⟩, ⟨26573⟩⟩
def transferEvent : Nat := 64557
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64555 .coefficient, .predecessor 1 64556 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64555 .coefficient)
      LeftBound64386.bound (LeftBound64386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64556 .coefficient)
      LeftBound64369.bound (LeftBound64369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64369.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64386.bound, LeftBound64369.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64386.bound, LeftBound64369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64386.actual selector witness, LeftBound64369.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64557

namespace LeftBound64560
def owner : Owner := ⟨.program ⟨214⟩, ⟨26573⟩⟩
def transferEvent : Nat := 64560
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64554 .summary, .result 64376 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64554 .summary)
      LeftBound64388.bound (LeftBound64388.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20471⟩⟩) (rawTerms := some (Proof.Events252.exact64554RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64376 .summary)
      LeftBound64371.bound (LeftBound64371.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26572⟩⟩) (rawTerms := some (Proof.Events251.exact64376RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64371.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64388.bound, LeftBound64371.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64388.bound, LeftBound64371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64388.actual selector witness, LeftBound64371.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64560

namespace LeftBound64564
def owner : Owner := ⟨.program ⟨214⟩, ⟨26574⟩⟩
def transferEvent : Nat := 64564
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64562 .coefficient) (.predecessor 1 64563 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64562 .coefficient)
      LeftBound64557.bound (LeftBound64557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64557.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64563 .coefficient)
      LeftBound5838.bound (LeftBound5838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64557.bound LeftBound5838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64557.bound, LeftBound5838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64557.actual selector witness) * (LeftBound5838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64564

namespace LeftBound64565
def owner : Owner := ⟨.program ⟨214⟩, ⟨26574⟩⟩
def transferEvent : Nat := 64565
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩ [⟨.result 5835 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5835 .coefficient)
      LeftAuthority5834.bound (LeftAuthority5834.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6671⟩⟩) (rawTerms := some (Proof.Events022.exact5835RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5834.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5834.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5834.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64565

namespace LeftBound64566
def owner : Owner := ⟨.program ⟨214⟩, ⟨26574⟩⟩
def transferEvent : Nat := 64566
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64561 .summary) (.transfer 64565) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64561 .summary)
      LeftBound64560.bound (LeftBound64560.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26573⟩⟩) (rawTerms := some (Proof.Events252.exact64561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64560.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64565)
      LeftBound64565.bound (LeftBound64565.actual selector witness) := by
  exact .transfer (LeftBound64565.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64560.bound LeftBound64565.bound
def bound : CoeffClass := .finite ⟨4741295067215179835091451904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64560.bound, LeftBound64565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64560.actual selector witness) * (LeftBound64565.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64566

namespace LeftBound64581
def owner : Owner := ⟨.program ⟨214⟩, ⟨26365⟩⟩
def transferEvent : Nat := 64581
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64579 .coefficient) (.predecessor 1 64580 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64579 .coefficient)
      LeftBound59138.bound (LeftBound59138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64580 .coefficient)
      LeftAuthority64577.bound (LeftAuthority64577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64577.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59138.bound LeftAuthority64577.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59138.bound, LeftAuthority64577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59138.actual selector witness) * (LeftAuthority64577.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64581

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
