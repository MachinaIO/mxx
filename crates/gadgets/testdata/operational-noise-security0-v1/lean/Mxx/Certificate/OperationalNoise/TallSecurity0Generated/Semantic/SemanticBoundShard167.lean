import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard166

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound25510
def owner : Owner := ⟨.program ⟨214⟩, ⟨14763⟩⟩
def transferEvent : Nat := 25510
def frameStart : Nat := 25423
def rule : BoundRule := .sum [.predecessor 0 25508 .coefficient, .predecessor 1 25509 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25508 .coefficient)
      LeftBound25505.bound (LeftBound25505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25509 .coefficient)
      LeftBound25482.bound (LeftBound25482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25482.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25482.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25505.bound, LeftBound25482.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25505.bound, LeftBound25482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25505.actual selector witness, LeftBound25482.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25510

namespace LeftBound25514
def owner : Owner := ⟨.program ⟨214⟩, ⟨26238⟩⟩
def transferEvent : Nat := 25514
def frameStart : Nat := 25423
def rule : BoundRule := .product (.predecessor 0 25512 .coefficient) (.predecessor 1 25513 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25512 .coefficient)
      LeftBound25510.bound (LeftBound25510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25513 .coefficient)
      LeftAuthority25467.bound (LeftAuthority25467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25467.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25467.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25510.bound LeftAuthority25467.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25510.bound, LeftAuthority25467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25510.actual selector witness) * (LeftAuthority25467.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25514

namespace LeftBound25525
def owner : Owner := ⟨.program ⟨214⟩, ⟨16192⟩⟩
def transferEvent : Nat := 25525
def frameStart : Nat := 25423
def rule : BoundRule := .product (.predecessor 0 25523 .coefficient) (.predecessor 1 25524 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25523 .coefficient)
      LeftAuthority25478.bound (LeftAuthority25478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25524 .coefficient)
      LeftAuthority25521.bound (LeftAuthority25521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority25478.bound LeftAuthority25521.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25478.bound, LeftAuthority25521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority25478.actual selector witness) * (LeftAuthority25521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25525

namespace LeftBound25533
def owner : Owner := ⟨.program ⟨214⟩, ⟨16193⟩⟩
def transferEvent : Nat := 25533
def frameStart : Nat := 25423
def rule : BoundRule := .sum [.predecessor 0 25531 .coefficient, .predecessor 1 25532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25531 .coefficient)
      LeftAuthority25529.bound (LeftAuthority25529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25529.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25532 .coefficient)
      LeftBound25525.bound (LeftBound25525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25525.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25525.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority25529.bound, LeftBound25525.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25529.bound, LeftBound25525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority25529.actual selector witness, LeftBound25525.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25533

namespace LeftBound25537
def owner : Owner := ⟨.program ⟨214⟩, ⟨26239⟩⟩
def transferEvent : Nat := 25537
def frameStart : Nat := 25423
def rule : BoundRule := .sum [.predecessor 0 25535 .coefficient, .predecessor 1 25536 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25535 .coefficient)
      LeftBound25533.bound (LeftBound25533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25536 .coefficient)
      LeftBound25514.bound (LeftBound25514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25533.bound, LeftBound25514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25533.bound, LeftBound25514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25533.actual selector witness, LeftBound25514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25537

namespace LeftBound25550
def owner : Owner := ⟨.program ⟨214⟩, ⟨26237⟩⟩
def transferEvent : Nat := 25550
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25548 .coefficient, .predecessor 1 25549 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25548 .coefficient)
      LeftBound25371.bound (LeftBound25371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25549 .coefficient)
      LeftBound25354.bound (LeftBound25354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25371.bound, LeftBound25354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25371.bound, LeftBound25354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25371.actual selector witness, LeftBound25354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25550

namespace LeftBound25553
def owner : Owner := ⟨.program ⟨214⟩, ⟨26237⟩⟩
def transferEvent : Nat := 25553
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 25547 .summary, .result 25361 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25547 .summary)
      LeftBound25373.bound (LeftBound25373.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19687⟩⟩) (rawTerms := some (Proof.Events099.exact25547RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25361 .summary)
      LeftBound25356.bound (LeftBound25356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26236⟩⟩) (rawTerms := some (Proof.Events099.exact25361RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25373.bound, LeftBound25356.bound]
def bound : CoeffClass := .finite ⟨352091253649408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25373.bound, LeftBound25356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25373.actual selector witness, LeftBound25356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25553

namespace LeftBound25557
def owner : Owner := ⟨.program ⟨214⟩, ⟨28341⟩⟩
def transferEvent : Nat := 25557
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25555 .coefficient) (.predecessor 1 25556 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25555 .coefficient)
      LeftBound25550.bound (LeftBound25550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25556 .coefficient)
      LeftAuthority25276.bound (LeftAuthority25276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25276.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25550.bound LeftAuthority25276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25550.bound, LeftAuthority25276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25550.actual selector witness) * (LeftAuthority25276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25557

namespace LeftBound25558
def owner : Owner := ⟨.program ⟨214⟩, ⟨28341⟩⟩
def transferEvent : Nat := 25558
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩ [⟨.result 25277 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25277 .coefficient)
      LeftAuthority25276.bound (LeftAuthority25276.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28339⟩⟩) (rawTerms := some (Proof.Events098.exact25277RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25276.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25276.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25276.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25558

namespace LeftBound25559
def owner : Owner := ⟨.program ⟨214⟩, ⟨28341⟩⟩
def transferEvent : Nat := 25559
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 25554 .summary) (.transfer 25558) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25554 .summary)
      LeftBound25553.bound (LeftBound25553.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26237⟩⟩) (rawTerms := some (Proof.Events099.exact25554RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25553.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25558)
      LeftBound25558.bound (LeftBound25558.actual selector witness) := by
  exact .transfer (LeftBound25558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25553.bound LeftBound25558.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25553.bound, LeftBound25558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25553.actual selector witness) * (LeftBound25558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25559

namespace LeftBound25570
def owner : Owner := ⟨.program ⟨214⟩, ⟨21702⟩⟩
def transferEvent : Nat := 25570
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 25568 .coefficient) (.value (.predecessor 1 25569 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25568 .coefficient)
      LeftAuthority25566.bound (LeftAuthority25566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25566.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25569 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority25566.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25566.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25566.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25570

namespace LeftBound25574
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def transferEvent : Nat := 25574
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25572 .coefficient) (.predecessor 1 25573 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25572 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25573 .coefficient)
      LeftBound25570.bound (LeftBound25570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25570.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25570.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound25570.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound25570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound25570.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25574

namespace LeftBound25575
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def transferEvent : Nat := 25575
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩ [⟨.result 25567 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25567 .coefficient)
      LeftAuthority25566.bound (LeftAuthority25566.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21700⟩⟩) (rawTerms := some (Proof.Events099.exact25567RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25566.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25566.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25566.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25566.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25575

namespace LeftBound25576
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def transferEvent : Nat := 25576
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 25575) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25575)
      LeftBound25575.bound (LeftBound25575.actual selector witness) := by
  exact .transfer (LeftBound25575.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound25575.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound25575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound25575.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25576

namespace LeftBound25671
def owner : Owner := ⟨.program ⟨214⟩, ⟨16191⟩⟩
def transferEvent : Nat := 25671
def frameStart : Nat := 25632
def rule : BoundRule := .identity (.predecessor 0 25670 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25670 .coefficient)
      LeftAuthority25668.bound (LeftAuthority25668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25668.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25668.derived selector witness)

def rawBound : CoeffClass := LeftAuthority25668.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority25668.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25671

namespace LeftBound25688
def owner : Owner := ⟨.program ⟨214⟩, ⟨16230⟩⟩
def transferEvent : Nat := 25688
def frameStart : Nat := 25632
def rule : BoundRule := .sum [.predecessor 0 25686 .coefficient, .predecessor 1 25687 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25686 .coefficient)
      LeftBound25671.bound (LeftBound25671.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25687 .coefficient)
      LeftAuthority25684.bound (LeftAuthority25684.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25684.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25671.bound, LeftAuthority25684.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25671.bound, LeftAuthority25684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25671.actual selector witness, LeftAuthority25684.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25688

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
