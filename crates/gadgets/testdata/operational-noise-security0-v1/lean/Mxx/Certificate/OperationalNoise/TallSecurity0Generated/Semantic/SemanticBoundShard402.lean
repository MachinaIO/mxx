import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard401

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound59040
def owner : Owner := ⟨.program ⟨214⟩, ⟨10489⟩⟩
def transferEvent : Nat := 59040
def frameStart : Nat := 59011
def rule : BoundRule := .product (.predecessor 0 59038 .coefficient) (.predecessor 1 59039 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59038 .coefficient)
      LeftAuthority59036.bound (LeftAuthority59036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59036.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59039 .coefficient)
      LeftAuthority59033.bound (LeftAuthority59033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59033.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority59036.bound LeftAuthority59033.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59036.bound, LeftAuthority59033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority59036.actual selector witness) * (LeftAuthority59033.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59040

namespace LeftBound59044
def owner : Owner := ⟨.program ⟨214⟩, ⟨10490⟩⟩
def transferEvent : Nat := 59044
def frameStart : Nat := 59011
def rule : BoundRule := .identity (.predecessor 0 59043 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59043 .coefficient)
      LeftBound59040.bound (LeftBound59040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59040.derived selector witness)

def rawBound : CoeffClass := LeftBound59040.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound59040.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound59044

namespace LeftBound59061
def owner : Owner := ⟨.program ⟨214⟩, ⟨10580⟩⟩
def transferEvent : Nat := 59061
def frameStart : Nat := 59011
def rule : BoundRule := .sum [.predecessor 0 59059 .coefficient, .predecessor 1 59060 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59059 .coefficient)
      LeftBound59044.bound (LeftBound59044.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound59044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59060 .coefficient)
      LeftAuthority59057.bound (LeftAuthority59057.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority59057.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59044.bound, LeftAuthority59057.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59044.bound, LeftAuthority59057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59044.actual selector witness, LeftAuthority59057.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59061

namespace LeftBound59064
def owner : Owner := ⟨.program ⟨214⟩, ⟨10581⟩⟩
def transferEvent : Nat := 59064
def frameStart : Nat := 59011
def rule : BoundRule := .identity (.predecessor 0 59063 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59063 .coefficient)
      LeftBound59061.bound (LeftBound59061.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound59061.derived selector witness)

def rawBound : CoeffClass := LeftBound59061.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound59061.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound59064

namespace LeftBound59070
def owner : Owner := ⟨.program ⟨214⟩, ⟨10582⟩⟩
def transferEvent : Nat := 59070
def frameStart : Nat := 59011
def rule : BoundRule := .product (.predecessor 0 59068 .coefficient) (.predecessor 1 59069 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59068 .coefficient)
      LeftAuthority59066.bound (LeftAuthority59066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59066.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59069 .coefficient)
      LeftBound59064.bound (LeftBound59064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59064.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority59066.bound LeftBound59064.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59066.bound, LeftBound59064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority59066.actual selector witness) * (LeftBound59064.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59070

namespace LeftBound59086
def owner : Owner := ⟨.program ⟨214⟩, ⟨7832⟩⟩
def transferEvent : Nat := 59086
def frameStart : Nat := 59011
def rule : BoundRule := .scale (.predecessor 0 59084 .coefficient) (.value (.predecessor 1 59085 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59084 .coefficient)
      LeftAuthority59082.bound (LeftAuthority59082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59085 .coefficient)
      LeftAuthority59073.bound (LeftAuthority59073.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority59073.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority59082.bound LeftAuthority59073.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59082.bound, LeftAuthority59073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority59082.actual selector witness) * (LeftAuthority59073.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound59086

namespace LeftBound59089
def owner : Owner := ⟨.program ⟨214⟩, ⟨6771⟩⟩
def transferEvent : Nat := 59089
def frameStart : Nat := 59011
def rule : BoundRule := .identity (.predecessor 0 59088 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59088 .coefficient)
      LeftAuthority59076.bound (LeftAuthority59076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59076.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59076.derived selector witness)

def rawBound : CoeffClass := LeftAuthority59076.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority59076.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound59089

namespace LeftBound59093
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def transferEvent : Nat := 59093
def frameStart : Nat := 59011
def rule : BoundRule := .product (.predecessor 0 59091 .coefficient) (.predecessor 1 59092 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59091 .coefficient)
      LeftBound59089.bound (LeftBound59089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59089.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59089.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59092 .coefficient)
      LeftBound59086.bound (LeftBound59086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59086.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59089.bound LeftBound59086.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59089.bound, LeftBound59086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59089.actual selector witness) * (LeftBound59086.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59093

namespace LeftBound59098
def owner : Owner := ⟨.program ⟨214⟩, ⟨10583⟩⟩
def transferEvent : Nat := 59098
def frameStart : Nat := 59011
def rule : BoundRule := .sum [.predecessor 0 59096 .coefficient, .predecessor 1 59097 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59096 .coefficient)
      LeftBound59093.bound (LeftBound59093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59093.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59097 .coefficient)
      LeftBound59070.bound (LeftBound59070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59093.bound, LeftBound59070.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59093.bound, LeftBound59070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59093.actual selector witness, LeftBound59070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59098

namespace LeftBound59102
def owner : Owner := ⟨.program ⟨214⟩, ⟨24919⟩⟩
def transferEvent : Nat := 59102
def frameStart : Nat := 59011
def rule : BoundRule := .product (.predecessor 0 59100 .coefficient) (.predecessor 1 59101 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59100 .coefficient)
      LeftBound59098.bound (LeftBound59098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59098.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59101 .coefficient)
      LeftAuthority59055.bound (LeftAuthority59055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59055.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59055.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59098.bound LeftAuthority59055.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59098.bound, LeftAuthority59055.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59098.actual selector witness) * (LeftAuthority59055.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59102

namespace LeftBound59113
def owner : Owner := ⟨.program ⟨214⟩, ⟨14798⟩⟩
def transferEvent : Nat := 59113
def frameStart : Nat := 59011
def rule : BoundRule := .product (.predecessor 0 59111 .coefficient) (.predecessor 1 59112 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59111 .coefficient)
      LeftAuthority59066.bound (LeftAuthority59066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59066.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59112 .coefficient)
      LeftAuthority59109.bound (LeftAuthority59109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59109.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59109.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority59066.bound LeftAuthority59109.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59066.bound, LeftAuthority59109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority59066.actual selector witness) * (LeftAuthority59109.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59113

namespace LeftBound59121
def owner : Owner := ⟨.program ⟨214⟩, ⟨14799⟩⟩
def transferEvent : Nat := 59121
def frameStart : Nat := 59011
def rule : BoundRule := .sum [.predecessor 0 59119 .coefficient, .predecessor 1 59120 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59119 .coefficient)
      LeftAuthority59117.bound (LeftAuthority59117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59117.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59120 .coefficient)
      LeftBound59113.bound (LeftBound59113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority59117.bound, LeftBound59113.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59117.bound, LeftBound59113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority59117.actual selector witness, LeftBound59113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59121

namespace LeftBound59125
def owner : Owner := ⟨.program ⟨214⟩, ⟨24920⟩⟩
def transferEvent : Nat := 59125
def frameStart : Nat := 59011
def rule : BoundRule := .sum [.predecessor 0 59123 .coefficient, .predecessor 1 59124 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59123 .coefficient)
      LeftBound59121.bound (LeftBound59121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59124 .coefficient)
      LeftBound59102.bound (LeftBound59102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59121.bound, LeftBound59102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59121.bound, LeftBound59102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59121.actual selector witness, LeftBound59102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59125

namespace LeftBound59138
def owner : Owner := ⟨.program ⟨214⟩, ⟨24918⟩⟩
def transferEvent : Nat := 59138
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59136 .coefficient, .predecessor 1 59137 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59136 .coefficient)
      LeftBound58959.bound (LeftBound58959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59137 .coefficient)
      LeftBound58942.bound (LeftBound58942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact58949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58942.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58959.bound, LeftBound58942.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58959.bound, LeftBound58942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58959.actual selector witness, LeftBound58942.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59138

namespace LeftBound59141
def owner : Owner := ⟨.program ⟨214⟩, ⟨24918⟩⟩
def transferEvent : Nat := 59141
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59135 .summary, .result 58949 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59135 .summary)
      LeftBound58961.bound (LeftBound58961.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19031⟩⟩) (rawTerms := some (Proof.Events230.exact59135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58949 .summary)
      LeftBound58944.bound (LeftBound58944.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24917⟩⟩) (rawTerms := some (Proof.Events230.exact58949RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58944.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58961.bound, LeftBound58944.bound]
def bound : CoeffClass := .finite ⟨352011863863296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58961.bound, LeftBound58944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58961.actual selector witness, LeftBound58944.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59141

namespace LeftBound59145
def owner : Owner := ⟨.program ⟨214⟩, ⟨26372⟩⟩
def transferEvent : Nat := 59145
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 59143 .coefficient) (.predecessor 1 59144 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59143 .coefficient)
      LeftBound59138.bound (LeftBound59138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59144 .coefficient)
      LeftAuthority58864.bound (LeftAuthority58864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58864.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59138.bound LeftAuthority58864.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59138.bound, LeftAuthority58864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59138.actual selector witness) * (LeftAuthority58864.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59145

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
