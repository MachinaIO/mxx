import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard292

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound43355
def owner : Owner := ⟨.program ⟨214⟩, ⟨25076⟩⟩
def transferEvent : Nat := 43355
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43350 .summary) (.transfer 43354) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43350 .summary)
      LeftBound43349.bound (LeftBound43349.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11000⟩⟩) (rawTerms := some (Proof.Events169.exact43350RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43354)
      LeftBound43354.bound (LeftBound43354.actual selector witness) := by
  exact .transfer (LeftBound43354.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43349.bound LeftBound43354.bound
def bound : CoeffClass := .finite ⟨350206667259904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43349.bound, LeftBound43354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43349.actual selector witness) * (LeftBound43354.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43355

namespace LeftBound43366
def owner : Owner := ⟨.program ⟨214⟩, ⟨19178⟩⟩
def transferEvent : Nat := 43366
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 43364 .coefficient) (.value (.predecessor 1 43365 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43364 .coefficient)
      LeftAuthority43362.bound (LeftAuthority43362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43365 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority43362.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43362.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43362.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43366

namespace LeftBound43370
def owner : Owner := ⟨.program ⟨214⟩, ⟨19179⟩⟩
def transferEvent : Nat := 43370
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43368 .coefficient) (.predecessor 1 43369 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43368 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43369 .coefficient)
      LeftBound43366.bound (LeftBound43366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43366.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound43366.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound43366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound43366.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43370

namespace LeftBound43371
def owner : Owner := ⟨.program ⟨214⟩, ⟨19179⟩⟩
def transferEvent : Nat := 43371
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19176⟩⟩]⟩ [⟨.result 43363 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43363 .coefficient)
      LeftAuthority43362.bound (LeftAuthority43362.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19176⟩⟩) (rawTerms := some (Proof.Events169.exact43363RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43362.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43362.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43362.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43371

namespace LeftBound43372
def owner : Owner := ⟨.program ⟨214⟩, ⟨19179⟩⟩
def transferEvent : Nat := 43372
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 43371) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43371)
      LeftBound43371.bound (LeftBound43371.actual selector witness) := by
  exact .transfer (LeftBound43371.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound43371.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound43371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound43371.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43372

namespace LeftBound43451
def owner : Owner := ⟨.program ⟨214⟩, ⟨10994⟩⟩
def transferEvent : Nat := 43451
def frameStart : Nat := 43422
def rule : BoundRule := .product (.predecessor 0 43449 .coefficient) (.predecessor 1 43450 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43449 .coefficient)
      LeftAuthority43447.bound (LeftAuthority43447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43450 .coefficient)
      LeftAuthority43444.bound (LeftAuthority43444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43444.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43444.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority43447.bound LeftAuthority43444.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43447.bound, LeftAuthority43444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority43447.actual selector witness) * (LeftAuthority43444.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43451

namespace LeftBound43455
def owner : Owner := ⟨.program ⟨214⟩, ⟨10995⟩⟩
def transferEvent : Nat := 43455
def frameStart : Nat := 43422
def rule : BoundRule := .identity (.predecessor 0 43454 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43454 .coefficient)
      LeftBound43451.bound (LeftBound43451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43451.derived selector witness)

def rawBound : CoeffClass := LeftBound43451.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound43451.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43455

namespace LeftBound43472
def owner : Owner := ⟨.program ⟨214⟩, ⟨11081⟩⟩
def transferEvent : Nat := 43472
def frameStart : Nat := 43422
def rule : BoundRule := .sum [.predecessor 0 43470 .coefficient, .predecessor 1 43471 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43470 .coefficient)
      LeftBound43455.bound (LeftBound43455.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43471 .coefficient)
      LeftAuthority43468.bound (LeftAuthority43468.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority43468.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43455.bound, LeftAuthority43468.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43455.bound, LeftAuthority43468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43455.actual selector witness, LeftAuthority43468.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43472

namespace LeftBound43475
def owner : Owner := ⟨.program ⟨214⟩, ⟨11082⟩⟩
def transferEvent : Nat := 43475
def frameStart : Nat := 43422
def rule : BoundRule := .identity (.predecessor 0 43474 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43474 .coefficient)
      LeftBound43472.bound (LeftBound43472.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43472.derived selector witness)

def rawBound : CoeffClass := LeftBound43472.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound43472.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43475

namespace LeftBound43481
def owner : Owner := ⟨.program ⟨214⟩, ⟨11083⟩⟩
def transferEvent : Nat := 43481
def frameStart : Nat := 43422
def rule : BoundRule := .product (.predecessor 0 43479 .coefficient) (.predecessor 1 43480 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43479 .coefficient)
      LeftAuthority43477.bound (LeftAuthority43477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43480 .coefficient)
      LeftBound43475.bound (LeftBound43475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority43477.bound LeftBound43475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43477.bound, LeftBound43475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority43477.actual selector witness) * (LeftBound43475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43481

namespace LeftBound43497
def owner : Owner := ⟨.program ⟨214⟩, ⟨7838⟩⟩
def transferEvent : Nat := 43497
def frameStart : Nat := 43422
def rule : BoundRule := .scale (.predecessor 0 43495 .coefficient) (.value (.predecessor 1 43496 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43495 .coefficient)
      LeftAuthority43493.bound (LeftAuthority43493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43493.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43496 .coefficient)
      LeftAuthority43484.bound (LeftAuthority43484.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority43484.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority43493.bound LeftAuthority43484.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43493.bound, LeftAuthority43484.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43493.actual selector witness) * (LeftAuthority43484.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43497

namespace LeftBound43500
def owner : Owner := ⟨.program ⟨214⟩, ⟨6791⟩⟩
def transferEvent : Nat := 43500
def frameStart : Nat := 43422
def rule : BoundRule := .identity (.predecessor 0 43499 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43499 .coefficient)
      LeftAuthority43487.bound (LeftAuthority43487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43487.derived selector witness)

def rawBound : CoeffClass := LeftAuthority43487.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority43487.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43500

namespace LeftBound43504
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def transferEvent : Nat := 43504
def frameStart : Nat := 43422
def rule : BoundRule := .product (.predecessor 0 43502 .coefficient) (.predecessor 1 43503 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43502 .coefficient)
      LeftBound43500.bound (LeftBound43500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43503 .coefficient)
      LeftBound43497.bound (LeftBound43497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43497.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43500.bound LeftBound43497.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43500.bound, LeftBound43497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43500.actual selector witness) * (LeftBound43497.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43504

namespace LeftBound43509
def owner : Owner := ⟨.program ⟨214⟩, ⟨11084⟩⟩
def transferEvent : Nat := 43509
def frameStart : Nat := 43422
def rule : BoundRule := .sum [.predecessor 0 43507 .coefficient, .predecessor 1 43508 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43507 .coefficient)
      LeftBound43504.bound (LeftBound43504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43504.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43504.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43508 .coefficient)
      LeftBound43481.bound (LeftBound43481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43504.bound, LeftBound43481.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43504.bound, LeftBound43481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43504.actual selector witness, LeftBound43481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43509

namespace LeftBound43513
def owner : Owner := ⟨.program ⟨214⟩, ⟨25078⟩⟩
def transferEvent : Nat := 43513
def frameStart : Nat := 43422
def rule : BoundRule := .product (.predecessor 0 43511 .coefficient) (.predecessor 1 43512 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43511 .coefficient)
      LeftBound43509.bound (LeftBound43509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43512 .coefficient)
      LeftAuthority43466.bound (LeftAuthority43466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43466.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43509.bound LeftAuthority43466.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43509.bound, LeftAuthority43466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43509.actual selector witness) * (LeftAuthority43466.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43513

namespace LeftBound43524
def owner : Owner := ⟨.program ⟨214⟩, ⟨15124⟩⟩
def transferEvent : Nat := 43524
def frameStart : Nat := 43422
def rule : BoundRule := .product (.predecessor 0 43522 .coefficient) (.predecessor 1 43523 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43522 .coefficient)
      LeftAuthority43477.bound (LeftAuthority43477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43523 .coefficient)
      LeftAuthority43520.bound (LeftAuthority43520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority43477.bound LeftAuthority43520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43477.bound, LeftAuthority43520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority43477.actual selector witness) * (LeftAuthority43520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43524

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
