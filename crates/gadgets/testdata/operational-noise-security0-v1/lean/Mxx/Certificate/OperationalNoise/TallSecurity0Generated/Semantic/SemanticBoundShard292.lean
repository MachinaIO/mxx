import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard291

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound43304
def owner : Owner := ⟨.program ⟨214⟩, ⟨10998⟩⟩
def transferEvent : Nat := 43304
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
end LeftBound43304

namespace LeftBound43309
def owner : Owner := ⟨.program ⟨214⟩, ⟨10999⟩⟩
def transferEvent : Nat := 43309
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43307 .coefficient) (.predecessor 1 43308 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43307 .coefficient)
      LeftBound43303.bound (LeftBound43303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43303.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43308 .coefficient)
      LeftAuthority1937.bound (LeftAuthority1937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1937.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound43303.bound LeftAuthority1937.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43303.bound, LeftAuthority1937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound43303.actual selector witness) * (LeftAuthority1937.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43309

namespace LeftBound43310
def owner : Owner := ⟨.program ⟨214⟩, ⟨10999⟩⟩
def transferEvent : Nat := 43310
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10852⟩⟩], []⟩ [⟨.result 1938 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1938 .coefficient)
      LeftAuthority1937.bound (LeftAuthority1937.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10852⟩⟩) (rawTerms := some (Proof.Events007.exact1938RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1937.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1937.bound []
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1937.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43310

namespace LeftBound43311
def owner : Owner := ⟨.program ⟨214⟩, ⟨10999⟩⟩
def transferEvent : Nat := 43311
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43306 .summary) (.transfer 43310) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43306 .summary)
      LeftBound43304.bound (LeftBound43304.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10998⟩⟩) (rawTerms := some (Proof.Events169.exact43306RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43304.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43310)
      LeftBound43310.bound (LeftBound43310.actual selector witness) := by
  exact .transfer (LeftBound43310.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound43304.bound LeftBound43310.bound
def bound : CoeffClass := .finite ⟨3328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43304.bound, LeftBound43310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound43304.actual selector witness) * (LeftBound43310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43311

namespace LeftBound43317
def owner : Owner := ⟨.program ⟨214⟩, ⟨10853⟩⟩
def transferEvent : Nat := 43317
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 43315 .coefficient) (.predecessor 1 43316 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43315 .coefficient)
      LeftAuthority1937.bound (LeftAuthority1937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43316 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1937.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1937.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1937.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43317

namespace LeftBound43322
def owner : Owner := ⟨.program ⟨214⟩, ⟨7323⟩⟩
def transferEvent : Nat := 43322
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43320 .coefficient) (.predecessor 1 43321 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43320 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43321 .coefficient)
      LeftBound14027.bound (LeftBound14027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound14027.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound14027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound14027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43322

namespace LeftBound43327
def owner : Owner := ⟨.program ⟨214⟩, ⟨10854⟩⟩
def transferEvent : Nat := 43327
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43325 .coefficient, .predecessor 1 43326 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43325 .coefficient)
      LeftBound43322.bound (LeftBound43322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43326 .coefficient)
      LeftBound43317.bound (LeftBound43317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43322.bound, LeftBound43317.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43322.bound, LeftBound43317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43322.actual selector witness, LeftBound43317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43327

namespace LeftBound43331
def owner : Owner := ⟨.program ⟨214⟩, ⟨10855⟩⟩
def transferEvent : Nat := 43331
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43329 .coefficient, .predecessor 1 43330 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43329 .coefficient)
      LeftBound43327.bound (LeftBound43327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43327.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43330 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43327.bound, LeftBound14019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43327.bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43327.actual selector witness, LeftBound14019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43331

namespace LeftBound43332
def owner : Owner := ⟨.program ⟨214⟩, ⟨10855⟩⟩
def transferEvent : Nat := 43332
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
end LeftBound43332

namespace LeftBound43337
def owner : Owner := ⟨.program ⟨214⟩, ⟨10856⟩⟩
def transferEvent : Nat := 43337
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43335 .coefficient) (.predecessor 1 43336 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43335 .coefficient)
      LeftBound43331.bound (LeftBound43331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43336 .coefficient)
      LeftBound14016.bound (LeftBound14016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43331.bound LeftBound14016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43331.bound, LeftBound14016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43331.actual selector witness) * (LeftBound14016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43337

namespace LeftBound43338
def owner : Owner := ⟨.program ⟨214⟩, ⟨10856⟩⟩
def transferEvent : Nat := 43338
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩ [⟨.result 14013 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14013 .coefficient)
      LeftAuthority14012.bound (LeftAuthority14012.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7837⟩⟩) (rawTerms := some (Proof.Events054.exact14013RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14012.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14012.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14012.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43338

namespace LeftBound43339
def owner : Owner := ⟨.program ⟨214⟩, ⟨10856⟩⟩
def transferEvent : Nat := 43339
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43334 .summary) (.transfer 43338) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43334 .summary)
      LeftBound43332.bound (LeftBound43332.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10855⟩⟩) (rawTerms := some (Proof.Events169.exact43334RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43338)
      LeftBound43338.bound (LeftBound43338.actual selector witness) := by
  exact .transfer (LeftBound43338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43332.bound LeftBound43338.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43332.bound, LeftBound43338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43332.actual selector witness) * (LeftBound43338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43339

namespace LeftBound43347
def owner : Owner := ⟨.program ⟨214⟩, ⟨11000⟩⟩
def transferEvent : Nat := 43347
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43345 .coefficient, .predecessor 1 43346 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43345 .coefficient)
      LeftBound43337.bound (LeftBound43337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43346 .coefficient)
      LeftBound43309.bound (LeftBound43309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43337.bound, LeftBound43309.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43337.bound, LeftBound43309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43337.actual selector witness, LeftBound43309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43347

namespace LeftBound43349
def owner : Owner := ⟨.program ⟨214⟩, ⟨11000⟩⟩
def transferEvent : Nat := 43349
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 43344 .summary, .result 43314 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43344 .summary)
      LeftBound43339.bound (LeftBound43339.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10856⟩⟩) (rawTerms := some (Proof.Events169.exact43344RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43314 .summary)
      LeftBound43311.bound (LeftBound43311.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10999⟩⟩) (rawTerms := some (Proof.Events169.exact43314RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43339.bound, LeftBound43311.bound]
def bound : CoeffClass := .finite ⟨95423744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43339.bound, LeftBound43311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43339.actual selector witness, LeftBound43311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43349

namespace LeftBound43353
def owner : Owner := ⟨.program ⟨214⟩, ⟨25076⟩⟩
def transferEvent : Nat := 43353
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43351 .coefficient) (.predecessor 1 43352 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43351 .coefficient)
      LeftBound43347.bound (LeftBound43347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43352 .coefficient)
      LeftAuthority43285.bound (LeftAuthority43285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43285.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43285.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43347.bound LeftAuthority43285.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43347.bound, LeftAuthority43285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43347.actual selector witness) * (LeftAuthority43285.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43353

namespace LeftBound43354
def owner : Owner := ⟨.program ⟨214⟩, ⟨25076⟩⟩
def transferEvent : Nat := 43354
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩ [⟨.result 43286 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43286 .coefficient)
      LeftAuthority43285.bound (LeftAuthority43285.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25075⟩⟩) (rawTerms := some (Proof.Events169.exact43286RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43285.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43285.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43285.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43285.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43354

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
