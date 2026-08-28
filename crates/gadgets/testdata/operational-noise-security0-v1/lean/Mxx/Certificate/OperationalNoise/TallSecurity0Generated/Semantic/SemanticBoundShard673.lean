import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard672

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98304
def owner : Owner := ⟨.program ⟨214⟩, ⟨11544⟩⟩
def transferEvent : Nat := 98304
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98302 .coefficient, .predecessor 1 98303 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98302 .coefficient)
      LeftBound98300.bound (LeftBound98300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98303 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98300.bound, LeftBound10972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98300.bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98300.actual selector witness, LeftBound10972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98304

namespace LeftBound98305
def owner : Owner := ⟨.program ⟨214⟩, ⟨11544⟩⟩
def transferEvent : Nat := 98305
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩ [⟨.result 10973 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10973 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨94⟩⟩) (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10972.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10972.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98305

namespace LeftBound98310
def owner : Owner := ⟨.program ⟨214⟩, ⟨14400⟩⟩
def transferEvent : Nat := 98310
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98308 .coefficient) (.predecessor 1 98309 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98308 .coefficient)
      LeftBound98304.bound (LeftBound98304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98304.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98304.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98309 .coefficient)
      LeftAuthority4775.bound (LeftAuthority4775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4775.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound98304.bound LeftAuthority4775.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98304.bound, LeftAuthority4775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound98304.actual selector witness) * (LeftAuthority4775.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98310

namespace LeftBound98311
def owner : Owner := ⟨.program ⟨214⟩, ⟨14400⟩⟩
def transferEvent : Nat := 98311
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14397⟩⟩], []⟩ [⟨.result 4776 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4776 .coefficient)
      LeftAuthority4775.bound (LeftAuthority4775.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14397⟩⟩) (rawTerms := some (Proof.Events018.exact4776RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4775.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4775.bound []
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4775.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98311

namespace LeftBound98312
def owner : Owner := ⟨.program ⟨214⟩, ⟨14400⟩⟩
def transferEvent : Nat := 98312
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98307 .summary) (.transfer 98311) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98307 .summary)
      LeftBound98305.bound (LeftBound98305.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11544⟩⟩) (rawTerms := some (Proof.Events384.exact98307RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98311)
      LeftBound98311.bound (LeftBound98311.actual selector witness) := by
  exact .transfer (LeftBound98311.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound98305.bound LeftBound98311.bound
def bound : CoeffClass := .finite ⟨18304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98305.bound, LeftBound98311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound98305.actual selector witness) * (LeftBound98311.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98312

namespace LeftBound98318
def owner : Owner := ⟨.program ⟨214⟩, ⟨14401⟩⟩
def transferEvent : Nat := 98318
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 98316 .coefficient) (.predecessor 1 98317 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98316 .coefficient)
      LeftAuthority4775.bound (LeftAuthority4775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98317 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4775.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4775.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4775.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98318

namespace LeftBound98323
def owner : Owner := ⟨.program ⟨214⟩, ⟨7098⟩⟩
def transferEvent : Nat := 98323
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98321 .coefficient) (.predecessor 1 98322 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98321 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98322 .coefficient)
      LeftBound11021.bound (LeftBound11021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound11021.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound11021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound11021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98323

namespace LeftBound98328
def owner : Owner := ⟨.program ⟨214⟩, ⟨14402⟩⟩
def transferEvent : Nat := 98328
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98326 .coefficient, .predecessor 1 98327 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98326 .coefficient)
      LeftBound98323.bound (LeftBound98323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98327 .coefficient)
      LeftBound98318.bound (LeftBound98318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98323.bound, LeftBound98318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98323.bound, LeftBound98318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98323.actual selector witness, LeftBound98318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98328

namespace LeftBound98332
def owner : Owner := ⟨.program ⟨214⟩, ⟨14403⟩⟩
def transferEvent : Nat := 98332
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98330 .coefficient, .predecessor 1 98331 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98330 .coefficient)
      LeftBound98328.bound (LeftBound98328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98331 .coefficient)
      LeftBound11013.bound (LeftBound11013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98328.bound, LeftBound11013.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98328.bound, LeftBound11013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98328.actual selector witness, LeftBound11013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98332

namespace LeftBound98333
def owner : Owner := ⟨.program ⟨214⟩, ⟨14403⟩⟩
def transferEvent : Nat := 98333
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨75⟩⟩]⟩ [⟨.result 11014 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11014 .coefficient)
      LeftBound11013.bound (LeftBound11013.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨75⟩⟩) (rawTerms := some (Proof.Events043.exact11014RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11013.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11013.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98333

namespace LeftBound98338
def owner : Owner := ⟨.program ⟨214⟩, ⟨14404⟩⟩
def transferEvent : Nat := 98338
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98336 .coefficient) (.predecessor 1 98337 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98336 .coefficient)
      LeftBound98332.bound (LeftBound98332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98332.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98337 .coefficient)
      LeftBound11010.bound (LeftBound11010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11010.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98332.bound LeftBound11010.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98332.bound, LeftBound11010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98332.actual selector witness) * (LeftBound11010.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98338

namespace LeftBound98339
def owner : Owner := ⟨.program ⟨214⟩, ⟨14404⟩⟩
def transferEvent : Nat := 98339
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩ [⟨.result 11007 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11007 .coefficient)
      LeftAuthority11006.bound (LeftAuthority11006.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7855⟩⟩) (rawTerms := some (Proof.Events042.exact11007RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11006.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11006.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11006.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98339

namespace LeftBound98340
def owner : Owner := ⟨.program ⟨214⟩, ⟨14404⟩⟩
def transferEvent : Nat := 98340
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98335 .summary) (.transfer 98339) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98335 .summary)
      LeftBound98333.bound (LeftBound98333.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14403⟩⟩) (rawTerms := some (Proof.Events384.exact98335RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98339)
      LeftBound98339.bound (LeftBound98339.actual selector witness) := by
  exact .transfer (LeftBound98339.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98333.bound LeftBound98339.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98333.bound, LeftBound98339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98333.actual selector witness) * (LeftBound98339.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98340

namespace LeftBound98348
def owner : Owner := ⟨.program ⟨214⟩, ⟨14405⟩⟩
def transferEvent : Nat := 98348
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98346 .coefficient, .predecessor 1 98347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98346 .coefficient)
      LeftBound98338.bound (LeftBound98338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98347 .coefficient)
      LeftBound98310.bound (LeftBound98310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98338.bound, LeftBound98310.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98338.bound, LeftBound98310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98338.actual selector witness, LeftBound98310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98348

namespace LeftBound98350
def owner : Owner := ⟨.program ⟨214⟩, ⟨14405⟩⟩
def transferEvent : Nat := 98350
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 98345 .summary, .result 98315 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98345 .summary)
      LeftBound98340.bound (LeftBound98340.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14404⟩⟩) (rawTerms := some (Proof.Events384.exact98345RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98315 .summary)
      LeftBound98312.bound (LeftBound98312.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14400⟩⟩) (rawTerms := some (Proof.Events384.exact98315RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98312.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98340.bound, LeftBound98312.bound]
def bound : CoeffClass := .finite ⟨95438720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98340.bound, LeftBound98312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98340.actual selector witness, LeftBound98312.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98350

namespace LeftBound98354
def owner : Owner := ⟨.program ⟨214⟩, ⟨26131⟩⟩
def transferEvent : Nat := 98354
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98352 .coefficient) (.predecessor 1 98353 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98352 .coefficient)
      LeftBound98348.bound (LeftBound98348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98353 .coefficient)
      LeftAuthority98286.bound (LeftAuthority98286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98286.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98348.bound LeftAuthority98286.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98348.bound, LeftAuthority98286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98348.actual selector witness) * (LeftAuthority98286.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98354

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
