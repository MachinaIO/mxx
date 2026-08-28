import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard164

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound25295
def owner : Owner := ⟨.program ⟨214⟩, ⟨7351⟩⟩
def transferEvent : Nat := 25295
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25293 .coefficient) (.predecessor 1 25294 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25293 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25294 .coefficient)
      LeftBound10479.bound (LeftBound10479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound10479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound10479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound10479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25295

namespace LeftBound25300
def owner : Owner := ⟨.program ⟨214⟩, ⟨11651⟩⟩
def transferEvent : Nat := 25300
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25298 .coefficient, .predecessor 1 25299 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25298 .coefficient)
      LeftBound25295.bound (LeftBound25295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25299 .coefficient)
      LeftBound25290.bound (LeftBound25290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25290.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25295.bound, LeftBound25290.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25295.bound, LeftBound25290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25295.actual selector witness, LeftBound25290.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25300

namespace LeftBound25304
def owner : Owner := ⟨.program ⟨214⟩, ⟨11652⟩⟩
def transferEvent : Nat := 25304
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25302 .coefficient, .predecessor 1 25303 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25302 .coefficient)
      LeftBound25300.bound (LeftBound25300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25303 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25300.bound, LeftBound10471.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25300.bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25300.actual selector witness, LeftBound10471.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25304

namespace LeftBound25305
def owner : Owner := ⟨.program ⟨214⟩, ⟨11652⟩⟩
def transferEvent : Nat := 25305
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩ [⟨.result 10472 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10472 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨95⟩⟩) (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10471.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10471.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25305

namespace LeftBound25310
def owner : Owner := ⟨.program ⟨214⟩, ⟨14671⟩⟩
def transferEvent : Nat := 25310
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25308 .coefficient) (.predecessor 1 25309 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25308 .coefficient)
      LeftBound25304.bound (LeftBound25304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25304.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25304.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25309 .coefficient)
      LeftAuthority1028.bound (LeftAuthority1028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1028.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1028.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound25304.bound LeftAuthority1028.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25304.bound, LeftAuthority1028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound25304.actual selector witness) * (LeftAuthority1028.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25310

namespace LeftBound25311
def owner : Owner := ⟨.program ⟨214⟩, ⟨14671⟩⟩
def transferEvent : Nat := 25311
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14668⟩⟩], []⟩ [⟨.result 1029 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1029 .coefficient)
      LeftAuthority1028.bound (LeftAuthority1028.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14668⟩⟩) (rawTerms := some (Proof.Events004.exact1029RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1028.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1028.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1028.bound []
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1028.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25311

namespace LeftBound25312
def owner : Owner := ⟨.program ⟨214⟩, ⟨14671⟩⟩
def transferEvent : Nat := 25312
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 25307 .summary) (.transfer 25311) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25307 .summary)
      LeftBound25305.bound (LeftBound25305.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11652⟩⟩) (rawTerms := some (Proof.Events098.exact25307RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25311)
      LeftBound25311.bound (LeftBound25311.actual selector witness) := by
  exact .transfer (LeftBound25311.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound25305.bound LeftBound25311.bound
def bound : CoeffClass := .finite ⟨23296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25305.bound, LeftBound25311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound25305.actual selector witness) * (LeftBound25311.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25312

namespace LeftBound25318
def owner : Owner := ⟨.program ⟨214⟩, ⟨14672⟩⟩
def transferEvent : Nat := 25318
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 25316 .coefficient) (.predecessor 1 25317 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25316 .coefficient)
      LeftAuthority1028.bound (LeftAuthority1028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1028.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25317 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1028.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1028.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1028.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25318

namespace LeftBound25323
def owner : Owner := ⟨.program ⟨214⟩, ⟨7332⟩⟩
def transferEvent : Nat := 25323
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25321 .coefficient) (.predecessor 1 25322 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25321 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25322 .coefficient)
      LeftBound10520.bound (LeftBound10520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound10520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound10520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound10520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25323

namespace LeftBound25328
def owner : Owner := ⟨.program ⟨214⟩, ⟨14673⟩⟩
def transferEvent : Nat := 25328
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25326 .coefficient, .predecessor 1 25327 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25326 .coefficient)
      LeftBound25323.bound (LeftBound25323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25327 .coefficient)
      LeftBound25318.bound (LeftBound25318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25323.bound, LeftBound25318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25323.bound, LeftBound25318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25323.actual selector witness, LeftBound25318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25328

namespace LeftBound25332
def owner : Owner := ⟨.program ⟨214⟩, ⟨14674⟩⟩
def transferEvent : Nat := 25332
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25330 .coefficient, .predecessor 1 25331 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25330 .coefficient)
      LeftBound25328.bound (LeftBound25328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25331 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25328.bound, LeftBound10512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25328.bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25328.actual selector witness, LeftBound10512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25332

namespace LeftBound25333
def owner : Owner := ⟨.program ⟨214⟩, ⟨14674⟩⟩
def transferEvent : Nat := 25333
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩ [⟨.result 10513 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10513 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨76⟩⟩) (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10512.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10512.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25333

namespace LeftBound25338
def owner : Owner := ⟨.program ⟨214⟩, ⟨14675⟩⟩
def transferEvent : Nat := 25338
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25336 .coefficient) (.predecessor 1 25337 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25336 .coefficient)
      LeftBound25332.bound (LeftBound25332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25332.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25337 .coefficient)
      LeftBound10509.bound (LeftBound10509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10509.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25332.bound LeftBound10509.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25332.bound, LeftBound10509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25332.actual selector witness) * (LeftBound10509.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25338

namespace LeftBound25339
def owner : Owner := ⟨.program ⟨214⟩, ⟨14675⟩⟩
def transferEvent : Nat := 25339
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩ [⟨.result 10506 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10506 .coefficient)
      LeftAuthority10505.bound (LeftAuthority10505.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7858⟩⟩) (rawTerms := some (Proof.Events041.exact10506RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10505.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10505.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10505.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25339

namespace LeftBound25340
def owner : Owner := ⟨.program ⟨214⟩, ⟨14675⟩⟩
def transferEvent : Nat := 25340
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 25335 .summary) (.transfer 25339) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25335 .summary)
      LeftBound25333.bound (LeftBound25333.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14674⟩⟩) (rawTerms := some (Proof.Events098.exact25335RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25339)
      LeftBound25339.bound (LeftBound25339.actual selector witness) := by
  exact .transfer (LeftBound25339.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25333.bound LeftBound25339.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25333.bound, LeftBound25339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25333.actual selector witness) * (LeftBound25339.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25340

namespace LeftBound25348
def owner : Owner := ⟨.program ⟨214⟩, ⟨14676⟩⟩
def transferEvent : Nat := 25348
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25346 .coefficient, .predecessor 1 25347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25346 .coefficient)
      LeftBound25338.bound (LeftBound25338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25347 .coefficient)
      LeftBound25310.bound (LeftBound25310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25338.bound, LeftBound25310.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25338.bound, LeftBound25310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25338.actual selector witness, LeftBound25310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25348

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
