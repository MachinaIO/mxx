import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard399
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard429

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64305
def owner : Owner := ⟨.program ⟨214⟩, ⟨15161⟩⟩
def transferEvent : Nat := 64305
def frameStart : Nat := 64232
def rule : BoundRule := .sum [.predecessor 0 64303 .coefficient, .predecessor 1 64304 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64303 .coefficient)
      LeftAuthority64301.bound (LeftAuthority64301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64304 .coefficient)
      LeftBound64297.bound (LeftBound64297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64297.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64301.bound, LeftBound64297.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64301.bound, LeftBound64297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority64301.actual selector witness, LeftBound64297.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64305

namespace LeftBound64309
def owner : Owner := ⟨.program ⟨214⟩, ⟨26788⟩⟩
def transferEvent : Nat := 64309
def frameStart : Nat := 64232
def rule : BoundRule := .product (.predecessor 0 64307 .coefficient) (.predecessor 1 64308 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64307 .coefficient)
      LeftBound64305.bound (LeftBound64305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64305.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64308 .coefficient)
      LeftAuthority64282.bound (LeftAuthority64282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64282.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64305.bound LeftAuthority64282.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64305.bound, LeftAuthority64282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64305.actual selector witness) * (LeftAuthority64282.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64309

namespace LeftBound64320
def owner : Owner := ⟨.program ⟨214⟩, ⟨15216⟩⟩
def transferEvent : Nat := 64320
def frameStart : Nat := 64232
def rule : BoundRule := .product (.predecessor 0 64318 .coefficient) (.predecessor 1 64319 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64318 .coefficient)
      LeftAuthority64293.bound (LeftAuthority64293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64319 .coefficient)
      LeftAuthority64316.bound (LeftAuthority64316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64316.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64316.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority64293.bound LeftAuthority64316.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64293.bound, LeftAuthority64316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority64293.actual selector witness) * (LeftAuthority64316.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64320

namespace LeftBound64328
def owner : Owner := ⟨.program ⟨214⟩, ⟨15217⟩⟩
def transferEvent : Nat := 64328
def frameStart : Nat := 64232
def rule : BoundRule := .sum [.predecessor 0 64326 .coefficient, .predecessor 1 64327 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64326 .coefficient)
      LeftAuthority64324.bound (LeftAuthority64324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64324.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64327 .coefficient)
      LeftBound64320.bound (LeftBound64320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64324.bound, LeftBound64320.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64324.bound, LeftBound64320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority64324.actual selector witness, LeftBound64320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64328

namespace LeftBound64332
def owner : Owner := ⟨.program ⟨214⟩, ⟨26793⟩⟩
def transferEvent : Nat := 64332
def frameStart : Nat := 64232
def rule : BoundRule := .sum [.predecessor 0 64330 .coefficient, .predecessor 1 64331 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64330 .coefficient)
      LeftBound64328.bound (LeftBound64328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64331 .coefficient)
      LeftBound64309.bound (LeftBound64309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64328.bound, LeftBound64309.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64328.bound, LeftBound64309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64328.actual selector witness, LeftBound64309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64332

namespace LeftBound64345
def owner : Owner := ⟨.program ⟨214⟩, ⟨26790⟩⟩
def transferEvent : Nat := 64345
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64343 .coefficient, .predecessor 1 64344 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64343 .coefficient)
      LeftBound64174.bound (LeftBound64174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64344 .coefficient)
      LeftBound64157.bound (LeftBound64157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64174.bound, LeftBound64157.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64174.bound, LeftBound64157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64174.actual selector witness, LeftBound64157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64345

namespace LeftBound64348
def owner : Owner := ⟨.program ⟨214⟩, ⟨26790⟩⟩
def transferEvent : Nat := 64348
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64342 .summary, .result 64164 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64342 .summary)
      LeftBound64176.bound (LeftBound64176.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20615⟩⟩) (rawTerms := some (Proof.Events251.exact64342RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64164 .summary)
      LeftBound64159.bound (LeftBound64159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26789⟩⟩) (rawTerms := some (Proof.Events250.exact64164RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64176.bound, LeftBound64159.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64176.bound, LeftBound64159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64176.actual selector witness, LeftBound64159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64348

namespace LeftBound64352
def owner : Owner := ⟨.program ⟨214⟩, ⟨26791⟩⟩
def transferEvent : Nat := 64352
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64350 .coefficient) (.predecessor 1 64351 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64350 .coefficient)
      LeftBound64345.bound (LeftBound64345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64351 .coefficient)
      LeftBound5818.bound (LeftBound5818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64345.bound LeftBound5818.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64345.bound, LeftBound5818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64345.actual selector witness) * (LeftBound5818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64352

namespace LeftBound64353
def owner : Owner := ⟨.program ⟨214⟩, ⟨26791⟩⟩
def transferEvent : Nat := 64353
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩ [⟨.result 5815 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5815 .coefficient)
      LeftAuthority5814.bound (LeftAuthority5814.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6663⟩⟩) (rawTerms := some (Proof.Events022.exact5815RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5814.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5814.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5814.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64353

namespace LeftBound64354
def owner : Owner := ⟨.program ⟨214⟩, ⟨26791⟩⟩
def transferEvent : Nat := 64354
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64349 .summary) (.transfer 64353) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64349 .summary)
      LeftBound64348.bound (LeftBound64348.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26790⟩⟩) (rawTerms := some (Proof.Events251.exact64349RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64353)
      LeftBound64353.bound (LeftBound64353.actual selector witness) := by
  exact .transfer (LeftBound64353.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64348.bound LeftBound64353.bound
def bound : CoeffClass := .finite ⟨4741336194231092170536779776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64348.bound, LeftBound64353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64348.actual selector witness) * (LeftBound64353.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64354

namespace LeftBound64369
def owner : Owner := ⟨.program ⟨214⟩, ⟨26572⟩⟩
def transferEvent : Nat := 64369
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64367 .coefficient) (.predecessor 1 64368 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64367 .coefficient)
      LeftBound58656.bound (LeftBound58656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64368 .coefficient)
      LeftAuthority64365.bound (LeftAuthority64365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64365.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64365.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58656.bound LeftAuthority64365.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58656.bound, LeftAuthority64365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58656.actual selector witness) * (LeftAuthority64365.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64369

namespace LeftBound64370
def owner : Owner := ⟨.program ⟨214⟩, ⟨26572⟩⟩
def transferEvent : Nat := 64370
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26570⟩⟩]⟩ [⟨.result 64366 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64366 .coefficient)
      LeftAuthority64365.bound (LeftAuthority64365.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26570⟩⟩) (rawTerms := some (Proof.Events251.exact64366RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64365.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64365.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64365.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64365.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64370

namespace LeftBound64371
def owner : Owner := ⟨.program ⟨214⟩, ⟨26572⟩⟩
def transferEvent : Nat := 64371
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 58660 .summary) (.transfer 64370) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58660 .summary)
      LeftBound58659.bound (LeftBound58659.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24995⟩⟩) (rawTerms := some (Proof.Events229.exact58660RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64370)
      LeftBound64370.bound (LeftBound64370.actual selector witness) := by
  exact .transfer (LeftBound64370.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58659.bound LeftBound64370.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58659.bound, LeftBound64370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58659.actual selector witness) * (LeftBound64370.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64371

namespace LeftBound64382
def owner : Owner := ⟨.program ⟨214⟩, ⟨20470⟩⟩
def transferEvent : Nat := 64382
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 64380 .coefficient) (.value (.predecessor 1 64381 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64380 .coefficient)
      LeftAuthority64378.bound (LeftAuthority64378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64381 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority64378.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64378.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64378.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound64382

namespace LeftBound64386
def owner : Owner := ⟨.program ⟨214⟩, ⟨20471⟩⟩
def transferEvent : Nat := 64386
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64384 .coefficient) (.predecessor 1 64385 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64384 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64385 .coefficient)
      LeftBound64382.bound (LeftBound64382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64382.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound64382.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound64382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound64382.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64386

namespace LeftBound64387
def owner : Owner := ⟨.program ⟨214⟩, ⟨20471⟩⟩
def transferEvent : Nat := 64387
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20468⟩⟩]⟩ [⟨.result 64379 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64379 .coefficient)
      LeftAuthority64378.bound (LeftAuthority64378.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20468⟩⟩) (rawTerms := some (Proof.Events251.exact64379RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64378.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64378.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64378.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64387

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
