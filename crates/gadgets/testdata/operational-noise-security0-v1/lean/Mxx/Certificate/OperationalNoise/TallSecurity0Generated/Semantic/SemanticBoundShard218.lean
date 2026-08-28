import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard170
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard171
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard217

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound33575
def owner : Owner := ⟨.program ⟨214⟩, ⟨28333⟩⟩
def transferEvent : Nat := 33575
def frameStart : Nat := 33498
def rule : BoundRule := .product (.predecessor 0 33573 .coefficient) (.predecessor 1 33574 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33573 .coefficient)
      LeftBound33571.bound (LeftBound33571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33574 .coefficient)
      LeftAuthority33548.bound (LeftAuthority33548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33548.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33571.bound LeftAuthority33548.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33571.bound, LeftAuthority33548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33571.actual selector witness) * (LeftAuthority33548.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33575

namespace LeftBound33586
def owner : Owner := ⟨.program ⟨214⟩, ⟨17676⟩⟩
def transferEvent : Nat := 33586
def frameStart : Nat := 33498
def rule : BoundRule := .product (.predecessor 0 33584 .coefficient) (.predecessor 1 33585 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33584 .coefficient)
      LeftAuthority33559.bound (LeftAuthority33559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33559.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33585 .coefficient)
      LeftAuthority33582.bound (LeftAuthority33582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority33559.bound LeftAuthority33582.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33559.bound, LeftAuthority33582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority33559.actual selector witness) * (LeftAuthority33582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33586

namespace LeftBound33594
def owner : Owner := ⟨.program ⟨214⟩, ⟨17677⟩⟩
def transferEvent : Nat := 33594
def frameStart : Nat := 33498
def rule : BoundRule := .sum [.predecessor 0 33592 .coefficient, .predecessor 1 33593 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33592 .coefficient)
      LeftAuthority33590.bound (LeftAuthority33590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33590.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33593 .coefficient)
      LeftBound33586.bound (LeftBound33586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33586.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority33590.bound, LeftBound33586.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33590.bound, LeftBound33586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority33590.actual selector witness, LeftBound33586.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33594

namespace LeftBound33598
def owner : Owner := ⟨.program ⟨214⟩, ⟨28338⟩⟩
def transferEvent : Nat := 33598
def frameStart : Nat := 33498
def rule : BoundRule := .sum [.predecessor 0 33596 .coefficient, .predecessor 1 33597 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33596 .coefficient)
      LeftBound33594.bound (LeftBound33594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33597 .coefficient)
      LeftBound33575.bound (LeftBound33575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33575.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33594.bound, LeftBound33575.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33594.bound, LeftBound33575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33594.actual selector witness, LeftBound33575.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33598

namespace LeftBound33611
def owner : Owner := ⟨.program ⟨214⟩, ⟨28335⟩⟩
def transferEvent : Nat := 33611
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 33609 .coefficient, .predecessor 1 33610 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33609 .coefficient)
      LeftBound33440.bound (LeftBound33440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33610 .coefficient)
      LeftBound33423.bound (LeftBound33423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33440.bound, LeftBound33423.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33440.bound, LeftBound33423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33440.actual selector witness, LeftBound33423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33611

namespace LeftBound33614
def owner : Owner := ⟨.program ⟨214⟩, ⟨28335⟩⟩
def transferEvent : Nat := 33614
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 33608 .summary, .result 33430 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33608 .summary)
      LeftBound33442.bound (LeftBound33442.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21631⟩⟩) (rawTerms := some (Proof.Events131.exact33608RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33430 .summary)
      LeftBound33425.bound (LeftBound33425.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28334⟩⟩) (rawTerms := some (Proof.Events130.exact33430RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33442.bound, LeftBound33425.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33442.bound, LeftBound33425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33442.actual selector witness, LeftBound33425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33614

namespace LeftBound33618
def owner : Owner := ⟨.program ⟨214⟩, ⟨28336⟩⟩
def transferEvent : Nat := 33618
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33616 .coefficient) (.predecessor 1 33617 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33616 .coefficient)
      LeftBound33611.bound (LeftBound33611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33617 .coefficient)
      LeftBound5678.bound (LeftBound5678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33611.bound LeftBound5678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33611.bound, LeftBound5678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33611.actual selector witness) * (LeftBound5678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33618

namespace LeftBound33619
def owner : Owner := ⟨.program ⟨214⟩, ⟨28336⟩⟩
def transferEvent : Nat := 33619
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩ [⟨.result 5675 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5675 .coefficient)
      LeftAuthority5674.bound (LeftAuthority5674.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6681⟩⟩) (rawTerms := some (Proof.Events022.exact5675RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5674.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5674.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5674.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33619

namespace LeftBound33620
def owner : Owner := ⟨.program ⟨214⟩, ⟨28336⟩⟩
def transferEvent : Nat := 33620
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 33615 .summary) (.transfer 33619) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33615 .summary)
      LeftBound33614.bound (LeftBound33614.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28335⟩⟩) (rawTerms := some (Proof.Events131.exact33615RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33619)
      LeftBound33619.bound (LeftBound33619.actual selector witness) := by
  exact .transfer (LeftBound33619.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33614.bound LeftBound33619.bound
def bound : CoeffClass := .finite ⟨4742323242612988221224648704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33614.bound, LeftBound33619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33614.actual selector witness) * (LeftBound33619.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33620

namespace LeftBound33635
def owner : Owner := ⟨.program ⟨214⟩, ⟨28117⟩⟩
def transferEvent : Nat := 33635
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33633 .coefficient) (.predecessor 1 33634 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33633 .coefficient)
      LeftBound26032.bound (LeftBound26032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33634 .coefficient)
      LeftAuthority33631.bound (LeftAuthority33631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33631.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26032.bound LeftAuthority33631.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26032.bound, LeftAuthority33631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26032.actual selector witness) * (LeftAuthority33631.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33635

namespace LeftBound33636
def owner : Owner := ⟨.program ⟨214⟩, ⟨28117⟩⟩
def transferEvent : Nat := 33636
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28115⟩⟩]⟩ [⟨.result 33632 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33632 .coefficient)
      LeftAuthority33631.bound (LeftAuthority33631.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28115⟩⟩) (rawTerms := some (Proof.Events131.exact33632RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33631.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority33631.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33631.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33636

namespace LeftBound33637
def owner : Owner := ⟨.program ⟨214⟩, ⟨28117⟩⟩
def transferEvent : Nat := 33637
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26036 .summary) (.transfer 33636) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26036 .summary)
      LeftBound26035.bound (LeftBound26035.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26160⟩⟩) (rawTerms := some (Proof.Events101.exact26036RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33636)
      LeftBound33636.bound (LeftBound33636.actual selector witness) := by
  exact .transfer (LeftBound33636.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26035.bound LeftBound33636.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26035.bound, LeftBound33636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26035.actual selector witness) * (LeftBound33636.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33637

namespace LeftBound33648
def owner : Owner := ⟨.program ⟨214⟩, ⟨21486⟩⟩
def transferEvent : Nat := 33648
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 33646 .coefficient) (.value (.predecessor 1 33647 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33646 .coefficient)
      LeftAuthority33644.bound (LeftAuthority33644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33647 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority33644.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33644.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33644.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound33648

namespace LeftBound33652
def owner : Owner := ⟨.program ⟨214⟩, ⟨21487⟩⟩
def transferEvent : Nat := 33652
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33650 .coefficient) (.predecessor 1 33651 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33650 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33651 .coefficient)
      LeftBound33648.bound (LeftBound33648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33648.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound33648.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound33648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound33648.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33652

namespace LeftBound33653
def owner : Owner := ⟨.program ⟨214⟩, ⟨21487⟩⟩
def transferEvent : Nat := 33653
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21484⟩⟩]⟩ [⟨.result 33645 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33645 .coefficient)
      LeftAuthority33644.bound (LeftAuthority33644.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21484⟩⟩) (rawTerms := some (Proof.Events131.exact33645RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33644.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority33644.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33644.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33653

namespace LeftBound33654
def owner : Owner := ⟨.program ⟨214⟩, ⟨21487⟩⟩
def transferEvent : Nat := 33654
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 33653) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33653)
      LeftBound33653.bound (LeftBound33653.actual selector witness) := by
  exact .transfer (LeftBound33653.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound33653.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound33653.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound33653.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33654

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
