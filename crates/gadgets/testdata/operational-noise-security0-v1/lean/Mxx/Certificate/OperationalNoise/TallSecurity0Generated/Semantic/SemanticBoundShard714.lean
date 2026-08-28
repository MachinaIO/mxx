import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard653
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard713

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104184
def owner : Owner := ⟨.program ⟨214⟩, ⟨22472⟩⟩
def transferEvent : Nat := 104184
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 104183) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104183)
      LeftBound104183.bound (LeftBound104183.actual selector witness) := by
  exact .transfer (LeftBound104183.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound104183.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound104183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound104183.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104184

namespace LeftBound104255
def owner : Owner := ⟨.program ⟨214⟩, ⟨16743⟩⟩
def transferEvent : Nat := 104255
def frameStart : Nat := 104228
def rule : BoundRule := .identity (.predecessor 0 104254 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104254 .coefficient)
      LeftAuthority104252.bound (LeftAuthority104252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104252.derived selector witness)

def rawBound : CoeffClass := LeftAuthority104252.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority104252.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound104255

namespace LeftBound104272
def owner : Owner := ⟨.program ⟨214⟩, ⟨16819⟩⟩
def transferEvent : Nat := 104272
def frameStart : Nat := 104228
def rule : BoundRule := .sum [.predecessor 0 104270 .coefficient, .predecessor 1 104271 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104270 .coefficient)
      LeftBound104255.bound (LeftBound104255.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound104255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104271 .coefficient)
      LeftAuthority104268.bound (LeftAuthority104268.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority104268.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104255.bound, LeftAuthority104268.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104255.bound, LeftAuthority104268.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104255.actual selector witness, LeftAuthority104268.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104272

namespace LeftBound104275
def owner : Owner := ⟨.program ⟨214⟩, ⟨16820⟩⟩
def transferEvent : Nat := 104275
def frameStart : Nat := 104228
def rule : BoundRule := .identity (.predecessor 0 104274 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104274 .coefficient)
      LeftBound104272.bound (LeftBound104272.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound104272.derived selector witness)

def rawBound : CoeffClass := LeftBound104272.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound104272.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound104275

namespace LeftBound104281
def owner : Owner := ⟨.program ⟨214⟩, ⟨16821⟩⟩
def transferEvent : Nat := 104281
def frameStart : Nat := 104228
def rule : BoundRule := .product (.predecessor 0 104279 .coefficient) (.predecessor 1 104280 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104279 .coefficient)
      LeftAuthority104277.bound (LeftAuthority104277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104277.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104280 .coefficient)
      LeftBound104275.bound (LeftBound104275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104275.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104275.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority104277.bound LeftBound104275.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104277.bound, LeftBound104275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority104277.actual selector witness) * (LeftBound104275.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104281

namespace LeftBound104289
def owner : Owner := ⟨.program ⟨214⟩, ⟨16822⟩⟩
def transferEvent : Nat := 104289
def frameStart : Nat := 104228
def rule : BoundRule := .sum [.predecessor 0 104287 .coefficient, .predecessor 1 104288 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104287 .coefficient)
      LeftAuthority104285.bound (LeftAuthority104285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104285.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104288 .coefficient)
      LeftBound104281.bound (LeftBound104281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104285.bound, LeftBound104281.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104285.bound, LeftBound104281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104285.actual selector witness, LeftBound104281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104289

namespace LeftBound104293
def owner : Owner := ⟨.program ⟨214⟩, ⟨29561⟩⟩
def transferEvent : Nat := 104293
def frameStart : Nat := 104228
def rule : BoundRule := .product (.predecessor 0 104291 .coefficient) (.predecessor 1 104292 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104291 .coefficient)
      LeftBound104289.bound (LeftBound104289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104292 .coefficient)
      LeftAuthority104266.bound (LeftAuthority104266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104289.bound LeftAuthority104266.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104289.bound, LeftAuthority104266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104289.actual selector witness) * (LeftAuthority104266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104293

namespace LeftBound104304
def owner : Owner := ⟨.program ⟨214⟩, ⟨17486⟩⟩
def transferEvent : Nat := 104304
def frameStart : Nat := 104228
def rule : BoundRule := .product (.predecessor 0 104302 .coefficient) (.predecessor 1 104303 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104302 .coefficient)
      LeftAuthority104277.bound (LeftAuthority104277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104277.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104303 .coefficient)
      LeftAuthority104300.bound (LeftAuthority104300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104300.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104300.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority104277.bound LeftAuthority104300.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104277.bound, LeftAuthority104300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority104277.actual selector witness) * (LeftAuthority104300.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104304

namespace LeftBound104312
def owner : Owner := ⟨.program ⟨214⟩, ⟨17487⟩⟩
def transferEvent : Nat := 104312
def frameStart : Nat := 104228
def rule : BoundRule := .sum [.predecessor 0 104310 .coefficient, .predecessor 1 104311 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104310 .coefficient)
      LeftAuthority104308.bound (LeftAuthority104308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104311 .coefficient)
      LeftBound104304.bound (LeftBound104304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104304.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104304.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104308.bound, LeftBound104304.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104308.bound, LeftBound104304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104308.actual selector witness, LeftBound104304.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104312

namespace LeftBound104316
def owner : Owner := ⟨.program ⟨214⟩, ⟨29566⟩⟩
def transferEvent : Nat := 104316
def frameStart : Nat := 104228
def rule : BoundRule := .sum [.predecessor 0 104314 .coefficient, .predecessor 1 104315 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104314 .coefficient)
      LeftBound104312.bound (LeftBound104312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104312.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104315 .coefficient)
      LeftBound104293.bound (LeftBound104293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104293.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104312.bound, LeftBound104293.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104312.bound, LeftBound104293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104312.actual selector witness, LeftBound104293.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104316

namespace LeftBound104329
def owner : Owner := ⟨.program ⟨214⟩, ⟨29563⟩⟩
def transferEvent : Nat := 104329
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104327 .coefficient, .predecessor 1 104328 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104327 .coefficient)
      LeftBound104182.bound (LeftBound104182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104328 .coefficient)
      LeftBound104165.bound (LeftBound104165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104165.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104182.bound, LeftBound104165.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104182.bound, LeftBound104165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104182.actual selector witness, LeftBound104165.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104329

namespace LeftBound104332
def owner : Owner := ⟨.program ⟨214⟩, ⟨29563⟩⟩
def transferEvent : Nat := 104332
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104326 .summary, .result 104172 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104326 .summary)
      LeftBound104184.bound (LeftBound104184.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22472⟩⟩) (rawTerms := some (Proof.Events407.exact104326RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104172 .summary)
      LeftBound104167.bound (LeftBound104167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29562⟩⟩) (rawTerms := some (Proof.Events406.exact104172RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104184.bound, LeftBound104167.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104184.bound, LeftBound104167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104184.actual selector witness, LeftBound104167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104332

namespace LeftBound104336
def owner : Owner := ⟨.program ⟨214⟩, ⟨29564⟩⟩
def transferEvent : Nat := 104336
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104334 .coefficient) (.predecessor 1 104335 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104334 .coefficient)
      LeftBound104329.bound (LeftBound104329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104329.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104335 .coefficient)
      LeftBound5558.bound (LeftBound5558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104329.bound LeftBound5558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104329.bound, LeftBound5558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104329.actual selector witness) * (LeftBound5558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104336

namespace LeftBound104337
def owner : Owner := ⟨.program ⟨214⟩, ⟨29564⟩⟩
def transferEvent : Nat := 104337
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩ [⟨.result 5555 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5555 .coefficient)
      LeftAuthority5554.bound (LeftAuthority5554.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6661⟩⟩) (rawTerms := some (Proof.Events021.exact5555RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5554.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5554.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5554.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104337

namespace LeftBound104338
def owner : Owner := ⟨.program ⟨214⟩, ⟨29564⟩⟩
def transferEvent : Nat := 104338
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 104333 .summary) (.transfer 104337) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104333 .summary)
      LeftBound104332.bound (LeftBound104332.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29563⟩⟩) (rawTerms := some (Proof.Events407.exact104333RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104337)
      LeftBound104337.bound (LeftBound104337.actual selector witness) := by
  exact .transfer (LeftBound104337.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104332.bound LeftBound104337.bound
def bound : CoeffClass := .finite ⟨4743310290994884271912517632, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104332.bound, LeftBound104337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104332.actual selector witness) * (LeftBound104337.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104338

namespace LeftBound104353
def owner : Owner := ⟨.program ⟨214⟩, ⟨29345⟩⟩
def transferEvent : Nat := 104353
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104351 .coefficient) (.predecessor 1 104352 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104351 .coefficient)
      LeftBound95922.bound (LeftBound95922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104352 .coefficient)
      LeftAuthority104349.bound (LeftAuthority104349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104349.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104349.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95922.bound LeftAuthority104349.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95922.bound, LeftAuthority104349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95922.actual selector witness) * (LeftAuthority104349.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104353

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
