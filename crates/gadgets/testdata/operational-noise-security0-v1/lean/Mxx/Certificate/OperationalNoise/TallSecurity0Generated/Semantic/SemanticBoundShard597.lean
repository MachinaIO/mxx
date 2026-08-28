import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard596

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87184
def owner : Owner := ⟨.program ⟨214⟩, ⟨10846⟩⟩
def transferEvent : Nat := 87184
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87179 .summary) (.transfer 87183) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87179 .summary)
      LeftBound87177.bound (LeftBound87177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10845⟩⟩) (rawTerms := some (Proof.Events340.exact87179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87183)
      LeftBound87183.bound (LeftBound87183.actual selector witness) := by
  exact .transfer (LeftBound87183.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87177.bound LeftBound87183.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87177.bound, LeftBound87183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87177.actual selector witness) * (LeftBound87183.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87184

namespace LeftBound87192
def owner : Owner := ⟨.program ⟨214⟩, ⟨10984⟩⟩
def transferEvent : Nat := 87192
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87190 .coefficient, .predecessor 1 87191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87190 .coefficient)
      LeftBound87182.bound (LeftBound87182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87191 .coefficient)
      LeftBound87154.bound (LeftBound87154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87154.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87154.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87182.bound, LeftBound87154.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87182.bound, LeftBound87154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87182.actual selector witness, LeftBound87154.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87192

namespace LeftBound87194
def owner : Owner := ⟨.program ⟨214⟩, ⟨10984⟩⟩
def transferEvent : Nat := 87194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 87189 .summary, .result 87159 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87189 .summary)
      LeftBound87184.bound (LeftBound87184.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10846⟩⟩) (rawTerms := some (Proof.Events340.exact87189RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87159 .summary)
      LeftBound87156.bound (LeftBound87156.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10983⟩⟩) (rawTerms := some (Proof.Events340.exact87159RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87156.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87184.bound, LeftBound87156.bound]
def bound : CoeffClass := .finite ⟨95423744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87184.bound, LeftBound87156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87184.actual selector witness, LeftBound87156.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87194

namespace LeftBound87198
def owner : Owner := ⟨.program ⟨214⟩, ⟨25066⟩⟩
def transferEvent : Nat := 87198
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87196 .coefficient) (.predecessor 1 87197 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87196 .coefficient)
      LeftBound87192.bound (LeftBound87192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87197 .coefficient)
      LeftAuthority87130.bound (LeftAuthority87130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87130.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87192.bound LeftAuthority87130.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87192.bound, LeftAuthority87130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87192.actual selector witness) * (LeftAuthority87130.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87198

namespace LeftBound87199
def owner : Owner := ⟨.program ⟨214⟩, ⟨25066⟩⟩
def transferEvent : Nat := 87199
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩ [⟨.result 87131 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87131 .coefficient)
      LeftAuthority87130.bound (LeftAuthority87130.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25065⟩⟩) (rawTerms := some (Proof.Events340.exact87131RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87130.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87130.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87130.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87199

namespace LeftBound87200
def owner : Owner := ⟨.program ⟨214⟩, ⟨25066⟩⟩
def transferEvent : Nat := 87200
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87195 .summary) (.transfer 87199) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87195 .summary)
      LeftBound87194.bound (LeftBound87194.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10984⟩⟩) (rawTerms := some (Proof.Events340.exact87195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87199)
      LeftBound87199.bound (LeftBound87199.actual selector witness) := by
  exact .transfer (LeftBound87199.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87194.bound LeftBound87199.bound
def bound : CoeffClass := .finite ⟨350206667259904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87194.bound, LeftBound87199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87194.actual selector witness) * (LeftBound87199.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87200

namespace LeftBound87211
def owner : Owner := ⟨.program ⟨214⟩, ⟨19170⟩⟩
def transferEvent : Nat := 87211
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 87209 .coefficient) (.value (.predecessor 1 87210 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87209 .coefficient)
      LeftAuthority87207.bound (LeftAuthority87207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87207.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87210 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority87207.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87207.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87207.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87211

namespace LeftBound87215
def owner : Owner := ⟨.program ⟨214⟩, ⟨19171⟩⟩
def transferEvent : Nat := 87215
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87213 .coefficient) (.predecessor 1 87214 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87213 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87214 .coefficient)
      LeftBound87211.bound (LeftBound87211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87211.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound87211.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound87211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound87211.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87215

namespace LeftBound87216
def owner : Owner := ⟨.program ⟨214⟩, ⟨19171⟩⟩
def transferEvent : Nat := 87216
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19168⟩⟩]⟩ [⟨.result 87208 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87208 .coefficient)
      LeftAuthority87207.bound (LeftAuthority87207.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19168⟩⟩) (rawTerms := some (Proof.Events340.exact87208RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87207.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87207.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87207.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87207.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87216

namespace LeftBound87217
def owner : Owner := ⟨.program ⟨214⟩, ⟨19171⟩⟩
def transferEvent : Nat := 87217
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 87216) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87216)
      LeftBound87216.bound (LeftBound87216.actual selector witness) := by
  exact .transfer (LeftBound87216.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound87216.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound87216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound87216.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87217

namespace LeftBound87296
def owner : Owner := ⟨.program ⟨214⟩, ⟨10978⟩⟩
def transferEvent : Nat := 87296
def frameStart : Nat := 87267
def rule : BoundRule := .product (.predecessor 0 87294 .coefficient) (.predecessor 1 87295 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87294 .coefficient)
      LeftAuthority87292.bound (LeftAuthority87292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87292.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87292.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87295 .coefficient)
      LeftAuthority87289.bound (LeftAuthority87289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority87292.bound LeftAuthority87289.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87292.bound, LeftAuthority87289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority87292.actual selector witness) * (LeftAuthority87289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87296

namespace LeftBound87300
def owner : Owner := ⟨.program ⟨214⟩, ⟨10979⟩⟩
def transferEvent : Nat := 87300
def frameStart : Nat := 87267
def rule : BoundRule := .identity (.predecessor 0 87299 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87299 .coefficient)
      LeftBound87296.bound (LeftBound87296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87296.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87296.derived selector witness)

def rawBound : CoeffClass := LeftBound87296.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87296.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound87296.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87300

namespace LeftBound87317
def owner : Owner := ⟨.program ⟨214⟩, ⟨11073⟩⟩
def transferEvent : Nat := 87317
def frameStart : Nat := 87267
def rule : BoundRule := .sum [.predecessor 0 87315 .coefficient, .predecessor 1 87316 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87315 .coefficient)
      LeftBound87300.bound (LeftBound87300.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87316 .coefficient)
      LeftAuthority87313.bound (LeftAuthority87313.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority87313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87300.bound, LeftAuthority87313.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87300.bound, LeftAuthority87313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87300.actual selector witness, LeftAuthority87313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87317

namespace LeftBound87320
def owner : Owner := ⟨.program ⟨214⟩, ⟨11074⟩⟩
def transferEvent : Nat := 87320
def frameStart : Nat := 87267
def rule : BoundRule := .identity (.predecessor 0 87319 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87319 .coefficient)
      LeftBound87317.bound (LeftBound87317.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87317.derived selector witness)

def rawBound : CoeffClass := LeftBound87317.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound87317.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87320

namespace LeftBound87326
def owner : Owner := ⟨.program ⟨214⟩, ⟨11075⟩⟩
def transferEvent : Nat := 87326
def frameStart : Nat := 87267
def rule : BoundRule := .product (.predecessor 0 87324 .coefficient) (.predecessor 1 87325 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87324 .coefficient)
      LeftAuthority87322.bound (LeftAuthority87322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87325 .coefficient)
      LeftBound87320.bound (LeftBound87320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87320.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority87322.bound LeftBound87320.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87322.bound, LeftBound87320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority87322.actual selector witness) * (LeftBound87320.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87326

namespace LeftBound87340
def owner : Owner := ⟨.program ⟨214⟩, ⟨7838⟩⟩
def transferEvent : Nat := 87340
def frameStart : Nat := 87267
def rule : BoundRule := .scale (.predecessor 0 87338 .coefficient) (.value (.predecessor 1 87339 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87338 .coefficient)
      LeftAuthority87336.bound (LeftAuthority87336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87339 .coefficient)
      LeftAuthority87270.bound (LeftAuthority87270.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority87270.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority87336.bound LeftAuthority87270.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87336.bound, LeftAuthority87270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87336.actual selector witness) * (LeftAuthority87270.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87340

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
