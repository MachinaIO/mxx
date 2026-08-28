import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard462

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound68259
def owner : Owner := ⟨.program ⟨214⟩, ⟨11956⟩⟩
def transferEvent : Nat := 68259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68257 .coefficient, .predecessor 1 68258 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68257 .coefficient)
      LeftBound68249.bound (LeftBound68249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68249.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68258 .coefficient)
      LeftBound68221.bound (LeftBound68221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68249.bound, LeftBound68221.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68249.bound, LeftBound68221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68249.actual selector witness, LeftBound68221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68259

namespace LeftBound68261
def owner : Owner := ⟨.program ⟨214⟩, ⟨11956⟩⟩
def transferEvent : Nat := 68261
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 68256 .summary, .result 68226 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68256 .summary)
      LeftBound68251.bound (LeftBound68251.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9714⟩⟩) (rawTerms := some (Proof.Events266.exact68256RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68226 .summary)
      LeftBound68223.bound (LeftBound68223.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11955⟩⟩) (rawTerms := some (Proof.Events266.exact68226RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68223.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68251.bound, LeftBound68223.bound]
def bound : CoeffClass := .finite ⟨95450368, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68251.bound, LeftBound68223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68251.actual selector witness, LeftBound68223.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68261

namespace LeftBound68265
def owner : Owner := ⟨.program ⟨214⟩, ⟨25215⟩⟩
def transferEvent : Nat := 68265
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68263 .coefficient) (.predecessor 1 68264 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68263 .coefficient)
      LeftBound68259.bound (LeftBound68259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68264 .coefficient)
      LeftAuthority68197.bound (LeftAuthority68197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68197.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68197.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68259.bound LeftAuthority68197.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68259.bound, LeftAuthority68197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68259.actual selector witness) * (LeftAuthority68197.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68265

namespace LeftBound68266
def owner : Owner := ⟨.program ⟨214⟩, ⟨25215⟩⟩
def transferEvent : Nat := 68266
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩ [⟨.result 68198 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68198 .coefficient)
      LeftAuthority68197.bound (LeftAuthority68197.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25214⟩⟩) (rawTerms := some (Proof.Events266.exact68198RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68197.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68197.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority68197.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68197.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68266

namespace LeftBound68267
def owner : Owner := ⟨.program ⟨214⟩, ⟨25215⟩⟩
def transferEvent : Nat := 68267
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68262 .summary) (.transfer 68266) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68262 .summary)
      LeftBound68261.bound (LeftBound68261.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11956⟩⟩) (rawTerms := some (Proof.Events266.exact68262RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68266)
      LeftBound68266.bound (LeftBound68266.actual selector witness) := by
  exact .transfer (LeftBound68266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68261.bound LeftBound68266.bound
def bound : CoeffClass := .finite ⟨350304377765888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68261.bound, LeftBound68266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68261.actual selector witness) * (LeftBound68266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68267

namespace LeftBound68278
def owner : Owner := ⟨.program ⟨214⟩, ⟨19814⟩⟩
def transferEvent : Nat := 68278
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 68276 .coefficient) (.value (.predecessor 1 68277 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68276 .coefficient)
      LeftAuthority68274.bound (LeftAuthority68274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68274.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68277 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority68274.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68274.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68274.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68278

namespace LeftBound68282
def owner : Owner := ⟨.program ⟨214⟩, ⟨19815⟩⟩
def transferEvent : Nat := 68282
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68280 .coefficient) (.predecessor 1 68281 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68280 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68281 .coefficient)
      LeftBound68278.bound (LeftBound68278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68278.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound68278.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound68278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound68278.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68282

namespace LeftBound68283
def owner : Owner := ⟨.program ⟨214⟩, ⟨19815⟩⟩
def transferEvent : Nat := 68283
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩ [⟨.result 68275 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68275 .coefficient)
      LeftAuthority68274.bound (LeftAuthority68274.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19812⟩⟩) (rawTerms := some (Proof.Events266.exact68275RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68274.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68274.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority68274.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68274.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68283

namespace LeftBound68284
def owner : Owner := ⟨.program ⟨214⟩, ⟨19815⟩⟩
def transferEvent : Nat := 68284
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 68283) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68283)
      LeftBound68283.bound (LeftBound68283.actual selector witness) := by
  exact .transfer (LeftBound68283.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound68283.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound68283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound68283.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68284

namespace LeftBound68363
def owner : Owner := ⟨.program ⟨214⟩, ⟨11950⟩⟩
def transferEvent : Nat := 68363
def frameStart : Nat := 68334
def rule : BoundRule := .product (.predecessor 0 68361 .coefficient) (.predecessor 1 68362 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68361 .coefficient)
      LeftAuthority68359.bound (LeftAuthority68359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68362 .coefficient)
      LeftAuthority68356.bound (LeftAuthority68356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68356.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68356.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority68359.bound LeftAuthority68356.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68359.bound, LeftAuthority68356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority68359.actual selector witness) * (LeftAuthority68356.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68363

namespace LeftBound68367
def owner : Owner := ⟨.program ⟨214⟩, ⟨11951⟩⟩
def transferEvent : Nat := 68367
def frameStart : Nat := 68334
def rule : BoundRule := .identity (.predecessor 0 68366 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68366 .coefficient)
      LeftBound68363.bound (LeftBound68363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68363.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68363.derived selector witness)

def rawBound : CoeffClass := LeftBound68363.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound68363.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68367

namespace LeftBound68384
def owner : Owner := ⟨.program ⟨214⟩, ⟨12049⟩⟩
def transferEvent : Nat := 68384
def frameStart : Nat := 68334
def rule : BoundRule := .sum [.predecessor 0 68382 .coefficient, .predecessor 1 68383 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68382 .coefficient)
      LeftBound68367.bound (LeftBound68367.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound68367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68383 .coefficient)
      LeftAuthority68380.bound (LeftAuthority68380.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority68380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68367.bound, LeftAuthority68380.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68367.bound, LeftAuthority68380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68367.actual selector witness, LeftAuthority68380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68384

namespace LeftBound68387
def owner : Owner := ⟨.program ⟨214⟩, ⟨12050⟩⟩
def transferEvent : Nat := 68387
def frameStart : Nat := 68334
def rule : BoundRule := .identity (.predecessor 0 68386 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68386 .coefficient)
      LeftBound68384.bound (LeftBound68384.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound68384.derived selector witness)

def rawBound : CoeffClass := LeftBound68384.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound68384.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68387

namespace LeftBound68393
def owner : Owner := ⟨.program ⟨214⟩, ⟨12051⟩⟩
def transferEvent : Nat := 68393
def frameStart : Nat := 68334
def rule : BoundRule := .product (.predecessor 0 68391 .coefficient) (.predecessor 1 68392 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68391 .coefficient)
      LeftAuthority68389.bound (LeftAuthority68389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68392 .coefficient)
      LeftBound68387.bound (LeftBound68387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68387.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68387.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority68389.bound LeftBound68387.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68389.bound, LeftBound68387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority68389.actual selector witness) * (LeftBound68387.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68393

namespace LeftBound68409
def owner : Owner := ⟨.program ⟨214⟩, ⟨7865⟩⟩
def transferEvent : Nat := 68409
def frameStart : Nat := 68334
def rule : BoundRule := .scale (.predecessor 0 68407 .coefficient) (.value (.predecessor 1 68408 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68407 .coefficient)
      LeftAuthority68405.bound (LeftAuthority68405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68408 .coefficient)
      LeftAuthority68396.bound (LeftAuthority68396.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority68396.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority68405.bound LeftAuthority68396.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68405.bound, LeftAuthority68396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68405.actual selector witness) * (LeftAuthority68396.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68409

namespace LeftBound68412
def owner : Owner := ⟨.program ⟨214⟩, ⟨6764⟩⟩
def transferEvent : Nat := 68412
def frameStart : Nat := 68334
def rule : BoundRule := .identity (.predecessor 0 68411 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68411 .coefficient)
      LeftAuthority68399.bound (LeftAuthority68399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68399.derived selector witness)

def rawBound : CoeffClass := LeftAuthority68399.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority68399.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68412

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
