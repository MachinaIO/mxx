import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard567

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound83344
def owner : Owner := ⟨.program ⟨214⟩, ⟨9614⟩⟩
def transferEvent : Nat := 83344
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83339 .summary) (.transfer 83343) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83339 .summary)
      LeftBound83337.bound (LeftBound83337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9613⟩⟩) (rawTerms := some (Proof.Events325.exact83339RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83343)
      LeftBound83343.bound (LeftBound83343.actual selector witness) := by
  exact .transfer (LeftBound83343.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83337.bound LeftBound83343.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83337.bound, LeftBound83343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83337.actual selector witness) * (LeftBound83343.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83344

namespace LeftBound83352
def owner : Owner := ⟨.program ⟨214⟩, ⟨11768⟩⟩
def transferEvent : Nat := 83352
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83350 .coefficient, .predecessor 1 83351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83350 .coefficient)
      LeftBound83342.bound (LeftBound83342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83351 .coefficient)
      LeftBound83314.bound (LeftBound83314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83314.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83342.bound, LeftBound83314.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83342.bound, LeftBound83314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83342.actual selector witness, LeftBound83314.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83352

namespace LeftBound83354
def owner : Owner := ⟨.program ⟨214⟩, ⟨11768⟩⟩
def transferEvent : Nat := 83354
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 83349 .summary, .result 83319 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83349 .summary)
      LeftBound83344.bound (LeftBound83344.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9614⟩⟩) (rawTerms := some (Proof.Events325.exact83349RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83319 .summary)
      LeftBound83316.bound (LeftBound83316.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11767⟩⟩) (rawTerms := some (Proof.Events325.exact83319RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83316.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83344.bound, LeftBound83316.bound]
def bound : CoeffClass := .finite ⟨95445376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83344.bound, LeftBound83316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83344.actual selector witness, LeftBound83316.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83354

namespace LeftBound83358
def owner : Owner := ⟨.program ⟨214⟩, ⟨25143⟩⟩
def transferEvent : Nat := 83358
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83356 .coefficient) (.predecessor 1 83357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83356 .coefficient)
      LeftBound83352.bound (LeftBound83352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83357 .coefficient)
      LeftAuthority83290.bound (LeftAuthority83290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83290.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83290.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83352.bound LeftAuthority83290.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83352.bound, LeftAuthority83290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83352.actual selector witness) * (LeftAuthority83290.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83358

namespace LeftBound83359
def owner : Owner := ⟨.program ⟨214⟩, ⟨25143⟩⟩
def transferEvent : Nat := 83359
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩ [⟨.result 83291 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83291 .coefficient)
      LeftAuthority83290.bound (LeftAuthority83290.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25142⟩⟩) (rawTerms := some (Proof.Events325.exact83291RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83290.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83290.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority83290.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83290.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83359

namespace LeftBound83360
def owner : Owner := ⟨.program ⟨214⟩, ⟨25143⟩⟩
def transferEvent : Nat := 83360
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83355 .summary) (.transfer 83359) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83355 .summary)
      LeftBound83354.bound (LeftBound83354.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11768⟩⟩) (rawTerms := some (Proof.Events325.exact83355RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83359)
      LeftBound83359.bound (LeftBound83359.actual selector witness) := by
  exact .transfer (LeftBound83359.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83354.bound LeftBound83359.bound
def bound : CoeffClass := .finite ⟨350286057046016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83354.bound, LeftBound83359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83354.actual selector witness) * (LeftBound83359.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83360

namespace LeftBound83371
def owner : Owner := ⟨.program ⟨214⟩, ⟨19746⟩⟩
def transferEvent : Nat := 83371
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 83369 .coefficient) (.value (.predecessor 1 83370 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83369 .coefficient)
      LeftAuthority83367.bound (LeftAuthority83367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83370 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority83367.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83367.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83367.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83371

namespace LeftBound83375
def owner : Owner := ⟨.program ⟨214⟩, ⟨19747⟩⟩
def transferEvent : Nat := 83375
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83373 .coefficient) (.predecessor 1 83374 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83373 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83374 .coefficient)
      LeftBound83371.bound (LeftBound83371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83371.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound83371.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound83371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound83371.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83375

namespace LeftBound83376
def owner : Owner := ⟨.program ⟨214⟩, ⟨19747⟩⟩
def transferEvent : Nat := 83376
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩ [⟨.result 83368 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83368 .coefficient)
      LeftAuthority83367.bound (LeftAuthority83367.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19744⟩⟩) (rawTerms := some (Proof.Events325.exact83368RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83367.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority83367.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83367.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83376

namespace LeftBound83377
def owner : Owner := ⟨.program ⟨214⟩, ⟨19747⟩⟩
def transferEvent : Nat := 83377
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 83376) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83376)
      LeftBound83376.bound (LeftBound83376.actual selector witness) := by
  exact .transfer (LeftBound83376.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound83376.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound83376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound83376.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83377

namespace LeftBound83456
def owner : Owner := ⟨.program ⟨214⟩, ⟨11762⟩⟩
def transferEvent : Nat := 83456
def frameStart : Nat := 83427
def rule : BoundRule := .product (.predecessor 0 83454 .coefficient) (.predecessor 1 83455 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83454 .coefficient)
      LeftAuthority83452.bound (LeftAuthority83452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83452.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83455 .coefficient)
      LeftAuthority83449.bound (LeftAuthority83449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83449.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83449.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83452.bound LeftAuthority83449.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83452.bound, LeftAuthority83449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority83452.actual selector witness) * (LeftAuthority83449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83456

namespace LeftBound83460
def owner : Owner := ⟨.program ⟨214⟩, ⟨11763⟩⟩
def transferEvent : Nat := 83460
def frameStart : Nat := 83427
def rule : BoundRule := .identity (.predecessor 0 83459 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83459 .coefficient)
      LeftBound83456.bound (LeftBound83456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83456.derived selector witness)

def rawBound : CoeffClass := LeftBound83456.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound83456.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83460

namespace LeftBound83477
def owner : Owner := ⟨.program ⟨214⟩, ⟨11857⟩⟩
def transferEvent : Nat := 83477
def frameStart : Nat := 83427
def rule : BoundRule := .sum [.predecessor 0 83475 .coefficient, .predecessor 1 83476 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83475 .coefficient)
      LeftBound83460.bound (LeftBound83460.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83476 .coefficient)
      LeftAuthority83473.bound (LeftAuthority83473.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority83473.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83460.bound, LeftAuthority83473.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83460.bound, LeftAuthority83473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83460.actual selector witness, LeftAuthority83473.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83477

namespace LeftBound83480
def owner : Owner := ⟨.program ⟨214⟩, ⟨11858⟩⟩
def transferEvent : Nat := 83480
def frameStart : Nat := 83427
def rule : BoundRule := .identity (.predecessor 0 83479 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83479 .coefficient)
      LeftBound83477.bound (LeftBound83477.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83477.derived selector witness)

def rawBound : CoeffClass := LeftBound83477.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound83477.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83480

namespace LeftBound83486
def owner : Owner := ⟨.program ⟨214⟩, ⟨11859⟩⟩
def transferEvent : Nat := 83486
def frameStart : Nat := 83427
def rule : BoundRule := .product (.predecessor 0 83484 .coefficient) (.predecessor 1 83485 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83484 .coefficient)
      LeftAuthority83482.bound (LeftAuthority83482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83485 .coefficient)
      LeftBound83480.bound (LeftBound83480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83480.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority83482.bound LeftBound83480.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83482.bound, LeftBound83480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority83482.actual selector witness) * (LeftBound83480.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83486

namespace LeftBound83500
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def transferEvent : Nat := 83500
def frameStart : Nat := 83427
def rule : BoundRule := .scale (.predecessor 0 83498 .coefficient) (.value (.predecessor 1 83499 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83498 .coefficient)
      LeftAuthority83496.bound (LeftAuthority83496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83499 .coefficient)
      LeftAuthority83430.bound (LeftAuthority83430.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority83430.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority83496.bound LeftAuthority83430.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83496.bound, LeftAuthority83430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83496.actual selector witness) * (LeftAuthority83430.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83500

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
