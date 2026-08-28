import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard530

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound78799
def owner : Owner := ⟨.program ⟨214⟩, ⟨20607⟩⟩
def transferEvent : Nat := 78799
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78797 .coefficient) (.predecessor 1 78798 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78797 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78798 .coefficient)
      LeftBound78795.bound (LeftBound78795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78795.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound78795.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound78795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound78795.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78799

namespace LeftBound78800
def owner : Owner := ⟨.program ⟨214⟩, ⟨20607⟩⟩
def transferEvent : Nat := 78800
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩ [⟨.result 78792 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78792 .coefficient)
      LeftAuthority78791.bound (LeftAuthority78791.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20604⟩⟩) (rawTerms := some (Proof.Events307.exact78792RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78791.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority78791.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78791.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78800

namespace LeftBound78801
def owner : Owner := ⟨.program ⟨214⟩, ⟨20607⟩⟩
def transferEvent : Nat := 78801
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 78800) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78800)
      LeftBound78800.bound (LeftBound78800.actual selector witness) := by
  exact .transfer (LeftBound78800.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound78800.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound78800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound78800.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78801

namespace LeftBound78896
def owner : Owner := ⟨.program ⟨214⟩, ⟨15111⟩⟩
def transferEvent : Nat := 78896
def frameStart : Nat := 78857
def rule : BoundRule := .identity (.predecessor 0 78895 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78895 .coefficient)
      LeftAuthority78893.bound (LeftAuthority78893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78893.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78893.derived selector witness)

def rawBound : CoeffClass := LeftAuthority78893.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority78893.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78896

namespace LeftBound78913
def owner : Owner := ⟨.program ⟨214⟩, ⟨15150⟩⟩
def transferEvent : Nat := 78913
def frameStart : Nat := 78857
def rule : BoundRule := .sum [.predecessor 0 78911 .coefficient, .predecessor 1 78912 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78911 .coefficient)
      LeftBound78896.bound (LeftBound78896.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78912 .coefficient)
      LeftAuthority78909.bound (LeftAuthority78909.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority78909.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78896.bound, LeftAuthority78909.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78896.bound, LeftAuthority78909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78896.actual selector witness, LeftAuthority78909.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78913

namespace LeftBound78916
def owner : Owner := ⟨.program ⟨214⟩, ⟨15151⟩⟩
def transferEvent : Nat := 78916
def frameStart : Nat := 78857
def rule : BoundRule := .identity (.predecessor 0 78915 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78915 .coefficient)
      LeftBound78913.bound (LeftBound78913.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78913.derived selector witness)

def rawBound : CoeffClass := LeftBound78913.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound78913.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78916

namespace LeftBound78922
def owner : Owner := ⟨.program ⟨214⟩, ⟨15152⟩⟩
def transferEvent : Nat := 78922
def frameStart : Nat := 78857
def rule : BoundRule := .product (.predecessor 0 78920 .coefficient) (.predecessor 1 78921 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78920 .coefficient)
      LeftAuthority78918.bound (LeftAuthority78918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78918.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78921 .coefficient)
      LeftBound78916.bound (LeftBound78916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78916.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority78918.bound LeftBound78916.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78918.bound, LeftBound78916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority78918.actual selector witness) * (LeftBound78916.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78922

namespace LeftBound78930
def owner : Owner := ⟨.program ⟨214⟩, ⟨15153⟩⟩
def transferEvent : Nat := 78930
def frameStart : Nat := 78857
def rule : BoundRule := .sum [.predecessor 0 78928 .coefficient, .predecessor 1 78929 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78928 .coefficient)
      LeftAuthority78926.bound (LeftAuthority78926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78929 .coefficient)
      LeftBound78922.bound (LeftBound78922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78926.bound, LeftBound78922.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78926.bound, LeftBound78922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78926.actual selector witness, LeftBound78922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78930

namespace LeftBound78934
def owner : Owner := ⟨.program ⟨214⟩, ⟨26762⟩⟩
def transferEvent : Nat := 78934
def frameStart : Nat := 78857
def rule : BoundRule := .product (.predecessor 0 78932 .coefficient) (.predecessor 1 78933 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78932 .coefficient)
      LeftBound78930.bound (LeftBound78930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78933 .coefficient)
      LeftAuthority78907.bound (LeftAuthority78907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78907.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78907.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78930.bound LeftAuthority78907.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78930.bound, LeftAuthority78907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78930.actual selector witness) * (LeftAuthority78907.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78934

namespace LeftBound78945
def owner : Owner := ⟨.program ⟨214⟩, ⟨15206⟩⟩
def transferEvent : Nat := 78945
def frameStart : Nat := 78857
def rule : BoundRule := .product (.predecessor 0 78943 .coefficient) (.predecessor 1 78944 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78943 .coefficient)
      LeftAuthority78918.bound (LeftAuthority78918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78918.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78944 .coefficient)
      LeftAuthority78941.bound (LeftAuthority78941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78941.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78941.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority78918.bound LeftAuthority78941.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78918.bound, LeftAuthority78941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority78918.actual selector witness) * (LeftAuthority78941.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78945

namespace LeftBound78953
def owner : Owner := ⟨.program ⟨214⟩, ⟨15207⟩⟩
def transferEvent : Nat := 78953
def frameStart : Nat := 78857
def rule : BoundRule := .sum [.predecessor 0 78951 .coefficient, .predecessor 1 78952 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78951 .coefficient)
      LeftAuthority78949.bound (LeftAuthority78949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78949.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78952 .coefficient)
      LeftBound78945.bound (LeftBound78945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78945.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78949.bound, LeftBound78945.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78949.bound, LeftBound78945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78949.actual selector witness, LeftBound78945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78953

namespace LeftBound78957
def owner : Owner := ⟨.program ⟨214⟩, ⟨26767⟩⟩
def transferEvent : Nat := 78957
def frameStart : Nat := 78857
def rule : BoundRule := .sum [.predecessor 0 78955 .coefficient, .predecessor 1 78956 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78955 .coefficient)
      LeftBound78953.bound (LeftBound78953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78956 .coefficient)
      LeftBound78934.bound (LeftBound78934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78934.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78953.bound, LeftBound78934.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78953.bound, LeftBound78934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78953.actual selector witness, LeftBound78934.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78957

namespace LeftBound78970
def owner : Owner := ⟨.program ⟨214⟩, ⟨26764⟩⟩
def transferEvent : Nat := 78970
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 78968 .coefficient, .predecessor 1 78969 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78968 .coefficient)
      LeftBound78799.bound (LeftBound78799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78969 .coefficient)
      LeftBound78782.bound (LeftBound78782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78799.bound, LeftBound78782.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78799.bound, LeftBound78782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78799.actual selector witness, LeftBound78782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78970

namespace LeftBound78973
def owner : Owner := ⟨.program ⟨214⟩, ⟨26764⟩⟩
def transferEvent : Nat := 78973
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 78967 .summary, .result 78789 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78967 .summary)
      LeftBound78801.bound (LeftBound78801.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20607⟩⟩) (rawTerms := some (Proof.Events308.exact78967RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78789 .summary)
      LeftBound78784.bound (LeftBound78784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26763⟩⟩) (rawTerms := some (Proof.Events307.exact78789RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78784.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78801.bound, LeftBound78784.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78801.bound, LeftBound78784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78801.actual selector witness, LeftBound78784.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78973

namespace LeftBound78977
def owner : Owner := ⟨.program ⟨214⟩, ⟨26765⟩⟩
def transferEvent : Nat := 78977
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78975 .coefficient) (.predecessor 1 78976 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78975 .coefficient)
      LeftBound78970.bound (LeftBound78970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78976 .coefficient)
      LeftBound5818.bound (LeftBound5818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78970.bound LeftBound5818.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78970.bound, LeftBound5818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78970.actual selector witness) * (LeftBound5818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78977

namespace LeftBound78978
def owner : Owner := ⟨.program ⟨214⟩, ⟨26765⟩⟩
def transferEvent : Nat := 78978
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
end LeftBound78978

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
