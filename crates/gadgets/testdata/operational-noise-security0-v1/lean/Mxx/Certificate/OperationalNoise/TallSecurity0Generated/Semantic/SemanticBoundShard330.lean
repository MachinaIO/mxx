import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard301
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard329

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound49907
def owner : Owner := ⟨.program ⟨214⟩, ⟨15060⟩⟩
def transferEvent : Nat := 49907
def frameStart : Nat := 49819
def rule : BoundRule := .product (.predecessor 0 49905 .coefficient) (.predecessor 1 49906 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49905 .coefficient)
      LeftAuthority49880.bound (LeftAuthority49880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49880.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49906 .coefficient)
      LeftAuthority49903.bound (LeftAuthority49903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49903.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49903.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority49880.bound LeftAuthority49903.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49880.bound, LeftAuthority49903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority49880.actual selector witness) * (LeftAuthority49903.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49907

namespace LeftBound49915
def owner : Owner := ⟨.program ⟨214⟩, ⟨15061⟩⟩
def transferEvent : Nat := 49915
def frameStart : Nat := 49819
def rule : BoundRule := .sum [.predecessor 0 49913 .coefficient, .predecessor 1 49914 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49913 .coefficient)
      LeftAuthority49911.bound (LeftAuthority49911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49911.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49914 .coefficient)
      LeftBound49907.bound (LeftBound49907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49911.bound, LeftBound49907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49911.bound, LeftBound49907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49911.actual selector witness, LeftBound49907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49915

namespace LeftBound49919
def owner : Owner := ⟨.program ⟨214⟩, ⟨26589⟩⟩
def transferEvent : Nat := 49919
def frameStart : Nat := 49819
def rule : BoundRule := .sum [.predecessor 0 49917 .coefficient, .predecessor 1 49918 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49917 .coefficient)
      LeftBound49915.bound (LeftBound49915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49915.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49918 .coefficient)
      LeftBound49896.bound (LeftBound49896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49896.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49915.bound, LeftBound49896.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49915.bound, LeftBound49896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49915.actual selector witness, LeftBound49896.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49919

namespace LeftBound49932
def owner : Owner := ⟨.program ⟨214⟩, ⟨26586⟩⟩
def transferEvent : Nat := 49932
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 49930 .coefficient, .predecessor 1 49931 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49930 .coefficient)
      LeftBound49761.bound (LeftBound49761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact49929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49931 .coefficient)
      LeftBound49744.bound (LeftBound49744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49744.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49761.bound, LeftBound49744.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49761.bound, LeftBound49744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49761.actual selector witness, LeftBound49744.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49932

namespace LeftBound49935
def owner : Owner := ⟨.program ⟨214⟩, ⟨26586⟩⟩
def transferEvent : Nat := 49935
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 49929 .summary, .result 49751 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49929 .summary)
      LeftBound49763.bound (LeftBound49763.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20475⟩⟩) (rawTerms := some (Proof.Events195.exact49929RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49751 .summary)
      LeftBound49746.bound (LeftBound49746.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26585⟩⟩) (rawTerms := some (Proof.Events194.exact49751RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49746.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49763.bound, LeftBound49746.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49763.bound, LeftBound49746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49763.actual selector witness, LeftBound49746.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49935

namespace LeftBound49939
def owner : Owner := ⟨.program ⟨214⟩, ⟨26587⟩⟩
def transferEvent : Nat := 49939
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49937 .coefficient) (.predecessor 1 49938 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49937 .coefficient)
      LeftBound49932.bound (LeftBound49932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact49936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49932.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49938 .coefficient)
      LeftBound5838.bound (LeftBound5838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49932.bound LeftBound5838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49932.bound, LeftBound5838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49932.actual selector witness) * (LeftBound5838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49939

namespace LeftBound49940
def owner : Owner := ⟨.program ⟨214⟩, ⟨26587⟩⟩
def transferEvent : Nat := 49940
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩ [⟨.result 5835 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5835 .coefficient)
      LeftAuthority5834.bound (LeftAuthority5834.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6671⟩⟩) (rawTerms := some (Proof.Events022.exact5835RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5834.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5834.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5834.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49940

namespace LeftBound49941
def owner : Owner := ⟨.program ⟨214⟩, ⟨26587⟩⟩
def transferEvent : Nat := 49941
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 49936 .summary) (.transfer 49940) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49936 .summary)
      LeftBound49935.bound (LeftBound49935.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26586⟩⟩) (rawTerms := some (Proof.Events195.exact49936RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49940)
      LeftBound49940.bound (LeftBound49940.actual selector witness) := by
  exact .transfer (LeftBound49940.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49935.bound LeftBound49940.bound
def bound : CoeffClass := .finite ⟨4741295067215179835091451904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49935.bound, LeftBound49940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49935.actual selector witness) * (LeftBound49940.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49941

namespace LeftBound49956
def owner : Owner := ⟨.program ⟨214⟩, ⟨26377⟩⟩
def transferEvent : Nat := 49956
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49954 .coefficient) (.predecessor 1 49955 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49954 .coefficient)
      LeftBound44513.bound (LeftBound44513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49955 .coefficient)
      LeftAuthority49952.bound (LeftAuthority49952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact49953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49952.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44513.bound LeftAuthority49952.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44513.bound, LeftAuthority49952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44513.actual selector witness) * (LeftAuthority49952.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49956

namespace LeftBound49957
def owner : Owner := ⟨.program ⟨214⟩, ⟨26377⟩⟩
def transferEvent : Nat := 49957
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26375⟩⟩]⟩ [⟨.result 49953 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49953 .coefficient)
      LeftAuthority49952.bound (LeftAuthority49952.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26375⟩⟩) (rawTerms := some (Proof.Events195.exact49953RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49952.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority49952.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49952.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49957

namespace LeftBound49958
def owner : Owner := ⟨.program ⟨214⟩, ⟨26377⟩⟩
def transferEvent : Nat := 49958
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 44517 .summary) (.transfer 49957) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44517 .summary)
      LeftBound44516.bound (LeftBound44516.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24923⟩⟩) (rawTerms := some (Proof.Events173.exact44517RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49957)
      LeftBound49957.bound (LeftBound49957.actual selector witness) := by
  exact .transfer (LeftBound49957.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44516.bound LeftBound49957.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44516.bound, LeftBound49957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44516.actual selector witness) * (LeftBound49957.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49958

namespace LeftBound49969
def owner : Owner := ⟨.program ⟨214⟩, ⟨20330⟩⟩
def transferEvent : Nat := 49969
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 49967 .coefficient) (.value (.predecessor 1 49968 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49967 .coefficient)
      LeftAuthority49965.bound (LeftAuthority49965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact49966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49965.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49968 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority49965.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49965.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49965.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound49969

namespace LeftBound49973
def owner : Owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩
def transferEvent : Nat := 49973
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49971 .coefficient) (.predecessor 1 49972 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49971 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49972 .coefficient)
      LeftBound49969.bound (LeftBound49969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact49970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49969.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49969.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound49969.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound49969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound49969.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49973

namespace LeftBound49974
def owner : Owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩
def transferEvent : Nat := 49974
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩ [⟨.result 49966 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49966 .coefficient)
      LeftAuthority49965.bound (LeftAuthority49965.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20328⟩⟩) (rawTerms := some (Proof.Events195.exact49966RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49965.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49965.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority49965.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49965.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49974

namespace LeftBound49975
def owner : Owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩
def transferEvent : Nat := 49975
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 49974) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49974)
      LeftBound49974.bound (LeftBound49974.actual selector witness) := by
  exact .transfer (LeftBound49974.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound49974.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound49974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound49974.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49975

namespace LeftBound50070
def owner : Owner := ⟨.program ⟨214⟩, ⟨14801⟩⟩
def transferEvent : Nat := 50070
def frameStart : Nat := 50031
def rule : BoundRule := .identity (.predecessor 0 50069 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50069 .coefficient)
      LeftAuthority50067.bound (LeftAuthority50067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact50068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50067.derived selector witness)

def rawBound : CoeffClass := LeftAuthority50067.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority50067.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50070

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
