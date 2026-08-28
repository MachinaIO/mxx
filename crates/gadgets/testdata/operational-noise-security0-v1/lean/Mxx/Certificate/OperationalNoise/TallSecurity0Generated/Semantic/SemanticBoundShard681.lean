import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard680

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99208
def owner : Owner := ⟨.program ⟨214⟩, ⟨13970⟩⟩
def transferEvent : Nat := 99208
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99203 .summary) (.transfer 99207) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99203 .summary)
      LeftBound99201.bound (LeftBound99201.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13969⟩⟩) (rawTerms := some (Proof.Events387.exact99203RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99207)
      LeftBound99207.bound (LeftBound99207.actual selector witness) := by
  exact .transfer (LeftBound99207.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99201.bound LeftBound99207.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99201.bound, LeftBound99207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99201.actual selector witness) * (LeftBound99207.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99208

namespace LeftBound99216
def owner : Owner := ⟨.program ⟨214⟩, ⟨13971⟩⟩
def transferEvent : Nat := 99216
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99214 .coefficient, .predecessor 1 99215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99214 .coefficient)
      LeftBound99206.bound (LeftBound99206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99215 .coefficient)
      LeftBound99178.bound (LeftBound99178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99206.bound, LeftBound99178.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99206.bound, LeftBound99178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99206.actual selector witness, LeftBound99178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99216

namespace LeftBound99218
def owner : Owner := ⟨.program ⟨214⟩, ⟨13971⟩⟩
def transferEvent : Nat := 99218
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99213 .summary, .result 99183 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99213 .summary)
      LeftBound99208.bound (LeftBound99208.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13970⟩⟩) (rawTerms := some (Proof.Events387.exact99213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99183 .summary)
      LeftBound99180.bound (LeftBound99180.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13966⟩⟩) (rawTerms := some (Proof.Events387.exact99183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99180.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99208.bound, LeftBound99180.bound]
def bound : CoeffClass := .finite ⟨95433728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99208.bound, LeftBound99180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99208.actual selector witness, LeftBound99180.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99218

namespace LeftBound99222
def owner : Owner := ⟨.program ⟨214⟩, ⟨25977⟩⟩
def transferEvent : Nat := 99222
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99220 .coefficient) (.predecessor 1 99221 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99220 .coefficient)
      LeftBound99216.bound (LeftBound99216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99221 .coefficient)
      LeftAuthority99154.bound (LeftAuthority99154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99154.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99216.bound LeftAuthority99154.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99216.bound, LeftAuthority99154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99216.actual selector witness) * (LeftAuthority99154.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99222

namespace LeftBound99223
def owner : Owner := ⟨.program ⟨214⟩, ⟨25977⟩⟩
def transferEvent : Nat := 99223
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25976⟩⟩]⟩ [⟨.result 99155 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99155 .coefficient)
      LeftAuthority99154.bound (LeftAuthority99154.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25976⟩⟩) (rawTerms := some (Proof.Events387.exact99155RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99154.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority99154.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99154.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99223

namespace LeftBound99224
def owner : Owner := ⟨.program ⟨214⟩, ⟨25977⟩⟩
def transferEvent : Nat := 99224
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99219 .summary) (.transfer 99223) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99219 .summary)
      LeftBound99218.bound (LeftBound99218.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13971⟩⟩) (rawTerms := some (Proof.Events387.exact99219RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99223)
      LeftBound99223.bound (LeftBound99223.actual selector witness) := by
  exact .transfer (LeftBound99223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99218.bound LeftBound99223.bound
def bound : CoeffClass := .finite ⟨350243308699648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99218.bound, LeftBound99223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99218.actual selector witness) * (LeftBound99223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99224

namespace LeftBound99235
def owner : Owner := ⟨.program ⟨214⟩, ⟨19447⟩⟩
def transferEvent : Nat := 99235
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 99233 .coefficient) (.value (.predecessor 1 99234 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99233 .coefficient)
      LeftAuthority99231.bound (LeftAuthority99231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99231.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99234 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority99231.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99231.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99231.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound99235

namespace LeftBound99239
def owner : Owner := ⟨.program ⟨214⟩, ⟨19448⟩⟩
def transferEvent : Nat := 99239
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99237 .coefficient) (.predecessor 1 99238 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99237 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99238 .coefficient)
      LeftBound99235.bound (LeftBound99235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound99235.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound99235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound99235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99239

namespace LeftBound99240
def owner : Owner := ⟨.program ⟨214⟩, ⟨19448⟩⟩
def transferEvent : Nat := 99240
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19445⟩⟩]⟩ [⟨.result 99232 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99232 .coefficient)
      LeftAuthority99231.bound (LeftAuthority99231.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19445⟩⟩) (rawTerms := some (Proof.Events387.exact99232RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99231.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99231.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority99231.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99231.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99240

namespace LeftBound99241
def owner : Owner := ⟨.program ⟨214⟩, ⟨19448⟩⟩
def transferEvent : Nat := 99241
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 99240) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99240)
      LeftBound99240.bound (LeftBound99240.actual selector witness) := by
  exact .transfer (LeftBound99240.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound99240.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound99240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound99240.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99241

namespace LeftBound99296
def owner : Owner := ⟨.program ⟨214⟩, ⟨13964⟩⟩
def transferEvent : Nat := 99296
def frameStart : Nat := 99279
def rule : BoundRule := .product (.predecessor 0 99294 .coefficient) (.predecessor 1 99295 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99294 .coefficient)
      LeftAuthority99292.bound (LeftAuthority99292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99292.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99292.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99295 .coefficient)
      LeftAuthority99289.bound (LeftAuthority99289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority99292.bound LeftAuthority99289.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99292.bound, LeftAuthority99289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority99292.actual selector witness) * (LeftAuthority99289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99296

namespace LeftBound99300
def owner : Owner := ⟨.program ⟨214⟩, ⟨13965⟩⟩
def transferEvent : Nat := 99300
def frameStart : Nat := 99279
def rule : BoundRule := .identity (.predecessor 0 99299 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99299 .coefficient)
      LeftBound99296.bound (LeftBound99296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99296.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99296.derived selector witness)

def rawBound : CoeffClass := LeftBound99296.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99296.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound99296.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99300

namespace LeftBound99317
def owner : Owner := ⟨.program ⟨214⟩, ⟨14089⟩⟩
def transferEvent : Nat := 99317
def frameStart : Nat := 99279
def rule : BoundRule := .sum [.predecessor 0 99315 .coefficient, .predecessor 1 99316 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99315 .coefficient)
      LeftBound99300.bound (LeftBound99300.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99316 .coefficient)
      LeftAuthority99313.bound (LeftAuthority99313.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority99313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99300.bound, LeftAuthority99313.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99300.bound, LeftAuthority99313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99300.actual selector witness, LeftAuthority99313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99317

namespace LeftBound99320
def owner : Owner := ⟨.program ⟨214⟩, ⟨14090⟩⟩
def transferEvent : Nat := 99320
def frameStart : Nat := 99279
def rule : BoundRule := .identity (.predecessor 0 99319 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99319 .coefficient)
      LeftBound99317.bound (LeftBound99317.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99317.derived selector witness)

def rawBound : CoeffClass := LeftBound99317.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound99317.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99320

namespace LeftBound99326
def owner : Owner := ⟨.program ⟨214⟩, ⟨14091⟩⟩
def transferEvent : Nat := 99326
def frameStart : Nat := 99279
def rule : BoundRule := .product (.predecessor 0 99324 .coefficient) (.predecessor 1 99325 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99324 .coefficient)
      LeftAuthority99322.bound (LeftAuthority99322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99325 .coefficient)
      LeftBound99320.bound (LeftBound99320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99320.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority99322.bound LeftBound99320.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99322.bound, LeftBound99320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority99322.actual selector witness) * (LeftBound99320.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99326

namespace LeftBound99342
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def transferEvent : Nat := 99342
def frameStart : Nat := 99279
def rule : BoundRule := .scale (.predecessor 0 99340 .coefficient) (.value (.predecessor 1 99341 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99340 .coefficient)
      LeftAuthority99338.bound (LeftAuthority99338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99338.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99341 .coefficient)
      LeftAuthority99329.bound (LeftAuthority99329.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority99329.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority99338.bound LeftAuthority99329.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99338.bound, LeftAuthority99329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99338.actual selector witness) * (LeftAuthority99329.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound99342

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
