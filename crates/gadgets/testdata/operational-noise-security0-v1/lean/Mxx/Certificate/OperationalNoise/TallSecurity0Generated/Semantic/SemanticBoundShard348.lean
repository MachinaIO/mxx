import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard347

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound51859
def owner : Owner := ⟨.program ⟨214⟩, ⟨6768⟩⟩
def transferEvent : Nat := 51859
def frameStart : Nat := 51781
def rule : BoundRule := .identity (.predecessor 0 51858 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51858 .coefficient)
      LeftAuthority51846.bound (LeftAuthority51846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51846.derived selector witness)

def rawBound : CoeffClass := LeftAuthority51846.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority51846.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51859

namespace LeftBound51863
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def transferEvent : Nat := 51863
def frameStart : Nat := 51781
def rule : BoundRule := .product (.predecessor 0 51861 .coefficient) (.predecessor 1 51862 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51861 .coefficient)
      LeftBound51859.bound (LeftBound51859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51862 .coefficient)
      LeftBound51856.bound (LeftBound51856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51859.bound LeftBound51856.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51859.bound, LeftBound51856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51859.actual selector witness) * (LeftBound51856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51863

namespace LeftBound51868
def owner : Owner := ⟨.program ⟨214⟩, ⟨13061⟩⟩
def transferEvent : Nat := 51868
def frameStart : Nat := 51781
def rule : BoundRule := .sum [.predecessor 0 51866 .coefficient, .predecessor 1 51867 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51866 .coefficient)
      LeftBound51863.bound (LeftBound51863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51867 .coefficient)
      LeftBound51840.bound (LeftBound51840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51863.bound, LeftBound51840.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51863.bound, LeftBound51840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51863.actual selector witness, LeftBound51840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51868

namespace LeftBound51872
def owner : Owner := ⟨.program ⟨214⟩, ⟨25612⟩⟩
def transferEvent : Nat := 51872
def frameStart : Nat := 51781
def rule : BoundRule := .product (.predecessor 0 51870 .coefficient) (.predecessor 1 51871 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51870 .coefficient)
      LeftBound51868.bound (LeftBound51868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51871 .coefficient)
      LeftAuthority51825.bound (LeftAuthority51825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51825.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51825.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51868.bound LeftAuthority51825.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51868.bound, LeftAuthority51825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51868.actual selector witness) * (LeftAuthority51825.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51872

namespace LeftBound51883
def owner : Owner := ⟨.program ⟨214⟩, ⟨16758⟩⟩
def transferEvent : Nat := 51883
def frameStart : Nat := 51781
def rule : BoundRule := .product (.predecessor 0 51881 .coefficient) (.predecessor 1 51882 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51881 .coefficient)
      LeftAuthority51836.bound (LeftAuthority51836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51882 .coefficient)
      LeftAuthority51879.bound (LeftAuthority51879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51879.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51879.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority51836.bound LeftAuthority51879.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51836.bound, LeftAuthority51879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority51836.actual selector witness) * (LeftAuthority51879.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51883

namespace LeftBound51891
def owner : Owner := ⟨.program ⟨214⟩, ⟨16759⟩⟩
def transferEvent : Nat := 51891
def frameStart : Nat := 51781
def rule : BoundRule := .sum [.predecessor 0 51889 .coefficient, .predecessor 1 51890 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51889 .coefficient)
      LeftAuthority51887.bound (LeftAuthority51887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51890 .coefficient)
      LeftBound51883.bound (LeftBound51883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51883.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority51887.bound, LeftBound51883.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51887.bound, LeftBound51883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority51887.actual selector witness, LeftBound51883.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51891

namespace LeftBound51895
def owner : Owner := ⟨.program ⟨214⟩, ⟨25613⟩⟩
def transferEvent : Nat := 51895
def frameStart : Nat := 51781
def rule : BoundRule := .sum [.predecessor 0 51893 .coefficient, .predecessor 1 51894 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51893 .coefficient)
      LeftBound51891.bound (LeftBound51891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51894 .coefficient)
      LeftBound51872.bound (LeftBound51872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51891.bound, LeftBound51872.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51891.bound, LeftBound51872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51891.actual selector witness, LeftBound51872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51895

namespace LeftBound51908
def owner : Owner := ⟨.program ⟨214⟩, ⟨25611⟩⟩
def transferEvent : Nat := 51908
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51906 .coefficient, .predecessor 1 51907 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51906 .coefficient)
      LeftBound51729.bound (LeftBound51729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51907 .coefficient)
      LeftBound51712.bound (LeftBound51712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51729.bound, LeftBound51712.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51729.bound, LeftBound51712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51729.actual selector witness, LeftBound51712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51908

namespace LeftBound51911
def owner : Owner := ⟨.program ⟨214⟩, ⟨25611⟩⟩
def transferEvent : Nat := 51911
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 51905 .summary, .result 51719 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51905 .summary)
      LeftBound51731.bound (LeftBound51731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20111⟩⟩) (rawTerms := some (Proof.Events202.exact51905RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51719 .summary)
      LeftBound51714.bound (LeftBound51714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25610⟩⟩) (rawTerms := some (Proof.Events202.exact51719RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51714.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51731.bound, LeftBound51714.bound]
def bound : CoeffClass := .finite ⟨352164536528896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51731.bound, LeftBound51714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51731.actual selector witness, LeftBound51714.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51911

namespace LeftBound51915
def owner : Owner := ⟨.program ⟨214⟩, ⟨29617⟩⟩
def transferEvent : Nat := 51915
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51913 .coefficient) (.predecessor 1 51914 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51913 .coefficient)
      LeftBound51908.bound (LeftBound51908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51914 .coefficient)
      LeftAuthority51634.bound (LeftAuthority51634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51634.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51908.bound LeftAuthority51634.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51908.bound, LeftAuthority51634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51908.actual selector witness) * (LeftAuthority51634.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51915

namespace LeftBound51916
def owner : Owner := ⟨.program ⟨214⟩, ⟨29617⟩⟩
def transferEvent : Nat := 51916
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29615⟩⟩]⟩ [⟨.result 51635 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51635 .coefficient)
      LeftAuthority51634.bound (LeftAuthority51634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29615⟩⟩) (rawTerms := some (Proof.Events201.exact51635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority51634.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51634.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51916

namespace LeftBound51917
def owner : Owner := ⟨.program ⟨214⟩, ⟨29617⟩⟩
def transferEvent : Nat := 51917
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51912 .summary) (.transfer 51916) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51912 .summary)
      LeftBound51911.bound (LeftBound51911.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25611⟩⟩) (rawTerms := some (Proof.Events202.exact51912RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51916)
      LeftBound51916.bound (LeftBound51916.actual selector witness) := by
  exact .transfer (LeftBound51916.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51911.bound LeftBound51916.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51911.bound, LeftBound51916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51911.actual selector witness) * (LeftBound51916.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51917

namespace LeftBound51928
def owner : Owner := ⟨.program ⟨214⟩, ⟨22558⟩⟩
def transferEvent : Nat := 51928
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 51926 .coefficient) (.value (.predecessor 1 51927 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51926 .coefficient)
      LeftAuthority51924.bound (LeftAuthority51924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51927 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority51924.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51924.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51924.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51928

namespace LeftBound51932
def owner : Owner := ⟨.program ⟨214⟩, ⟨22559⟩⟩
def transferEvent : Nat := 51932
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51930 .coefficient) (.predecessor 1 51931 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51930 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51931 .coefficient)
      LeftBound51928.bound (LeftBound51928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51928.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound51928.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound51928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound51928.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51932

namespace LeftBound51933
def owner : Owner := ⟨.program ⟨214⟩, ⟨22559⟩⟩
def transferEvent : Nat := 51933
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22556⟩⟩]⟩ [⟨.result 51925 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51925 .coefficient)
      LeftAuthority51924.bound (LeftAuthority51924.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22556⟩⟩) (rawTerms := some (Proof.Events202.exact51925RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51924.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority51924.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51924.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51933

namespace LeftBound51934
def owner : Owner := ⟨.program ⟨214⟩, ⟨22559⟩⟩
def transferEvent : Nat := 51934
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 51933) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51933)
      LeftBound51933.bound (LeftBound51933.actual selector witness) := by
  exact .transfer (LeftBound51933.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound51933.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound51933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound51933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51934

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
