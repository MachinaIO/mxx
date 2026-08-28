import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard556

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound81902
def owner : Owner := ⟨.program ⟨214⟩, ⟨9929⟩⟩
def transferEvent : Nat := 81902
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81900 .coefficient) (.predecessor 1 81901 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81900 .coefficient)
      LeftBound81896.bound (LeftBound81896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81901 .coefficient)
      LeftBound8505.bound (LeftBound8505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81896.bound LeftBound8505.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81896.bound, LeftBound8505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81896.actual selector witness) * (LeftBound8505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81902

namespace LeftBound81903
def owner : Owner := ⟨.program ⟨214⟩, ⟨9929⟩⟩
def transferEvent : Nat := 81903
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩ [⟨.result 8502 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8502 .coefficient)
      LeftAuthority8501.bound (LeftAuthority8501.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7870⟩⟩) (rawTerms := some (Proof.Events033.exact8502RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8501.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8501.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8501.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81903

namespace LeftBound81904
def owner : Owner := ⟨.program ⟨214⟩, ⟨9929⟩⟩
def transferEvent : Nat := 81904
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 81899 .summary) (.transfer 81903) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81899 .summary)
      LeftBound81897.bound (LeftBound81897.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9928⟩⟩) (rawTerms := some (Proof.Events319.exact81899RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81903)
      LeftBound81903.bound (LeftBound81903.actual selector witness) := by
  exact .transfer (LeftBound81903.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81897.bound LeftBound81903.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81897.bound, LeftBound81903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81897.actual selector witness) * (LeftBound81903.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81904

namespace LeftBound81912
def owner : Owner := ⟨.program ⟨214⟩, ⟨12573⟩⟩
def transferEvent : Nat := 81912
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81910 .coefficient, .predecessor 1 81911 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81910 .coefficient)
      LeftBound81902.bound (LeftBound81902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81911 .coefficient)
      LeftBound81874.bound (LeftBound81874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81902.bound, LeftBound81874.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81902.bound, LeftBound81874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81902.actual selector witness, LeftBound81874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81912

namespace LeftBound81914
def owner : Owner := ⟨.program ⟨214⟩, ⟨12573⟩⟩
def transferEvent : Nat := 81914
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 81909 .summary, .result 81879 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81909 .summary)
      LeftBound81904.bound (LeftBound81904.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9929⟩⟩) (rawTerms := some (Proof.Events319.exact81909RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81879 .summary)
      LeftBound81876.bound (LeftBound81876.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12572⟩⟩) (rawTerms := some (Proof.Events319.exact81879RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81876.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81904.bound, LeftBound81876.bound]
def bound : CoeffClass := .finite ⟨95455360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81904.bound, LeftBound81876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81904.actual selector witness, LeftBound81876.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81914

namespace LeftBound81918
def owner : Owner := ⟨.program ⟨214⟩, ⟨25451⟩⟩
def transferEvent : Nat := 81918
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81916 .coefficient) (.predecessor 1 81917 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81916 .coefficient)
      LeftBound81912.bound (LeftBound81912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81917 .coefficient)
      LeftAuthority81850.bound (LeftAuthority81850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81850.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81912.bound LeftAuthority81850.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81912.bound, LeftAuthority81850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81912.actual selector witness) * (LeftAuthority81850.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81918

namespace LeftBound81919
def owner : Owner := ⟨.program ⟨214⟩, ⟨25451⟩⟩
def transferEvent : Nat := 81919
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩ [⟨.result 81851 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81851 .coefficient)
      LeftAuthority81850.bound (LeftAuthority81850.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25450⟩⟩) (rawTerms := some (Proof.Events319.exact81851RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81850.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority81850.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81850.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81919

namespace LeftBound81920
def owner : Owner := ⟨.program ⟨214⟩, ⟨25451⟩⟩
def transferEvent : Nat := 81920
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 81915 .summary) (.transfer 81919) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81915 .summary)
      LeftBound81914.bound (LeftBound81914.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12573⟩⟩) (rawTerms := some (Proof.Events319.exact81915RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81919)
      LeftBound81919.bound (LeftBound81919.actual selector witness) := by
  exact .transfer (LeftBound81919.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81914.bound LeftBound81919.bound
def bound : CoeffClass := .finite ⟨350322698485760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81914.bound, LeftBound81919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81914.actual selector witness) * (LeftBound81919.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81920

namespace LeftBound81931
def owner : Owner := ⟨.program ⟨214⟩, ⟨19962⟩⟩
def transferEvent : Nat := 81931
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 81929 .coefficient) (.value (.predecessor 1 81930 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81929 .coefficient)
      LeftAuthority81927.bound (LeftAuthority81927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact81928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81927.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81930 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority81927.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81927.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81927.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81931

namespace LeftBound81935
def owner : Owner := ⟨.program ⟨214⟩, ⟨19963⟩⟩
def transferEvent : Nat := 81935
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81933 .coefficient) (.predecessor 1 81934 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81933 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81934 .coefficient)
      LeftBound81931.bound (LeftBound81931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact81932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81931.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound81931.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound81931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound81931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81935

namespace LeftBound81936
def owner : Owner := ⟨.program ⟨214⟩, ⟨19963⟩⟩
def transferEvent : Nat := 81936
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩ [⟨.result 81928 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81928 .coefficient)
      LeftAuthority81927.bound (LeftAuthority81927.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19960⟩⟩) (rawTerms := some (Proof.Events320.exact81928RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81927.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81927.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority81927.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81927.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81936

namespace LeftBound81937
def owner : Owner := ⟨.program ⟨214⟩, ⟨19963⟩⟩
def transferEvent : Nat := 81937
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 81936) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81936)
      LeftBound81936.bound (LeftBound81936.actual selector witness) := by
  exact .transfer (LeftBound81936.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound81936.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound81936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound81936.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81937

namespace LeftBound82016
def owner : Owner := ⟨.program ⟨214⟩, ⟨12567⟩⟩
def transferEvent : Nat := 82016
def frameStart : Nat := 81987
def rule : BoundRule := .product (.predecessor 0 82014 .coefficient) (.predecessor 1 82015 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82014 .coefficient)
      LeftAuthority82012.bound (LeftAuthority82012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82015 .coefficient)
      LeftAuthority82009.bound (LeftAuthority82009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82009.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority82012.bound LeftAuthority82009.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82012.bound, LeftAuthority82009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority82012.actual selector witness) * (LeftAuthority82009.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82016

namespace LeftBound82020
def owner : Owner := ⟨.program ⟨214⟩, ⟨12568⟩⟩
def transferEvent : Nat := 82020
def frameStart : Nat := 81987
def rule : BoundRule := .identity (.predecessor 0 82019 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82019 .coefficient)
      LeftBound82016.bound (LeftBound82016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82016.derived selector witness)

def rawBound : CoeffClass := LeftBound82016.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound82016.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82020

namespace LeftBound82037
def owner : Owner := ⟨.program ⟨214⟩, ⟨12662⟩⟩
def transferEvent : Nat := 82037
def frameStart : Nat := 81987
def rule : BoundRule := .sum [.predecessor 0 82035 .coefficient, .predecessor 1 82036 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82035 .coefficient)
      LeftBound82020.bound (LeftBound82020.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82036 .coefficient)
      LeftAuthority82033.bound (LeftAuthority82033.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority82033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82020.bound, LeftAuthority82033.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82020.bound, LeftAuthority82033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82020.actual selector witness, LeftAuthority82033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82037

namespace LeftBound82040
def owner : Owner := ⟨.program ⟨214⟩, ⟨12663⟩⟩
def transferEvent : Nat := 82040
def frameStart : Nat := 81987
def rule : BoundRule := .identity (.predecessor 0 82039 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82039 .coefficient)
      LeftBound82037.bound (LeftBound82037.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82037.derived selector witness)

def rawBound : CoeffClass := LeftBound82037.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound82037.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82040

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
