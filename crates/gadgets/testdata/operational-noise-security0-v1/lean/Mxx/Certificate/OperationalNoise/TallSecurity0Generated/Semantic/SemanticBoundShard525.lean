import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard482
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard524

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound77885
def owner : Owner := ⟨.program ⟨214⟩, ⟨17163⟩⟩
def transferEvent : Nat := 77885
def frameStart : Nat := 77797
def rule : BoundRule := .product (.predecessor 0 77883 .coefficient) (.predecessor 1 77884 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77883 .coefficient)
      LeftAuthority77858.bound (LeftAuthority77858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77884 .coefficient)
      LeftAuthority77881.bound (LeftAuthority77881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77881.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority77858.bound LeftAuthority77881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77858.bound, LeftAuthority77881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority77858.actual selector witness) * (LeftAuthority77881.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77885

namespace LeftBound77893
def owner : Owner := ⟨.program ⟨214⟩, ⟨17164⟩⟩
def transferEvent : Nat := 77893
def frameStart : Nat := 77797
def rule : BoundRule := .sum [.predecessor 0 77891 .coefficient, .predecessor 1 77892 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77891 .coefficient)
      LeftAuthority77889.bound (LeftAuthority77889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77892 .coefficient)
      LeftBound77885.bound (LeftBound77885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77885.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77889.bound, LeftBound77885.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77889.bound, LeftBound77885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77889.actual selector witness, LeftBound77885.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77893

namespace LeftBound77897
def owner : Owner := ⟨.program ⟨214⟩, ⟨27852⟩⟩
def transferEvent : Nat := 77897
def frameStart : Nat := 77797
def rule : BoundRule := .sum [.predecessor 0 77895 .coefficient, .predecessor 1 77896 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77895 .coefficient)
      LeftBound77893.bound (LeftBound77893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77896 .coefficient)
      LeftBound77874.bound (LeftBound77874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77893.bound, LeftBound77874.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77893.bound, LeftBound77874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77893.actual selector witness, LeftBound77874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77897

namespace LeftBound77910
def owner : Owner := ⟨.program ⟨214⟩, ⟨27849⟩⟩
def transferEvent : Nat := 77910
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77908 .coefficient, .predecessor 1 77909 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77908 .coefficient)
      LeftBound77739.bound (LeftBound77739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77909 .coefficient)
      LeftBound77722.bound (LeftBound77722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77739.bound, LeftBound77722.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77739.bound, LeftBound77722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77739.actual selector witness, LeftBound77722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77910

namespace LeftBound77913
def owner : Owner := ⟨.program ⟨214⟩, ⟨27849⟩⟩
def transferEvent : Nat := 77913
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 77907 .summary, .result 77729 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77907 .summary)
      LeftBound77741.bound (LeftBound77741.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21327⟩⟩) (rawTerms := some (Proof.Events304.exact77907RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77729 .summary)
      LeftBound77724.bound (LeftBound77724.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27848⟩⟩) (rawTerms := some (Proof.Events303.exact77729RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77724.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77741.bound, LeftBound77724.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77741.bound, LeftBound77724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77741.actual selector witness, LeftBound77724.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77913

namespace LeftBound77917
def owner : Owner := ⟨.program ⟨214⟩, ⟨27850⟩⟩
def transferEvent : Nat := 77917
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77915 .coefficient) (.predecessor 1 77916 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77915 .coefficient)
      LeftBound77910.bound (LeftBound77910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77916 .coefficient)
      LeftBound5718.bound (LeftBound5718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77910.bound LeftBound5718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77910.bound, LeftBound5718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77910.actual selector witness) * (LeftBound5718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77917

namespace LeftBound77918
def owner : Owner := ⟨.program ⟨214⟩, ⟨27850⟩⟩
def transferEvent : Nat := 77918
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩ [⟨.result 5715 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5715 .coefficient)
      LeftAuthority5714.bound (LeftAuthority5714.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6641⟩⟩) (rawTerms := some (Proof.Events022.exact5715RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5714.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5714.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5714.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77918

namespace LeftBound77919
def owner : Owner := ⟨.program ⟨214⟩, ⟨27850⟩⟩
def transferEvent : Nat := 77919
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 77914 .summary) (.transfer 77918) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77914 .summary)
      LeftBound77913.bound (LeftBound77913.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27849⟩⟩) (rawTerms := some (Proof.Events304.exact77914RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77918)
      LeftBound77918.bound (LeftBound77918.actual selector witness) := by
  exact .transfer (LeftBound77918.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77913.bound LeftBound77918.bound
def bound : CoeffClass := .finite ⟨4741911972453864866771369984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77913.bound, LeftBound77918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77913.actual selector witness) * (LeftBound77918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77919

namespace LeftBound77934
def owner : Owner := ⟨.program ⟨214⟩, ⟨27631⟩⟩
def transferEvent : Nat := 77934
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77932 .coefficient) (.predecessor 1 77933 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77932 .coefficient)
      LeftBound70871.bound (LeftBound70871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77933 .coefficient)
      LeftAuthority77930.bound (LeftAuthority77930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77930.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77930.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70871.bound LeftAuthority77930.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70871.bound, LeftAuthority77930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70871.actual selector witness) * (LeftAuthority77930.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77934

namespace LeftBound77935
def owner : Owner := ⟨.program ⟨214⟩, ⟨27631⟩⟩
def transferEvent : Nat := 77935
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩ [⟨.result 77931 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77931 .coefficient)
      LeftAuthority77930.bound (LeftAuthority77930.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27629⟩⟩) (rawTerms := some (Proof.Events304.exact77931RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77930.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77930.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77930.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77930.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77935

namespace LeftBound77936
def owner : Owner := ⟨.program ⟨214⟩, ⟨27631⟩⟩
def transferEvent : Nat := 77936
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70875 .summary) (.transfer 77935) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70875 .summary)
      LeftBound70874.bound (LeftBound70874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25986⟩⟩) (rawTerms := some (Proof.Events276.exact70875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77935)
      LeftBound77935.bound (LeftBound77935.actual selector witness) := by
  exact .transfer (LeftBound77935.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70874.bound LeftBound77935.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70874.bound, LeftBound77935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70874.actual selector witness) * (LeftBound77935.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77936

namespace LeftBound77947
def owner : Owner := ⟨.program ⟨214⟩, ⟨21182⟩⟩
def transferEvent : Nat := 77947
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 77945 .coefficient) (.value (.predecessor 1 77946 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77945 .coefficient)
      LeftAuthority77943.bound (LeftAuthority77943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77946 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority77943.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77943.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77943.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound77947

namespace LeftBound77951
def owner : Owner := ⟨.program ⟨214⟩, ⟨21183⟩⟩
def transferEvent : Nat := 77951
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77949 .coefficient) (.predecessor 1 77950 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77949 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77950 .coefficient)
      LeftBound77947.bound (LeftBound77947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77947.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound77947.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound77947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound77947.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77951

namespace LeftBound77952
def owner : Owner := ⟨.program ⟨214⟩, ⟨21183⟩⟩
def transferEvent : Nat := 77952
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩ [⟨.result 77944 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77944 .coefficient)
      LeftAuthority77943.bound (LeftAuthority77943.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21180⟩⟩) (rawTerms := some (Proof.Events304.exact77944RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77943.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77943.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77943.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77952

namespace LeftBound77953
def owner : Owner := ⟨.program ⟨214⟩, ⟨21183⟩⟩
def transferEvent : Nat := 77953
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 77952) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77952)
      LeftBound77952.bound (LeftBound77952.actual selector witness) := by
  exact .transfer (LeftBound77952.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound77952.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound77952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound77952.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77953

namespace LeftBound78048
def owner : Owner := ⟨.program ⟨214⟩, ⟨15818⟩⟩
def transferEvent : Nat := 78048
def frameStart : Nat := 78009
def rule : BoundRule := .identity (.predecessor 0 78047 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78047 .coefficient)
      LeftAuthority78045.bound (LeftAuthority78045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact78046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78045.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78045.derived selector witness)

def rawBound : CoeffClass := LeftAuthority78045.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority78045.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78048

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
