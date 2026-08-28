import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard654

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96140
def owner : Owner := ⟨.program ⟨214⟩, ⟨12548⟩⟩
def transferEvent : Nat := 96140
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96138 .coefficient) (.predecessor 1 96139 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96138 .coefficient)
      LeftBound96134.bound (LeftBound96134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96139 .coefficient)
      LeftAuthority4660.bound (LeftAuthority4660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4660.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4660.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound96134.bound LeftAuthority4660.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96134.bound, LeftAuthority4660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound96134.actual selector witness) * (LeftAuthority4660.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96140

namespace LeftBound96141
def owner : Owner := ⟨.program ⟨214⟩, ⟨12548⟩⟩
def transferEvent : Nat := 96141
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩], []⟩ [⟨.result 4661 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4661 .coefficient)
      LeftAuthority4660.bound (LeftAuthority4660.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9910⟩⟩) (rawTerms := some (Proof.Events018.exact4661RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4660.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4660.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4660.bound []
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4660.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96141

namespace LeftBound96142
def owner : Owner := ⟨.program ⟨214⟩, ⟨12548⟩⟩
def transferEvent : Nat := 96142
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96137 .summary) (.transfer 96141) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96137 .summary)
      LeftBound96135.bound (LeftBound96135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12547⟩⟩) (rawTerms := some (Proof.Events375.exact96137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96141)
      LeftBound96141.bound (LeftBound96141.actual selector witness) := by
  exact .transfer (LeftBound96141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound96135.bound LeftBound96141.bound
def bound : CoeffClass := .finite ⟨34944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96135.bound, LeftBound96141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound96135.actual selector witness) * (LeftBound96141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96142

namespace LeftBound96148
def owner : Owner := ⟨.program ⟨214⟩, ⟨9911⟩⟩
def transferEvent : Nat := 96148
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 96146 .coefficient) (.predecessor 1 96147 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96146 .coefficient)
      LeftAuthority4660.bound (LeftAuthority4660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4660.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96147 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4660.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4660.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4660.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96148

namespace LeftBound96153
def owner : Owner := ⟨.program ⟨214⟩, ⟨7103⟩⟩
def transferEvent : Nat := 96153
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96151 .coefficient) (.predecessor 1 96152 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96151 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96152 .coefficient)
      LeftBound8516.bound (LeftBound8516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound8516.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound8516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound8516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96153

namespace LeftBound96158
def owner : Owner := ⟨.program ⟨214⟩, ⟨9912⟩⟩
def transferEvent : Nat := 96158
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96156 .coefficient, .predecessor 1 96157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96156 .coefficient)
      LeftBound96153.bound (LeftBound96153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96153.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96157 .coefficient)
      LeftBound96148.bound (LeftBound96148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96153.bound, LeftBound96148.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96153.bound, LeftBound96148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96153.actual selector witness, LeftBound96148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96158

namespace LeftBound96162
def owner : Owner := ⟨.program ⟨214⟩, ⟨9913⟩⟩
def transferEvent : Nat := 96162
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96160 .coefficient, .predecessor 1 96161 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96160 .coefficient)
      LeftBound96158.bound (LeftBound96158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96161 .coefficient)
      LeftBound8508.bound (LeftBound8508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96158.bound, LeftBound8508.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96158.bound, LeftBound8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96158.actual selector witness, LeftBound8508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96162

namespace LeftBound96163
def owner : Owner := ⟨.program ⟨214⟩, ⟨9913⟩⟩
def transferEvent : Nat := 96163
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨80⟩⟩]⟩ [⟨.result 8509 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8509 .coefficient)
      LeftBound8508.bound (LeftBound8508.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨80⟩⟩) (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8508.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8508.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96163

namespace LeftBound96168
def owner : Owner := ⟨.program ⟨214⟩, ⟨9914⟩⟩
def transferEvent : Nat := 96168
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96166 .coefficient) (.predecessor 1 96167 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96166 .coefficient)
      LeftBound96162.bound (LeftBound96162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96167 .coefficient)
      LeftBound8505.bound (LeftBound8505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96162.bound LeftBound8505.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96162.bound, LeftBound8505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96162.actual selector witness) * (LeftBound8505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96168

namespace LeftBound96169
def owner : Owner := ⟨.program ⟨214⟩, ⟨9914⟩⟩
def transferEvent : Nat := 96169
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
end LeftBound96169

namespace LeftBound96170
def owner : Owner := ⟨.program ⟨214⟩, ⟨9914⟩⟩
def transferEvent : Nat := 96170
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96165 .summary) (.transfer 96169) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96165 .summary)
      LeftBound96163.bound (LeftBound96163.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9913⟩⟩) (rawTerms := some (Proof.Events375.exact96165RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96163.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96169)
      LeftBound96169.bound (LeftBound96169.actual selector witness) := by
  exact .transfer (LeftBound96169.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96163.bound LeftBound96169.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96163.bound, LeftBound96169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96163.actual selector witness) * (LeftBound96169.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96170

namespace LeftBound96178
def owner : Owner := ⟨.program ⟨214⟩, ⟨12549⟩⟩
def transferEvent : Nat := 96178
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96176 .coefficient, .predecessor 1 96177 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96176 .coefficient)
      LeftBound96168.bound (LeftBound96168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96168.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96177 .coefficient)
      LeftBound96140.bound (LeftBound96140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96140.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96168.bound, LeftBound96140.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96168.bound, LeftBound96140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96168.actual selector witness, LeftBound96140.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96178

namespace LeftBound96180
def owner : Owner := ⟨.program ⟨214⟩, ⟨12549⟩⟩
def transferEvent : Nat := 96180
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 96175 .summary, .result 96145 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96175 .summary)
      LeftBound96170.bound (LeftBound96170.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9914⟩⟩) (rawTerms := some (Proof.Events375.exact96175RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96145 .summary)
      LeftBound96142.bound (LeftBound96142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12548⟩⟩) (rawTerms := some (Proof.Events375.exact96145RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96170.bound, LeftBound96142.bound]
def bound : CoeffClass := .finite ⟨95455360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96170.bound, LeftBound96142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96170.actual selector witness, LeftBound96142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96180

namespace LeftBound96184
def owner : Owner := ⟨.program ⟨214⟩, ⟨25438⟩⟩
def transferEvent : Nat := 96184
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96182 .coefficient) (.predecessor 1 96183 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96182 .coefficient)
      LeftBound96178.bound (LeftBound96178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96183 .coefficient)
      LeftAuthority96116.bound (LeftAuthority96116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96116.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96116.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96178.bound LeftAuthority96116.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96178.bound, LeftAuthority96116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96178.actual selector witness) * (LeftAuthority96116.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96184

namespace LeftBound96185
def owner : Owner := ⟨.program ⟨214⟩, ⟨25438⟩⟩
def transferEvent : Nat := 96185
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩ [⟨.result 96117 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96117 .coefficient)
      LeftAuthority96116.bound (LeftAuthority96116.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25437⟩⟩) (rawTerms := some (Proof.Events375.exact96117RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96116.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96116.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96116.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96116.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96185

namespace LeftBound96186
def owner : Owner := ⟨.program ⟨214⟩, ⟨25438⟩⟩
def transferEvent : Nat := 96186
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96181 .summary) (.transfer 96185) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96181 .summary)
      LeftBound96180.bound (LeftBound96180.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12549⟩⟩) (rawTerms := some (Proof.Events375.exact96181RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96185)
      LeftBound96185.bound (LeftBound96185.actual selector witness) := by
  exact .transfer (LeftBound96185.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96180.bound LeftBound96185.bound
def bound : CoeffClass := .finite ⟨350322698485760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96180.bound, LeftBound96185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96180.actual selector witness) * (LeftBound96185.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96186

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
