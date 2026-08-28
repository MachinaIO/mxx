import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard251

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound37997
def owner : Owner := ⟨.program ⟨214⟩, ⟨12586⟩⟩
def transferEvent : Nat := 37997
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37995 .coefficient, .predecessor 1 37996 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37995 .coefficient)
      LeftBound37992.bound (LeftBound37992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37996 .coefficient)
      LeftBound37987.bound (LeftBound37987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37992.bound, LeftBound37987.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37992.bound, LeftBound37987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37992.actual selector witness, LeftBound37987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37997

namespace LeftBound38001
def owner : Owner := ⟨.program ⟨214⟩, ⟨12587⟩⟩
def transferEvent : Nat := 38001
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37999 .coefficient, .predecessor 1 38000 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37999 .coefficient)
      LeftBound37997.bound (LeftBound37997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38000 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37997.bound, LeftBound8467.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37997.bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37997.actual selector witness, LeftBound8467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38001

namespace LeftBound38002
def owner : Owner := ⟨.program ⟨214⟩, ⟨12587⟩⟩
def transferEvent : Nat := 38002
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩ [⟨.result 8468 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8468 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨100⟩⟩) (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8467.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8467.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38002

namespace LeftBound38007
def owner : Owner := ⟨.program ⟨214⟩, ⟨12588⟩⟩
def transferEvent : Nat := 38007
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38005 .coefficient) (.predecessor 1 38006 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38005 .coefficient)
      LeftBound38001.bound (LeftBound38001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38006 .coefficient)
      LeftAuthority1684.bound (LeftAuthority1684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1684.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound38001.bound LeftAuthority1684.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38001.bound, LeftAuthority1684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound38001.actual selector witness) * (LeftAuthority1684.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38007

namespace LeftBound38008
def owner : Owner := ⟨.program ⟨214⟩, ⟨12588⟩⟩
def transferEvent : Nat := 38008
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9935⟩⟩], []⟩ [⟨.result 1685 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1685 .coefficient)
      LeftAuthority1684.bound (LeftAuthority1684.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9935⟩⟩) (rawTerms := some (Proof.Events006.exact1685RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1684.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1684.bound []
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1684.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38008

namespace LeftBound38009
def owner : Owner := ⟨.program ⟨214⟩, ⟨12588⟩⟩
def transferEvent : Nat := 38009
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38004 .summary) (.transfer 38008) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38004 .summary)
      LeftBound38002.bound (LeftBound38002.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12587⟩⟩) (rawTerms := some (Proof.Events148.exact38004RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38008)
      LeftBound38008.bound (LeftBound38008.actual selector witness) := by
  exact .transfer (LeftBound38008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound38002.bound LeftBound38008.bound
def bound : CoeffClass := .finite ⟨34944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38002.bound, LeftBound38008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound38002.actual selector witness) * (LeftBound38008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38009

namespace LeftBound38015
def owner : Owner := ⟨.program ⟨214⟩, ⟨9936⟩⟩
def transferEvent : Nat := 38015
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 38013 .coefficient) (.predecessor 1 38014 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38013 .coefficient)
      LeftAuthority1684.bound (LeftAuthority1684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38014 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1684.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1684.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1684.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38015

namespace LeftBound38020
def owner : Owner := ⟨.program ⟨214⟩, ⟨7298⟩⟩
def transferEvent : Nat := 38020
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38018 .coefficient) (.predecessor 1 38019 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38018 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38019 .coefficient)
      LeftBound8516.bound (LeftBound8516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound8516.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound8516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound8516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38020

namespace LeftBound38025
def owner : Owner := ⟨.program ⟨214⟩, ⟨9937⟩⟩
def transferEvent : Nat := 38025
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38023 .coefficient, .predecessor 1 38024 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38023 .coefficient)
      LeftBound38020.bound (LeftBound38020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38024 .coefficient)
      LeftBound38015.bound (LeftBound38015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38020.bound, LeftBound38015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38020.bound, LeftBound38015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38020.actual selector witness, LeftBound38015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38025

namespace LeftBound38029
def owner : Owner := ⟨.program ⟨214⟩, ⟨9938⟩⟩
def transferEvent : Nat := 38029
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38027 .coefficient, .predecessor 1 38028 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38027 .coefficient)
      LeftBound38025.bound (LeftBound38025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38025.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38028 .coefficient)
      LeftBound8508.bound (LeftBound8508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38025.bound, LeftBound8508.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38025.bound, LeftBound8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38025.actual selector witness, LeftBound8508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38029

namespace LeftBound38030
def owner : Owner := ⟨.program ⟨214⟩, ⟨9938⟩⟩
def transferEvent : Nat := 38030
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
end LeftBound38030

namespace LeftBound38035
def owner : Owner := ⟨.program ⟨214⟩, ⟨9939⟩⟩
def transferEvent : Nat := 38035
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38033 .coefficient) (.predecessor 1 38034 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38033 .coefficient)
      LeftBound38029.bound (LeftBound38029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38034 .coefficient)
      LeftBound8505.bound (LeftBound8505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38029.bound LeftBound8505.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38029.bound, LeftBound8505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38029.actual selector witness) * (LeftBound8505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38035

namespace LeftBound38036
def owner : Owner := ⟨.program ⟨214⟩, ⟨9939⟩⟩
def transferEvent : Nat := 38036
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
end LeftBound38036

namespace LeftBound38037
def owner : Owner := ⟨.program ⟨214⟩, ⟨9939⟩⟩
def transferEvent : Nat := 38037
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38032 .summary) (.transfer 38036) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38032 .summary)
      LeftBound38030.bound (LeftBound38030.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9938⟩⟩) (rawTerms := some (Proof.Events148.exact38032RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38036)
      LeftBound38036.bound (LeftBound38036.actual selector witness) := by
  exact .transfer (LeftBound38036.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38030.bound LeftBound38036.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38030.bound, LeftBound38036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38030.actual selector witness) * (LeftBound38036.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38037

namespace LeftBound38045
def owner : Owner := ⟨.program ⟨214⟩, ⟨12589⟩⟩
def transferEvent : Nat := 38045
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38043 .coefficient, .predecessor 1 38044 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38043 .coefficient)
      LeftBound38035.bound (LeftBound38035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38044 .coefficient)
      LeftBound38007.bound (LeftBound38007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38035.bound, LeftBound38007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38035.bound, LeftBound38007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38035.actual selector witness, LeftBound38007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38045

namespace LeftBound38047
def owner : Owner := ⟨.program ⟨214⟩, ⟨12589⟩⟩
def transferEvent : Nat := 38047
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 38042 .summary, .result 38012 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38042 .summary)
      LeftBound38037.bound (LeftBound38037.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9939⟩⟩) (rawTerms := some (Proof.Events148.exact38042RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38012 .summary)
      LeftBound38009.bound (LeftBound38009.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12588⟩⟩) (rawTerms := some (Proof.Events148.exact38012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38037.bound, LeftBound38009.bound]
def bound : CoeffClass := .finite ⟨95455360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38037.bound, LeftBound38009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38037.actual selector witness, LeftBound38009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38047

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
