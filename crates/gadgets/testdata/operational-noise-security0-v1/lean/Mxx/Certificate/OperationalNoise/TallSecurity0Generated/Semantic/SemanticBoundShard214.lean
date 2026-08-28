import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard160
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard213

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound32935
def owner : Owner := ⟨.program ⟨214⟩, ⟨16520⟩⟩
def transferEvent : Nat := 32935
def frameStart : Nat := 32862
def rule : BoundRule := .sum [.predecessor 0 32933 .coefficient, .predecessor 1 32934 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32933 .coefficient)
      LeftAuthority32931.bound (LeftAuthority32931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32934 .coefficient)
      LeftBound32927.bound (LeftBound32927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32931.bound, LeftBound32927.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32931.bound, LeftBound32927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32931.actual selector witness, LeftBound32927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32935

namespace LeftBound32939
def owner : Owner := ⟨.program ⟨214⟩, ⟨28984⟩⟩
def transferEvent : Nat := 32939
def frameStart : Nat := 32862
def rule : BoundRule := .product (.predecessor 0 32937 .coefficient) (.predecessor 1 32938 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32937 .coefficient)
      LeftBound32935.bound (LeftBound32935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32938 .coefficient)
      LeftAuthority32912.bound (LeftAuthority32912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32912.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32912.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32935.bound LeftAuthority32912.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32935.bound, LeftAuthority32912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32935.actual selector witness) * (LeftAuthority32912.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32939

namespace LeftBound32950
def owner : Owner := ⟨.program ⟨214⟩, ⟨17564⟩⟩
def transferEvent : Nat := 32950
def frameStart : Nat := 32862
def rule : BoundRule := .product (.predecessor 0 32948 .coefficient) (.predecessor 1 32949 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32948 .coefficient)
      LeftAuthority32923.bound (LeftAuthority32923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32949 .coefficient)
      LeftAuthority32946.bound (LeftAuthority32946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32946.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32946.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority32923.bound LeftAuthority32946.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32923.bound, LeftAuthority32946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority32923.actual selector witness) * (LeftAuthority32946.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32950

namespace LeftBound32958
def owner : Owner := ⟨.program ⟨214⟩, ⟨17565⟩⟩
def transferEvent : Nat := 32958
def frameStart : Nat := 32862
def rule : BoundRule := .sum [.predecessor 0 32956 .coefficient, .predecessor 1 32957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32956 .coefficient)
      LeftAuthority32954.bound (LeftAuthority32954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32954.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32957 .coefficient)
      LeftBound32950.bound (LeftBound32950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32950.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32954.bound, LeftBound32950.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32954.bound, LeftBound32950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32954.actual selector witness, LeftBound32950.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32958

namespace LeftBound32962
def owner : Owner := ⟨.program ⟨214⟩, ⟨28989⟩⟩
def transferEvent : Nat := 32962
def frameStart : Nat := 32862
def rule : BoundRule := .sum [.predecessor 0 32960 .coefficient, .predecessor 1 32961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32960 .coefficient)
      LeftBound32958.bound (LeftBound32958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32961 .coefficient)
      LeftBound32939.bound (LeftBound32939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32939.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32958.bound, LeftBound32939.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32958.bound, LeftBound32939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32958.actual selector witness, LeftBound32939.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32962

namespace LeftBound32975
def owner : Owner := ⟨.program ⟨214⟩, ⟨28986⟩⟩
def transferEvent : Nat := 32975
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 32973 .coefficient, .predecessor 1 32974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32973 .coefficient)
      LeftBound32804.bound (LeftBound32804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32804.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32974 .coefficient)
      LeftBound32787.bound (LeftBound32787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32787.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32804.bound, LeftBound32787.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32804.bound, LeftBound32787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32804.actual selector witness, LeftBound32787.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32975

namespace LeftBound32978
def owner : Owner := ⟨.program ⟨214⟩, ⟨28986⟩⟩
def transferEvent : Nat := 32978
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 32972 .summary, .result 32794 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32972 .summary)
      LeftBound32806.bound (LeftBound32806.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22063⟩⟩) (rawTerms := some (Proof.Events128.exact32972RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32794 .summary)
      LeftBound32789.bound (LeftBound32789.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28985⟩⟩) (rawTerms := some (Proof.Events128.exact32794RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32789.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32806.bound, LeftBound32789.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32806.bound, LeftBound32789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32806.actual selector witness, LeftBound32789.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32978

namespace LeftBound32982
def owner : Owner := ⟨.program ⟨214⟩, ⟨28987⟩⟩
def transferEvent : Nat := 32982
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32980 .coefficient) (.predecessor 1 32981 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32980 .coefficient)
      LeftBound32975.bound (LeftBound32975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32981 .coefficient)
      LeftBound5618.bound (LeftBound5618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32975.bound LeftBound5618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32975.bound, LeftBound5618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32975.actual selector witness) * (LeftBound5618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32982

namespace LeftBound32983
def owner : Owner := ⟨.program ⟨214⟩, ⟨28987⟩⟩
def transferEvent : Nat := 32983
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩ [⟨.result 5615 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5615 .coefficient)
      LeftAuthority5614.bound (LeftAuthority5614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6669⟩⟩) (rawTerms := some (Proof.Events021.exact5615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5614.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32983

namespace LeftBound32984
def owner : Owner := ⟨.program ⟨214⟩, ⟨28987⟩⟩
def transferEvent : Nat := 32984
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32979 .summary) (.transfer 32983) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32979 .summary)
      LeftBound32978.bound (LeftBound32978.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28986⟩⟩) (rawTerms := some (Proof.Events128.exact32979RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 32983)
      LeftBound32983.bound (LeftBound32983.actual selector witness) := by
  exact .transfer (LeftBound32983.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32978.bound LeftBound32983.bound
def bound : CoeffClass := .finite ⟨4742816766803936246568583168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32978.bound, LeftBound32983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32978.actual selector witness) * (LeftBound32983.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32984

namespace LeftBound32999
def owner : Owner := ⟨.program ⟨214⟩, ⟨28768⟩⟩
def transferEvent : Nat := 32999
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32997 .coefficient) (.predecessor 1 32998 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32997 .coefficient)
      LeftBound24586.bound (LeftBound24586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32998 .coefficient)
      LeftAuthority32995.bound (LeftAuthority32995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32995.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32995.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24586.bound LeftAuthority32995.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24586.bound, LeftAuthority32995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24586.actual selector witness) * (LeftAuthority32995.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32999

namespace LeftBound33000
def owner : Owner := ⟨.program ⟨214⟩, ⟨28768⟩⟩
def transferEvent : Nat := 33000
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28766⟩⟩]⟩ [⟨.result 32996 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32996 .coefficient)
      LeftAuthority32995.bound (LeftAuthority32995.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28766⟩⟩) (rawTerms := some (Proof.Events128.exact32996RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32995.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32995.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority32995.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority32995.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33000

namespace LeftBound33001
def owner : Owner := ⟨.program ⟨214⟩, ⟨28768⟩⟩
def transferEvent : Nat := 33001
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 24590 .summary) (.transfer 33000) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24590 .summary)
      LeftBound24589.bound (LeftBound24589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25236⟩⟩) (rawTerms := some (Proof.Events096.exact24590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33000)
      LeftBound33000.bound (LeftBound33000.actual selector witness) := by
  exact .transfer (LeftBound33000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24589.bound LeftBound33000.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24589.bound, LeftBound33000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24589.actual selector witness) * (LeftBound33000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33001

namespace LeftBound33012
def owner : Owner := ⟨.program ⟨214⟩, ⟨21918⟩⟩
def transferEvent : Nat := 33012
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 33010 .coefficient) (.value (.predecessor 1 33011 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33010 .coefficient)
      LeftAuthority33008.bound (LeftAuthority33008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact33009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33008.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33011 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority33008.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33008.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33008.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound33012

namespace LeftBound33016
def owner : Owner := ⟨.program ⟨214⟩, ⟨21919⟩⟩
def transferEvent : Nat := 33016
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33014 .coefficient) (.predecessor 1 33015 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33014 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33015 .coefficient)
      LeftBound33012.bound (LeftBound33012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact33013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound33012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound33012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound33012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33016

namespace LeftBound33017
def owner : Owner := ⟨.program ⟨214⟩, ⟨21919⟩⟩
def transferEvent : Nat := 33017
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21916⟩⟩]⟩ [⟨.result 33009 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33009 .coefficient)
      LeftAuthority33008.bound (LeftAuthority33008.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21916⟩⟩) (rawTerms := some (Proof.Events128.exact33009RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33008.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority33008.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33008.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33017

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
