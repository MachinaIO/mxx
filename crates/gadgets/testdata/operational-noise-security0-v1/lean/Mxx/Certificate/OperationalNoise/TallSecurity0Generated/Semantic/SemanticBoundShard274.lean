import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard273

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40900
def owner : Owner := ⟨.program ⟨214⟩, ⟨14228⟩⟩
def transferEvent : Nat := 40900
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14225⟩⟩], []⟩ [⟨.result 1823 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1823 .coefficient)
      LeftAuthority1822.bound (LeftAuthority1822.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14225⟩⟩) (rawTerms := some (Proof.Events007.exact1823RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1822.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1822.bound []
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1822.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40900

namespace LeftBound40901
def owner : Owner := ⟨.program ⟨214⟩, ⟨14228⟩⟩
def transferEvent : Nat := 40901
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40896 .summary) (.transfer 40900) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40896 .summary)
      LeftBound40894.bound (LeftBound40894.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11480⟩⟩) (rawTerms := some (Proof.Events159.exact40896RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40900)
      LeftBound40900.bound (LeftBound40900.actual selector witness) := by
  exact .transfer (LeftBound40900.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound40894.bound LeftBound40900.bound
def bound : CoeffClass := .finite ⟨14976, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40894.bound, LeftBound40900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound40894.actual selector witness) * (LeftBound40900.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40901

namespace LeftBound40907
def owner : Owner := ⟨.program ⟨214⟩, ⟨14229⟩⟩
def transferEvent : Nat := 40907
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 40905 .coefficient) (.predecessor 1 40906 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40905 .coefficient)
      LeftAuthority1822.bound (LeftAuthority1822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40906 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1822.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1822.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1822.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40907

namespace LeftBound40912
def owner : Owner := ⟨.program ⟨214⟩, ⟨7291⟩⟩
def transferEvent : Nat := 40912
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40910 .coefficient) (.predecessor 1 40911 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40910 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40911 .coefficient)
      LeftBound11522.bound (LeftBound11522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound11522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound11522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound11522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40912

namespace LeftBound40917
def owner : Owner := ⟨.program ⟨214⟩, ⟨14230⟩⟩
def transferEvent : Nat := 40917
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40915 .coefficient, .predecessor 1 40916 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40915 .coefficient)
      LeftBound40912.bound (LeftBound40912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40916 .coefficient)
      LeftBound40907.bound (LeftBound40907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40912.bound, LeftBound40907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40912.bound, LeftBound40907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40912.actual selector witness, LeftBound40907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40917

namespace LeftBound40921
def owner : Owner := ⟨.program ⟨214⟩, ⟨14231⟩⟩
def transferEvent : Nat := 40921
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40919 .coefficient, .predecessor 1 40920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40919 .coefficient)
      LeftBound40917.bound (LeftBound40917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40920 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40917.bound, LeftBound11514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40917.bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40917.actual selector witness, LeftBound11514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40921

namespace LeftBound40922
def owner : Owner := ⟨.program ⟨214⟩, ⟨14231⟩⟩
def transferEvent : Nat := 40922
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩ [⟨.result 11515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11515 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨73⟩⟩) (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11514.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40922

namespace LeftBound40927
def owner : Owner := ⟨.program ⟨214⟩, ⟨14232⟩⟩
def transferEvent : Nat := 40927
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40925 .coefficient) (.predecessor 1 40926 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40925 .coefficient)
      LeftBound40921.bound (LeftBound40921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40926 .coefficient)
      LeftBound11511.bound (LeftBound11511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40921.bound LeftBound11511.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40921.bound, LeftBound11511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40921.actual selector witness) * (LeftBound11511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40927

namespace LeftBound40928
def owner : Owner := ⟨.program ⟨214⟩, ⟨14232⟩⟩
def transferEvent : Nat := 40928
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩ [⟨.result 11508 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11508 .coefficient)
      LeftAuthority11507.bound (LeftAuthority11507.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7852⟩⟩) (rawTerms := some (Proof.Events044.exact11508RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11507.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11507.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11507.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40928

namespace LeftBound40929
def owner : Owner := ⟨.program ⟨214⟩, ⟨14232⟩⟩
def transferEvent : Nat := 40929
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40924 .summary) (.transfer 40928) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40924 .summary)
      LeftBound40922.bound (LeftBound40922.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14231⟩⟩) (rawTerms := some (Proof.Events159.exact40924RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40928)
      LeftBound40928.bound (LeftBound40928.actual selector witness) := by
  exact .transfer (LeftBound40928.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40922.bound LeftBound40928.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40922.bound, LeftBound40928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40922.actual selector witness) * (LeftBound40928.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40929

namespace LeftBound40937
def owner : Owner := ⟨.program ⟨214⟩, ⟨14233⟩⟩
def transferEvent : Nat := 40937
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40935 .coefficient, .predecessor 1 40936 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40935 .coefficient)
      LeftBound40927.bound (LeftBound40927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40936 .coefficient)
      LeftBound40899.bound (LeftBound40899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40899.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40927.bound, LeftBound40899.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40927.bound, LeftBound40899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40927.actual selector witness, LeftBound40899.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40937

namespace LeftBound40939
def owner : Owner := ⟨.program ⟨214⟩, ⟨14233⟩⟩
def transferEvent : Nat := 40939
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40934 .summary, .result 40904 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40934 .summary)
      LeftBound40929.bound (LeftBound40929.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14232⟩⟩) (rawTerms := some (Proof.Events159.exact40934RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40904 .summary)
      LeftBound40901.bound (LeftBound40901.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14228⟩⟩) (rawTerms := some (Proof.Events159.exact40904RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40901.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40929.bound, LeftBound40901.bound]
def bound : CoeffClass := .finite ⟨95435392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40929.bound, LeftBound40901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40929.actual selector witness, LeftBound40901.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40939

namespace LeftBound40943
def owner : Owner := ⟨.program ⟨214⟩, ⟨26077⟩⟩
def transferEvent : Nat := 40943
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40941 .coefficient) (.predecessor 1 40942 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40941 .coefficient)
      LeftBound40937.bound (LeftBound40937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40937.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40942 .coefficient)
      LeftAuthority40875.bound (LeftAuthority40875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40875.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40875.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40937.bound LeftAuthority40875.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40937.bound, LeftAuthority40875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40937.actual selector witness) * (LeftAuthority40875.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40943

namespace LeftBound40944
def owner : Owner := ⟨.program ⟨214⟩, ⟨26077⟩⟩
def transferEvent : Nat := 40944
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26076⟩⟩]⟩ [⟨.result 40876 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40876 .coefficient)
      LeftAuthority40875.bound (LeftAuthority40875.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26076⟩⟩) (rawTerms := some (Proof.Events159.exact40876RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40875.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40875.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority40875.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40875.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40944

namespace LeftBound40945
def owner : Owner := ⟨.program ⟨214⟩, ⟨26077⟩⟩
def transferEvent : Nat := 40945
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40940 .summary) (.transfer 40944) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40940 .summary)
      LeftBound40939.bound (LeftBound40939.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14233⟩⟩) (rawTerms := some (Proof.Events159.exact40940RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40944)
      LeftBound40944.bound (LeftBound40944.actual selector witness) := by
  exact .transfer (LeftBound40944.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40939.bound LeftBound40944.bound
def bound : CoeffClass := .finite ⟨350249415606272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40939.bound, LeftBound40944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40939.actual selector witness) * (LeftBound40944.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40945

namespace LeftBound40956
def owner : Owner := ⟨.program ⟨214⟩, ⟨19538⟩⟩
def transferEvent : Nat := 40956
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 40954 .coefficient) (.value (.predecessor 1 40955 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40954 .coefficient)
      LeftAuthority40952.bound (LeftAuthority40952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40955 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority40952.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40952.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40952.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40956

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
