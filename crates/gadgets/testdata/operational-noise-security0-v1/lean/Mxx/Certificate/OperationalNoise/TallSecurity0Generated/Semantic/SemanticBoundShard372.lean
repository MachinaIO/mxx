import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard371

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55055
def owner : Owner := ⟨.program ⟨214⟩, ⟨7255⟩⟩
def transferEvent : Nat := 55055
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55053 .coefficient) (.predecessor 1 55054 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55053 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55054 .coefficient)
      LeftBound11021.bound (LeftBound11021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound11021.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound11021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound11021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55055

namespace LeftBound55060
def owner : Owner := ⟨.program ⟨214⟩, ⟨14438⟩⟩
def transferEvent : Nat := 55060
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55058 .coefficient, .predecessor 1 55059 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55058 .coefficient)
      LeftBound55055.bound (LeftBound55055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55059 .coefficient)
      LeftBound55050.bound (LeftBound55050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55055.bound, LeftBound55050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55055.bound, LeftBound55050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55055.actual selector witness, LeftBound55050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55060

namespace LeftBound55064
def owner : Owner := ⟨.program ⟨214⟩, ⟨14439⟩⟩
def transferEvent : Nat := 55064
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55062 .coefficient, .predecessor 1 55063 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55062 .coefficient)
      LeftBound55060.bound (LeftBound55060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55060.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55060.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55063 .coefficient)
      LeftBound11013.bound (LeftBound11013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55060.bound, LeftBound11013.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55060.bound, LeftBound11013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55060.actual selector witness, LeftBound11013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55064

namespace LeftBound55065
def owner : Owner := ⟨.program ⟨214⟩, ⟨14439⟩⟩
def transferEvent : Nat := 55065
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨75⟩⟩]⟩ [⟨.result 11014 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11014 .coefficient)
      LeftBound11013.bound (LeftBound11013.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨75⟩⟩) (rawTerms := some (Proof.Events043.exact11014RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11013.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11013.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55065

namespace LeftBound55070
def owner : Owner := ⟨.program ⟨214⟩, ⟨14440⟩⟩
def transferEvent : Nat := 55070
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55068 .coefficient) (.predecessor 1 55069 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55068 .coefficient)
      LeftBound55064.bound (LeftBound55064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55069 .coefficient)
      LeftBound11010.bound (LeftBound11010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11010.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55064.bound LeftBound11010.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55064.bound, LeftBound11010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55064.actual selector witness) * (LeftBound11010.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55070

namespace LeftBound55071
def owner : Owner := ⟨.program ⟨214⟩, ⟨14440⟩⟩
def transferEvent : Nat := 55071
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩ [⟨.result 11007 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11007 .coefficient)
      LeftAuthority11006.bound (LeftAuthority11006.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7855⟩⟩) (rawTerms := some (Proof.Events042.exact11007RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11006.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11006.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11006.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55071

namespace LeftBound55072
def owner : Owner := ⟨.program ⟨214⟩, ⟨14440⟩⟩
def transferEvent : Nat := 55072
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55067 .summary) (.transfer 55071) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55067 .summary)
      LeftBound55065.bound (LeftBound55065.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14439⟩⟩) (rawTerms := some (Proof.Events215.exact55067RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55065.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55071)
      LeftBound55071.bound (LeftBound55071.actual selector witness) := by
  exact .transfer (LeftBound55071.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55065.bound LeftBound55071.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55065.bound, LeftBound55071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55065.actual selector witness) * (LeftBound55071.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55072

namespace LeftBound55080
def owner : Owner := ⟨.program ⟨214⟩, ⟨14441⟩⟩
def transferEvent : Nat := 55080
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55078 .coefficient, .predecessor 1 55079 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55078 .coefficient)
      LeftBound55070.bound (LeftBound55070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55079 .coefficient)
      LeftBound55042.bound (LeftBound55042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55070.bound, LeftBound55042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55070.bound, LeftBound55042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55070.actual selector witness, LeftBound55042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55080

namespace LeftBound55082
def owner : Owner := ⟨.program ⟨214⟩, ⟨14441⟩⟩
def transferEvent : Nat := 55082
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55077 .summary, .result 55047 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55077 .summary)
      LeftBound55072.bound (LeftBound55072.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14440⟩⟩) (rawTerms := some (Proof.Events215.exact55077RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55047 .summary)
      LeftBound55044.bound (LeftBound55044.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14436⟩⟩) (rawTerms := some (Proof.Events215.exact55047RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55044.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55072.bound, LeftBound55044.bound]
def bound : CoeffClass := .finite ⟨95438720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55072.bound, LeftBound55044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55072.actual selector witness, LeftBound55044.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55082

namespace LeftBound55086
def owner : Owner := ⟨.program ⟨214⟩, ⟨26149⟩⟩
def transferEvent : Nat := 55086
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55084 .coefficient) (.predecessor 1 55085 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55084 .coefficient)
      LeftBound55080.bound (LeftBound55080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55080.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55085 .coefficient)
      LeftAuthority55018.bound (LeftAuthority55018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact55019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55018.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55018.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55080.bound LeftAuthority55018.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55080.bound, LeftAuthority55018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55080.actual selector witness) * (LeftAuthority55018.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55086

namespace LeftBound55087
def owner : Owner := ⟨.program ⟨214⟩, ⟨26149⟩⟩
def transferEvent : Nat := 55087
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩ [⟨.result 55019 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55019 .coefficient)
      LeftAuthority55018.bound (LeftAuthority55018.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26148⟩⟩) (rawTerms := some (Proof.Events214.exact55019RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55018.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55018.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55018.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55018.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55087

namespace LeftBound55088
def owner : Owner := ⟨.program ⟨214⟩, ⟨26149⟩⟩
def transferEvent : Nat := 55088
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55083 .summary) (.transfer 55087) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55083 .summary)
      LeftBound55082.bound (LeftBound55082.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14441⟩⟩) (rawTerms := some (Proof.Events215.exact55083RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55087)
      LeftBound55087.bound (LeftBound55087.actual selector witness) := by
  exact .transfer (LeftBound55087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55082.bound LeftBound55087.bound
def bound : CoeffClass := .finite ⟨350261629419520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55082.bound, LeftBound55087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55082.actual selector witness) * (LeftBound55087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55088

namespace LeftBound55099
def owner : Owner := ⟨.program ⟨214⟩, ⟨19606⟩⟩
def transferEvent : Nat := 55099
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 55097 .coefficient) (.value (.predecessor 1 55098 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55097 .coefficient)
      LeftAuthority55095.bound (LeftAuthority55095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55098 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority55095.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55095.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55095.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55099

namespace LeftBound55103
def owner : Owner := ⟨.program ⟨214⟩, ⟨19607⟩⟩
def transferEvent : Nat := 55103
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55101 .coefficient) (.predecessor 1 55102 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55101 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55102 .coefficient)
      LeftBound55099.bound (LeftBound55099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55099.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound55099.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound55099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound55099.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55103

namespace LeftBound55104
def owner : Owner := ⟨.program ⟨214⟩, ⟨19607⟩⟩
def transferEvent : Nat := 55104
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩ [⟨.result 55096 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55096 .coefficient)
      LeftAuthority55095.bound (LeftAuthority55095.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19604⟩⟩) (rawTerms := some (Proof.Events215.exact55096RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55095.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55095.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55095.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55104

namespace LeftBound55105
def owner : Owner := ⟨.program ⟨214⟩, ⟨19607⟩⟩
def transferEvent : Nat := 55105
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 55104) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55104)
      LeftBound55104.bound (LeftBound55104.actual selector witness) := by
  exact .transfer (LeftBound55104.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound55104.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound55104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound55104.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55105

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
