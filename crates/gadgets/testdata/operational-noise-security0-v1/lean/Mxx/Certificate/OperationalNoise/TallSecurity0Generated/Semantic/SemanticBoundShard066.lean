import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard065

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound11074
def owner : Owner := ⟨.program ⟨214⟩, ⟨19619⟩⟩
def transferEvent : Nat := 11074
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19616⟩⟩]⟩ [⟨.result 11066 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11066 .coefficient)
      LeftAuthority11065.bound (LeftAuthority11065.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19616⟩⟩) (rawTerms := some (Proof.Events043.exact11066RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11065.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11065.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11065.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11065.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound11074

namespace LeftBound11075
def owner : Owner := ⟨.program ⟨214⟩, ⟨19619⟩⟩
def transferEvent : Nat := 11075
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 11074) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 11074)
      LeftBound11074.bound (LeftBound11074.actual selector witness) := by
  exact .transfer (LeftBound11074.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound11074.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound11074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound11074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11075

namespace LeftBound11154
def owner : Owner := ⟨.program ⟨214⟩, ⟨14461⟩⟩
def transferEvent : Nat := 11154
def frameStart : Nat := 11125
def rule : BoundRule := .product (.predecessor 0 11152 .coefficient) (.predecessor 1 11153 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11152 .coefficient)
      LeftAuthority11150.bound (LeftAuthority11150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11150.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11150.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11153 .coefficient)
      LeftAuthority11147.bound (LeftAuthority11147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11147.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11147.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11150.bound LeftAuthority11147.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11150.bound, LeftAuthority11147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority11150.actual selector witness) * (LeftAuthority11147.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11154

namespace LeftBound11158
def owner : Owner := ⟨.program ⟨214⟩, ⟨14462⟩⟩
def transferEvent : Nat := 11158
def frameStart : Nat := 11125
def rule : BoundRule := .identity (.predecessor 0 11157 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11157 .coefficient)
      LeftBound11154.bound (LeftBound11154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11154.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11154.derived selector witness)

def rawBound : CoeffClass := LeftBound11154.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound11154.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11158

namespace LeftBound11175
def owner : Owner := ⟨.program ⟨214⟩, ⟨14547⟩⟩
def transferEvent : Nat := 11175
def frameStart : Nat := 11125
def rule : BoundRule := .sum [.predecessor 0 11173 .coefficient, .predecessor 1 11174 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11173 .coefficient)
      LeftBound11158.bound (LeftBound11158.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound11158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11174 .coefficient)
      LeftAuthority11171.bound (LeftAuthority11171.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority11171.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11158.bound, LeftAuthority11171.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11158.bound, LeftAuthority11171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11158.actual selector witness, LeftAuthority11171.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11175

namespace LeftBound11178
def owner : Owner := ⟨.program ⟨214⟩, ⟨14548⟩⟩
def transferEvent : Nat := 11178
def frameStart : Nat := 11125
def rule : BoundRule := .identity (.predecessor 0 11177 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11177 .coefficient)
      LeftBound11175.bound (LeftBound11175.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound11175.derived selector witness)

def rawBound : CoeffClass := LeftBound11175.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound11175.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11178

namespace LeftBound11184
def owner : Owner := ⟨.program ⟨214⟩, ⟨14549⟩⟩
def transferEvent : Nat := 11184
def frameStart : Nat := 11125
def rule : BoundRule := .product (.predecessor 0 11182 .coefficient) (.predecessor 1 11183 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11182 .coefficient)
      LeftAuthority11180.bound (LeftAuthority11180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11183 .coefficient)
      LeftBound11178.bound (LeftBound11178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11178.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority11180.bound LeftBound11178.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11180.bound, LeftBound11178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority11180.actual selector witness) * (LeftBound11178.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11184

namespace LeftBound11200
def owner : Owner := ⟨.program ⟨214⟩, ⟨7856⟩⟩
def transferEvent : Nat := 11200
def frameStart : Nat := 11125
def rule : BoundRule := .scale (.predecessor 0 11198 .coefficient) (.value (.predecessor 1 11199 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11198 .coefficient)
      LeftAuthority11196.bound (LeftAuthority11196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11196.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11199 .coefficient)
      LeftAuthority11187.bound (LeftAuthority11187.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority11187.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority11196.bound LeftAuthority11187.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11196.bound, LeftAuthority11187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11196.actual selector witness) * (LeftAuthority11187.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound11200

namespace LeftBound11203
def owner : Owner := ⟨.program ⟨214⟩, ⟨6761⟩⟩
def transferEvent : Nat := 11203
def frameStart : Nat := 11125
def rule : BoundRule := .identity (.predecessor 0 11202 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11202 .coefficient)
      LeftAuthority11190.bound (LeftAuthority11190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11190.derived selector witness)

def rawBound : CoeffClass := LeftAuthority11190.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority11190.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11203

namespace LeftBound11207
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def transferEvent : Nat := 11207
def frameStart : Nat := 11125
def rule : BoundRule := .product (.predecessor 0 11205 .coefficient) (.predecessor 1 11206 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11205 .coefficient)
      LeftBound11203.bound (LeftBound11203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11206 .coefficient)
      LeftBound11200.bound (LeftBound11200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11203.bound LeftBound11200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11203.bound, LeftBound11200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11203.actual selector witness) * (LeftBound11200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11207

namespace LeftBound11212
def owner : Owner := ⟨.program ⟨214⟩, ⟨14550⟩⟩
def transferEvent : Nat := 11212
def frameStart : Nat := 11125
def rule : BoundRule := .sum [.predecessor 0 11210 .coefficient, .predecessor 1 11211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11210 .coefficient)
      LeftBound11207.bound (LeftBound11207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11211 .coefficient)
      LeftBound11184.bound (LeftBound11184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11207.bound, LeftBound11184.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11207.bound, LeftBound11184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11207.actual selector witness, LeftBound11184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11212

namespace LeftBound11216
def owner : Owner := ⟨.program ⟨214⟩, ⟨26166⟩⟩
def transferEvent : Nat := 11216
def frameStart : Nat := 11125
def rule : BoundRule := .product (.predecessor 0 11214 .coefficient) (.predecessor 1 11215 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11214 .coefficient)
      LeftBound11212.bound (LeftBound11212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11215 .coefficient)
      LeftAuthority11169.bound (LeftAuthority11169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11169.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11169.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11212.bound LeftAuthority11169.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11212.bound, LeftAuthority11169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11212.actual selector witness) * (LeftAuthority11169.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11216

namespace LeftBound11227
def owner : Owner := ⟨.program ⟨214⟩, ⟨16077⟩⟩
def transferEvent : Nat := 11227
def frameStart : Nat := 11125
def rule : BoundRule := .product (.predecessor 0 11225 .coefficient) (.predecessor 1 11226 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11225 .coefficient)
      LeftAuthority11180.bound (LeftAuthority11180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11226 .coefficient)
      LeftAuthority11223.bound (LeftAuthority11223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11180.bound LeftAuthority11223.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11180.bound, LeftAuthority11223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority11180.actual selector witness) * (LeftAuthority11223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11227

namespace LeftBound11235
def owner : Owner := ⟨.program ⟨214⟩, ⟨16078⟩⟩
def transferEvent : Nat := 11235
def frameStart : Nat := 11125
def rule : BoundRule := .sum [.predecessor 0 11233 .coefficient, .predecessor 1 11234 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11233 .coefficient)
      LeftAuthority11231.bound (LeftAuthority11231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11231.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11234 .coefficient)
      LeftBound11227.bound (LeftBound11227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11227.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority11231.bound, LeftBound11227.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11231.bound, LeftBound11227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority11231.actual selector witness, LeftBound11227.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11235

namespace LeftBound11239
def owner : Owner := ⟨.program ⟨214⟩, ⟨26167⟩⟩
def transferEvent : Nat := 11239
def frameStart : Nat := 11125
def rule : BoundRule := .sum [.predecessor 0 11237 .coefficient, .predecessor 1 11238 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11237 .coefficient)
      LeftBound11235.bound (LeftBound11235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11238 .coefficient)
      LeftBound11216.bound (LeftBound11216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11216.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11235.bound, LeftBound11216.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11235.bound, LeftBound11216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11235.actual selector witness, LeftBound11216.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11239

namespace LeftBound11252
def owner : Owner := ⟨.program ⟨214⟩, ⟨26165⟩⟩
def transferEvent : Nat := 11252
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11250 .coefficient, .predecessor 1 11251 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11250 .coefficient)
      LeftBound11073.bound (LeftBound11073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11251 .coefficient)
      LeftBound11056.bound (LeftBound11056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11056.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11073.bound, LeftBound11056.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11073.bound, LeftBound11056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11073.actual selector witness, LeftBound11056.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11252

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
