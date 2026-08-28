import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard663

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound97184
def owner : Owner := ⟨.program ⟨214⟩, ⟨12048⟩⟩
def transferEvent : Nat := 97184
def frameStart : Nat := 97109
def rule : BoundRule := .sum [.predecessor 0 97182 .coefficient, .predecessor 1 97183 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97182 .coefficient)
      LeftBound97179.bound (LeftBound97179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97183 .coefficient)
      LeftBound97156.bound (LeftBound97156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97156.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97179.bound, LeftBound97156.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97179.bound, LeftBound97156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97179.actual selector witness, LeftBound97156.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97184

namespace LeftBound97188
def owner : Owner := ⟨.program ⟨214⟩, ⟨25209⟩⟩
def transferEvent : Nat := 97188
def frameStart : Nat := 97109
def rule : BoundRule := .product (.predecessor 0 97186 .coefficient) (.predecessor 1 97187 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97186 .coefficient)
      LeftBound97184.bound (LeftBound97184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97187 .coefficient)
      LeftAuthority97141.bound (LeftAuthority97141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97141.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97184.bound LeftAuthority97141.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97184.bound, LeftAuthority97141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97184.actual selector witness) * (LeftAuthority97141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97188

namespace LeftBound97199
def owner : Owner := ⟨.program ⟨214⟩, ⟨16373⟩⟩
def transferEvent : Nat := 97199
def frameStart : Nat := 97109
def rule : BoundRule := .product (.predecessor 0 97197 .coefficient) (.predecessor 1 97198 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97197 .coefficient)
      LeftAuthority97152.bound (LeftAuthority97152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97198 .coefficient)
      LeftAuthority97195.bound (LeftAuthority97195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97195.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97195.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority97152.bound LeftAuthority97195.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97152.bound, LeftAuthority97195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority97152.actual selector witness) * (LeftAuthority97195.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97199

namespace LeftBound97207
def owner : Owner := ⟨.program ⟨214⟩, ⟨16374⟩⟩
def transferEvent : Nat := 97207
def frameStart : Nat := 97109
def rule : BoundRule := .sum [.predecessor 0 97205 .coefficient, .predecessor 1 97206 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97205 .coefficient)
      LeftAuthority97203.bound (LeftAuthority97203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97206 .coefficient)
      LeftBound97199.bound (LeftBound97199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97199.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97199.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority97203.bound, LeftBound97199.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97203.bound, LeftBound97199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority97203.actual selector witness, LeftBound97199.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97207

namespace LeftBound97211
def owner : Owner := ⟨.program ⟨214⟩, ⟨25210⟩⟩
def transferEvent : Nat := 97211
def frameStart : Nat := 97109
def rule : BoundRule := .sum [.predecessor 0 97209 .coefficient, .predecessor 1 97210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97209 .coefficient)
      LeftBound97207.bound (LeftBound97207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97210 .coefficient)
      LeftBound97188.bound (LeftBound97188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97188.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97207.bound, LeftBound97188.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97207.bound, LeftBound97188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97207.actual selector witness, LeftBound97188.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97211

namespace LeftBound97224
def owner : Owner := ⟨.program ⟨214⟩, ⟨25208⟩⟩
def transferEvent : Nat := 97224
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97222 .coefficient, .predecessor 1 97223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97222 .coefficient)
      LeftBound97069.bound (LeftBound97069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97223 .coefficient)
      LeftBound97052.bound (LeftBound97052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97069.bound, LeftBound97052.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97069.bound, LeftBound97052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97069.actual selector witness, LeftBound97052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97224

namespace LeftBound97227
def owner : Owner := ⟨.program ⟨214⟩, ⟨25208⟩⟩
def transferEvent : Nat := 97227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 97221 .summary, .result 97059 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97221 .summary)
      LeftBound97071.bound (LeftBound97071.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19808⟩⟩) (rawTerms := some (Proof.Events379.exact97221RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97059 .summary)
      LeftBound97054.bound (LeftBound97054.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25207⟩⟩) (rawTerms := some (Proof.Events379.exact97059RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97071.bound, LeftBound97054.bound]
def bound : CoeffClass := .finite ⟨352115681275904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97071.bound, LeftBound97054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97071.actual selector witness, LeftBound97054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97227

namespace LeftBound97231
def owner : Owner := ⟨.program ⟨214⟩, ⟨28701⟩⟩
def transferEvent : Nat := 97231
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97229 .coefficient) (.predecessor 1 97230 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97229 .coefficient)
      LeftBound97224.bound (LeftBound97224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97230 .coefficient)
      LeftAuthority96974.bound (LeftAuthority96974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96974.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96974.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97224.bound LeftAuthority96974.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97224.bound, LeftAuthority96974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97224.actual selector witness) * (LeftAuthority96974.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97231

namespace LeftBound97232
def owner : Owner := ⟨.program ⟨214⟩, ⟨28701⟩⟩
def transferEvent : Nat := 97232
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩ [⟨.result 96975 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96975 .coefficient)
      LeftAuthority96974.bound (LeftAuthority96974.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28699⟩⟩) (rawTerms := some (Proof.Events378.exact96975RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96974.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96974.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96974.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96974.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97232

namespace LeftBound97233
def owner : Owner := ⟨.program ⟨214⟩, ⟨28701⟩⟩
def transferEvent : Nat := 97233
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97228 .summary) (.transfer 97232) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97228 .summary)
      LeftBound97227.bound (LeftBound97227.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25208⟩⟩) (rawTerms := some (Proof.Events379.exact97228RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97232)
      LeftBound97232.bound (LeftBound97232.actual selector witness) := by
  exact .transfer (LeftBound97232.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97227.bound LeftBound97232.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97227.bound, LeftBound97232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97227.actual selector witness) * (LeftBound97232.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97233

namespace LeftBound97244
def owner : Owner := ⟨.program ⟨214⟩, ⟨21967⟩⟩
def transferEvent : Nat := 97244
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 97242 .coefficient) (.value (.predecessor 1 97243 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97242 .coefficient)
      LeftAuthority97240.bound (LeftAuthority97240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97240.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97243 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority97240.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97240.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97240.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97244

namespace LeftBound97248
def owner : Owner := ⟨.program ⟨214⟩, ⟨21968⟩⟩
def transferEvent : Nat := 97248
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97246 .coefficient) (.predecessor 1 97247 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97246 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97247 .coefficient)
      LeftBound97244.bound (LeftBound97244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound97244.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound97244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound97244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97248

namespace LeftBound97249
def owner : Owner := ⟨.program ⟨214⟩, ⟨21968⟩⟩
def transferEvent : Nat := 97249
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21965⟩⟩]⟩ [⟨.result 97241 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97241 .coefficient)
      LeftAuthority97240.bound (LeftAuthority97240.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21965⟩⟩) (rawTerms := some (Proof.Events379.exact97241RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97240.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority97240.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97240.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97249

namespace LeftBound97250
def owner : Owner := ⟨.program ⟨214⟩, ⟨21968⟩⟩
def transferEvent : Nat := 97250
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 97249) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97249)
      LeftBound97249.bound (LeftBound97249.actual selector witness) := by
  exact .transfer (LeftBound97249.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound97249.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound97249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound97249.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97250

namespace LeftBound97321
def owner : Owner := ⟨.program ⟨214⟩, ⟨16372⟩⟩
def transferEvent : Nat := 97321
def frameStart : Nat := 97294
def rule : BoundRule := .identity (.predecessor 0 97320 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97320 .coefficient)
      LeftAuthority97318.bound (LeftAuthority97318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97318.derived selector witness)

def rawBound : CoeffClass := LeftAuthority97318.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority97318.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97321

namespace LeftBound97338
def owner : Owner := ⟨.program ⟨214⟩, ⟨16413⟩⟩
def transferEvent : Nat := 97338
def frameStart : Nat := 97294
def rule : BoundRule := .sum [.predecessor 0 97336 .coefficient, .predecessor 1 97337 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97336 .coefficient)
      LeftBound97321.bound (LeftBound97321.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97337 .coefficient)
      LeftAuthority97334.bound (LeftAuthority97334.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority97334.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97321.bound, LeftAuthority97334.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97321.bound, LeftAuthority97334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97321.actual selector witness, LeftAuthority97334.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97338

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
