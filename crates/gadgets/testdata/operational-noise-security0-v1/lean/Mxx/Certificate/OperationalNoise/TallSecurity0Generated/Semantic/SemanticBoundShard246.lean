import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard245

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound37104
def owner : Owner := ⟨.program ⟨214⟩, ⟨20115⟩⟩
def transferEvent : Nat := 37104
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37102 .coefficient) (.predecessor 1 37103 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37102 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37103 .coefficient)
      LeftBound37100.bound (LeftBound37100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37100.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound37100.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound37100.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound37100.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37104

namespace LeftBound37105
def owner : Owner := ⟨.program ⟨214⟩, ⟨20115⟩⟩
def transferEvent : Nat := 37105
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20112⟩⟩]⟩ [⟨.result 37097 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37097 .coefficient)
      LeftAuthority37096.bound (LeftAuthority37096.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20112⟩⟩) (rawTerms := some (Proof.Events144.exact37097RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37096.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37096.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37096.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37096.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37105

namespace LeftBound37106
def owner : Owner := ⟨.program ⟨214⟩, ⟨20115⟩⟩
def transferEvent : Nat := 37106
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 37105) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 37105)
      LeftBound37105.bound (LeftBound37105.actual selector witness) := by
  exact .transfer (LeftBound37105.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound37105.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound37105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound37105.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37106

namespace LeftBound37185
def owner : Owner := ⟨.program ⟨214⟩, ⟨12975⟩⟩
def transferEvent : Nat := 37185
def frameStart : Nat := 37156
def rule : BoundRule := .product (.predecessor 0 37183 .coefficient) (.predecessor 1 37184 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37183 .coefficient)
      LeftAuthority37181.bound (LeftAuthority37181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37181.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37184 .coefficient)
      LeftAuthority37178.bound (LeftAuthority37178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37178.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37178.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority37181.bound LeftAuthority37178.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37181.bound, LeftAuthority37178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority37181.actual selector witness) * (LeftAuthority37178.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37185

namespace LeftBound37189
def owner : Owner := ⟨.program ⟨214⟩, ⟨12976⟩⟩
def transferEvent : Nat := 37189
def frameStart : Nat := 37156
def rule : BoundRule := .identity (.predecessor 0 37188 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37188 .coefficient)
      LeftBound37185.bound (LeftBound37185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37185.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37185.derived selector witness)

def rawBound : CoeffClass := LeftBound37185.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound37185.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37189

namespace LeftBound37206
def owner : Owner := ⟨.program ⟨214⟩, ⟨13062⟩⟩
def transferEvent : Nat := 37206
def frameStart : Nat := 37156
def rule : BoundRule := .sum [.predecessor 0 37204 .coefficient, .predecessor 1 37205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37204 .coefficient)
      LeftBound37189.bound (LeftBound37189.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound37189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37205 .coefficient)
      LeftAuthority37202.bound (LeftAuthority37202.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority37202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37189.bound, LeftAuthority37202.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37189.bound, LeftAuthority37202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37189.actual selector witness, LeftAuthority37202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37206

namespace LeftBound37209
def owner : Owner := ⟨.program ⟨214⟩, ⟨13063⟩⟩
def transferEvent : Nat := 37209
def frameStart : Nat := 37156
def rule : BoundRule := .identity (.predecessor 0 37208 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37208 .coefficient)
      LeftBound37206.bound (LeftBound37206.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound37206.derived selector witness)

def rawBound : CoeffClass := LeftBound37206.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound37206.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37209

namespace LeftBound37215
def owner : Owner := ⟨.program ⟨214⟩, ⟨13064⟩⟩
def transferEvent : Nat := 37215
def frameStart : Nat := 37156
def rule : BoundRule := .product (.predecessor 0 37213 .coefficient) (.predecessor 1 37214 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37213 .coefficient)
      LeftAuthority37211.bound (LeftAuthority37211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37214 .coefficient)
      LeftBound37209.bound (LeftBound37209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37209.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority37211.bound LeftBound37209.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37211.bound, LeftBound37209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority37211.actual selector witness) * (LeftBound37209.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37215

namespace LeftBound37231
def owner : Owner := ⟨.program ⟨214⟩, ⟨7877⟩⟩
def transferEvent : Nat := 37231
def frameStart : Nat := 37156
def rule : BoundRule := .scale (.predecessor 0 37229 .coefficient) (.value (.predecessor 1 37230 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37229 .coefficient)
      LeftAuthority37227.bound (LeftAuthority37227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37230 .coefficient)
      LeftAuthority37218.bound (LeftAuthority37218.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority37218.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority37227.bound LeftAuthority37218.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37227.bound, LeftAuthority37218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37227.actual selector witness) * (LeftAuthority37218.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound37231

namespace LeftBound37234
def owner : Owner := ⟨.program ⟨214⟩, ⟨6768⟩⟩
def transferEvent : Nat := 37234
def frameStart : Nat := 37156
def rule : BoundRule := .identity (.predecessor 0 37233 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37233 .coefficient)
      LeftAuthority37221.bound (LeftAuthority37221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37221.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37221.derived selector witness)

def rawBound : CoeffClass := LeftAuthority37221.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority37221.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37234

namespace LeftBound37238
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def transferEvent : Nat := 37238
def frameStart : Nat := 37156
def rule : BoundRule := .product (.predecessor 0 37236 .coefficient) (.predecessor 1 37237 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37236 .coefficient)
      LeftBound37234.bound (LeftBound37234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37234.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37237 .coefficient)
      LeftBound37231.bound (LeftBound37231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37234.bound LeftBound37231.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37234.bound, LeftBound37231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37234.actual selector witness) * (LeftBound37231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37238

namespace LeftBound37243
def owner : Owner := ⟨.program ⟨214⟩, ⟨13065⟩⟩
def transferEvent : Nat := 37243
def frameStart : Nat := 37156
def rule : BoundRule := .sum [.predecessor 0 37241 .coefficient, .predecessor 1 37242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37241 .coefficient)
      LeftBound37238.bound (LeftBound37238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37242 .coefficient)
      LeftBound37215.bound (LeftBound37215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37215.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37238.bound, LeftBound37215.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37238.bound, LeftBound37215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37238.actual selector witness, LeftBound37215.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37243

namespace LeftBound37247
def owner : Owner := ⟨.program ⟨214⟩, ⟨25617⟩⟩
def transferEvent : Nat := 37247
def frameStart : Nat := 37156
def rule : BoundRule := .product (.predecessor 0 37245 .coefficient) (.predecessor 1 37246 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37245 .coefficient)
      LeftBound37243.bound (LeftBound37243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37246 .coefficient)
      LeftAuthority37200.bound (LeftAuthority37200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37243.bound LeftAuthority37200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37243.bound, LeftAuthority37200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37243.actual selector witness) * (LeftAuthority37200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37247

namespace LeftBound37258
def owner : Owner := ⟨.program ⟨214⟩, ⟨16762⟩⟩
def transferEvent : Nat := 37258
def frameStart : Nat := 37156
def rule : BoundRule := .product (.predecessor 0 37256 .coefficient) (.predecessor 1 37257 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37256 .coefficient)
      LeftAuthority37211.bound (LeftAuthority37211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37257 .coefficient)
      LeftAuthority37254.bound (LeftAuthority37254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37254.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37254.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority37211.bound LeftAuthority37254.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37211.bound, LeftAuthority37254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority37211.actual selector witness) * (LeftAuthority37254.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37258

namespace LeftBound37266
def owner : Owner := ⟨.program ⟨214⟩, ⟨16763⟩⟩
def transferEvent : Nat := 37266
def frameStart : Nat := 37156
def rule : BoundRule := .sum [.predecessor 0 37264 .coefficient, .predecessor 1 37265 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37264 .coefficient)
      LeftAuthority37262.bound (LeftAuthority37262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37265 .coefficient)
      LeftBound37258.bound (LeftBound37258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37258.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority37262.bound, LeftBound37258.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37262.bound, LeftBound37258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority37262.actual selector witness, LeftBound37258.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37266

namespace LeftBound37270
def owner : Owner := ⟨.program ⟨214⟩, ⟨25618⟩⟩
def transferEvent : Nat := 37270
def frameStart : Nat := 37156
def rule : BoundRule := .sum [.predecessor 0 37268 .coefficient, .predecessor 1 37269 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37268 .coefficient)
      LeftBound37266.bound (LeftBound37266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37269 .coefficient)
      LeftBound37247.bound (LeftBound37247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37247.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37266.bound, LeftBound37247.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37266.bound, LeftBound37247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37266.actual selector witness, LeftBound37247.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37270

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
