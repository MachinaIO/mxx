import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard171

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26214
def owner : Owner := ⟨.program ⟨214⟩, ⟨28127⟩⟩
def transferEvent : Nat := 26214
def frameStart : Nat := 26114
def rule : BoundRule := .sum [.predecessor 0 26212 .coefficient, .predecessor 1 26213 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26212 .coefficient)
      LeftBound26210.bound (LeftBound26210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26213 .coefficient)
      LeftBound26191.bound (LeftBound26191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26210.bound, LeftBound26191.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26210.bound, LeftBound26191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26210.actual selector witness, LeftBound26191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26214

namespace LeftBound26227
def owner : Owner := ⟨.program ⟨214⟩, ⟨28125⟩⟩
def transferEvent : Nat := 26227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26225 .coefficient, .predecessor 1 26226 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26225 .coefficient)
      LeftBound26056.bound (LeftBound26056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26226 .coefficient)
      LeftBound26039.bound (LeftBound26039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26039.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26056.bound, LeftBound26039.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26056.bound, LeftBound26039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26056.actual selector witness, LeftBound26039.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26227

namespace LeftBound26230
def owner : Owner := ⟨.program ⟨214⟩, ⟨28125⟩⟩
def transferEvent : Nat := 26230
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26224 .summary, .result 26046 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26224 .summary)
      LeftBound26058.bound (LeftBound26058.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21559⟩⟩) (rawTerms := some (Proof.Events102.exact26224RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26046 .summary)
      LeftBound26041.bound (LeftBound26041.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28124⟩⟩) (rawTerms := some (Proof.Events101.exact26046RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26058.bound, LeftBound26041.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26058.bound, LeftBound26041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26058.actual selector witness, LeftBound26041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26230

namespace LeftBound26254
def owner : Owner := ⟨.program ⟨214⟩, ⟨11482⟩⟩
def transferEvent : Nat := 26254
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 26252 .coefficient) (.predecessor 1 26253 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26252 .coefficient)
      LeftAuthority1071.bound (LeftAuthority1071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1071.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26253 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1071.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1071.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1071.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26254

namespace LeftBound26259
def owner : Owner := ⟨.program ⟨214⟩, ⟨7349⟩⟩
def transferEvent : Nat := 26259
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26257 .coefficient) (.predecessor 1 26258 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26257 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26258 .coefficient)
      LeftBound11481.bound (LeftBound11481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound11481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound11481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound11481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26259

namespace LeftBound26264
def owner : Owner := ⟨.program ⟨214⟩, ⟨11483⟩⟩
def transferEvent : Nat := 26264
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26262 .coefficient, .predecessor 1 26263 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26262 .coefficient)
      LeftBound26259.bound (LeftBound26259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26263 .coefficient)
      LeftBound26254.bound (LeftBound26254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26254.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26254.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26259.bound, LeftBound26254.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26259.bound, LeftBound26254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26259.actual selector witness, LeftBound26254.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26264

namespace LeftBound26268
def owner : Owner := ⟨.program ⟨214⟩, ⟨11484⟩⟩
def transferEvent : Nat := 26268
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26266 .coefficient, .predecessor 1 26267 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26266 .coefficient)
      LeftBound26264.bound (LeftBound26264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26267 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26264.bound, LeftBound11473.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26264.bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26264.actual selector witness, LeftBound11473.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26268

namespace LeftBound26269
def owner : Owner := ⟨.program ⟨214⟩, ⟨11484⟩⟩
def transferEvent : Nat := 26269
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩ [⟨.result 11474 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11474 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨93⟩⟩) (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11473.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11473.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26269

namespace LeftBound26274
def owner : Owner := ⟨.program ⟨214⟩, ⟨14237⟩⟩
def transferEvent : Nat := 26274
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26272 .coefficient) (.predecessor 1 26273 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26272 .coefficient)
      LeftBound26268.bound (LeftBound26268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26273 .coefficient)
      LeftAuthority1074.bound (LeftAuthority1074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1074.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound26268.bound LeftAuthority1074.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26268.bound, LeftAuthority1074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound26268.actual selector witness) * (LeftAuthority1074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26274

namespace LeftBound26275
def owner : Owner := ⟨.program ⟨214⟩, ⟨14237⟩⟩
def transferEvent : Nat := 26275
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩ [⟨.result 1075 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1075 .coefficient)
      LeftAuthority1074.bound (LeftAuthority1074.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14234⟩⟩) (rawTerms := some (Proof.Events004.exact1075RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1074.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1074.bound []
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1074.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26275

namespace LeftBound26276
def owner : Owner := ⟨.program ⟨214⟩, ⟨14237⟩⟩
def transferEvent : Nat := 26276
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26271 .summary) (.transfer 26275) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26271 .summary)
      LeftBound26269.bound (LeftBound26269.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11484⟩⟩) (rawTerms := some (Proof.Events102.exact26271RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26275)
      LeftBound26275.bound (LeftBound26275.actual selector witness) := by
  exact .transfer (LeftBound26275.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26269.bound LeftBound26275.bound
def bound : CoeffClass := .finite ⟨14976, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26269.bound, LeftBound26275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26269.actual selector witness) * (LeftBound26275.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26276

namespace LeftBound26282
def owner : Owner := ⟨.program ⟨214⟩, ⟨14238⟩⟩
def transferEvent : Nat := 26282
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 26280 .coefficient) (.predecessor 1 26281 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26280 .coefficient)
      LeftAuthority1074.bound (LeftAuthority1074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26281 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1074.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1074.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1074.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26282

namespace LeftBound26287
def owner : Owner := ⟨.program ⟨214⟩, ⟨7329⟩⟩
def transferEvent : Nat := 26287
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26285 .coefficient) (.predecessor 1 26286 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26285 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26286 .coefficient)
      LeftBound11522.bound (LeftBound11522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound11522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound11522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound11522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26287

namespace LeftBound26292
def owner : Owner := ⟨.program ⟨214⟩, ⟨14239⟩⟩
def transferEvent : Nat := 26292
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26290 .coefficient, .predecessor 1 26291 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26290 .coefficient)
      LeftBound26287.bound (LeftBound26287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26291 .coefficient)
      LeftBound26282.bound (LeftBound26282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26287.bound, LeftBound26282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26287.bound, LeftBound26282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26287.actual selector witness, LeftBound26282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26292

namespace LeftBound26296
def owner : Owner := ⟨.program ⟨214⟩, ⟨14240⟩⟩
def transferEvent : Nat := 26296
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26294 .coefficient, .predecessor 1 26295 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26294 .coefficient)
      LeftBound26292.bound (LeftBound26292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26292.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26292.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26295 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26292.bound, LeftBound11514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26292.bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26292.actual selector witness, LeftBound11514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26296

namespace LeftBound26297
def owner : Owner := ⟨.program ⟨214⟩, ⟨14240⟩⟩
def transferEvent : Nat := 26297
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
end LeftBound26297

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
