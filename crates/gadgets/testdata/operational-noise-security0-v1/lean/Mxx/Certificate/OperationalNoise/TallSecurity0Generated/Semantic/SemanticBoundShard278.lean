import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard277

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound41404
def owner : Owner := ⟨.program ⟨214⟩, ⟨14014⟩⟩
def transferEvent : Nat := 41404
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩ [⟨.result 12016 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12016 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨72⟩⟩) (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12015.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12015.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41404

namespace LeftBound41409
def owner : Owner := ⟨.program ⟨214⟩, ⟨14015⟩⟩
def transferEvent : Nat := 41409
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41407 .coefficient) (.predecessor 1 41408 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41407 .coefficient)
      LeftBound41403.bound (LeftBound41403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41408 .coefficient)
      LeftBound12012.bound (LeftBound12012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41403.bound LeftBound12012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41403.bound, LeftBound12012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41403.actual selector witness) * (LeftBound12012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41409

namespace LeftBound41410
def owner : Owner := ⟨.program ⟨214⟩, ⟨14015⟩⟩
def transferEvent : Nat := 41410
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩ [⟨.result 12009 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12009 .coefficient)
      LeftAuthority12008.bound (LeftAuthority12008.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7849⟩⟩) (rawTerms := some (Proof.Events046.exact12009RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12008.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12008.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12008.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41410

namespace LeftBound41411
def owner : Owner := ⟨.program ⟨214⟩, ⟨14015⟩⟩
def transferEvent : Nat := 41411
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41406 .summary) (.transfer 41410) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41406 .summary)
      LeftBound41404.bound (LeftBound41404.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14014⟩⟩) (rawTerms := some (Proof.Events161.exact41406RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41410)
      LeftBound41410.bound (LeftBound41410.actual selector witness) := by
  exact .transfer (LeftBound41410.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41404.bound LeftBound41410.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41404.bound, LeftBound41410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41404.actual selector witness) * (LeftBound41410.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41411

namespace LeftBound41419
def owner : Owner := ⟨.program ⟨214⟩, ⟨14016⟩⟩
def transferEvent : Nat := 41419
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41417 .coefficient, .predecessor 1 41418 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41417 .coefficient)
      LeftBound41409.bound (LeftBound41409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41418 .coefficient)
      LeftBound41381.bound (LeftBound41381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41381.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41381.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41409.bound, LeftBound41381.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41409.bound, LeftBound41381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41409.actual selector witness, LeftBound41381.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41419

namespace LeftBound41421
def owner : Owner := ⟨.program ⟨214⟩, ⟨14016⟩⟩
def transferEvent : Nat := 41421
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 41416 .summary, .result 41386 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41416 .summary)
      LeftBound41411.bound (LeftBound41411.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14015⟩⟩) (rawTerms := some (Proof.Events161.exact41416RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41386 .summary)
      LeftBound41383.bound (LeftBound41383.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14011⟩⟩) (rawTerms := some (Proof.Events161.exact41386RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41383.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41411.bound, LeftBound41383.bound]
def bound : CoeffClass := .finite ⟨95433728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41411.bound, LeftBound41383.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41411.actual selector witness, LeftBound41383.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41421

namespace LeftBound41425
def owner : Owner := ⟨.program ⟨214⟩, ⟨26000⟩⟩
def transferEvent : Nat := 41425
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41423 .coefficient) (.predecessor 1 41424 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41423 .coefficient)
      LeftBound41419.bound (LeftBound41419.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41419.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41419.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41424 .coefficient)
      LeftAuthority41357.bound (LeftAuthority41357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41357.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41357.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41419.bound LeftAuthority41357.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41419.bound, LeftAuthority41357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41419.actual selector witness) * (LeftAuthority41357.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41425

namespace LeftBound41426
def owner : Owner := ⟨.program ⟨214⟩, ⟨26000⟩⟩
def transferEvent : Nat := 41426
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25999⟩⟩]⟩ [⟨.result 41358 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41358 .coefficient)
      LeftAuthority41357.bound (LeftAuthority41357.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25999⟩⟩) (rawTerms := some (Proof.Events161.exact41358RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41357.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41357.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority41357.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41357.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41426

namespace LeftBound41427
def owner : Owner := ⟨.program ⟨214⟩, ⟨26000⟩⟩
def transferEvent : Nat := 41427
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41422 .summary) (.transfer 41426) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41422 .summary)
      LeftBound41421.bound (LeftBound41421.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14016⟩⟩) (rawTerms := some (Proof.Events161.exact41422RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41426)
      LeftBound41426.bound (LeftBound41426.actual selector witness) := by
  exact .transfer (LeftBound41426.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41421.bound LeftBound41426.bound
def bound : CoeffClass := .finite ⟨350243308699648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41421.bound, LeftBound41426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41421.actual selector witness) * (LeftBound41426.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41427

namespace LeftBound41438
def owner : Owner := ⟨.program ⟨214⟩, ⟨19466⟩⟩
def transferEvent : Nat := 41438
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 41436 .coefficient) (.value (.predecessor 1 41437 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41436 .coefficient)
      LeftAuthority41434.bound (LeftAuthority41434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41437 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority41434.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41434.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41434.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound41438

namespace LeftBound41442
def owner : Owner := ⟨.program ⟨214⟩, ⟨19467⟩⟩
def transferEvent : Nat := 41442
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41440 .coefficient) (.predecessor 1 41441 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41440 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41441 .coefficient)
      LeftBound41438.bound (LeftBound41438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41438.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound41438.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound41438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound41438.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41442

namespace LeftBound41443
def owner : Owner := ⟨.program ⟨214⟩, ⟨19467⟩⟩
def transferEvent : Nat := 41443
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19464⟩⟩]⟩ [⟨.result 41435 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41435 .coefficient)
      LeftAuthority41434.bound (LeftAuthority41434.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19464⟩⟩) (rawTerms := some (Proof.Events161.exact41435RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41434.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority41434.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41434.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41443

namespace LeftBound41444
def owner : Owner := ⟨.program ⟨214⟩, ⟨19467⟩⟩
def transferEvent : Nat := 41444
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 41443) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41443)
      LeftBound41443.bound (LeftBound41443.actual selector witness) := by
  exact .transfer (LeftBound41443.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound41443.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound41443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound41443.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41444

namespace LeftBound41523
def owner : Owner := ⟨.program ⟨214⟩, ⟨14009⟩⟩
def transferEvent : Nat := 41523
def frameStart : Nat := 41494
def rule : BoundRule := .product (.predecessor 0 41521 .coefficient) (.predecessor 1 41522 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41521 .coefficient)
      LeftAuthority41519.bound (LeftAuthority41519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41522 .coefficient)
      LeftAuthority41516.bound (LeftAuthority41516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41516.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority41519.bound LeftAuthority41516.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41519.bound, LeftAuthority41516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority41519.actual selector witness) * (LeftAuthority41516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41523

namespace LeftBound41527
def owner : Owner := ⟨.program ⟨214⟩, ⟨14010⟩⟩
def transferEvent : Nat := 41527
def frameStart : Nat := 41494
def rule : BoundRule := .identity (.predecessor 0 41526 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41526 .coefficient)
      LeftBound41523.bound (LeftBound41523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41523.derived selector witness)

def rawBound : CoeffClass := LeftBound41523.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound41523.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41527

namespace LeftBound41544
def owner : Owner := ⟨.program ⟨214⟩, ⟨14105⟩⟩
def transferEvent : Nat := 41544
def frameStart : Nat := 41494
def rule : BoundRule := .sum [.predecessor 0 41542 .coefficient, .predecessor 1 41543 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41542 .coefficient)
      LeftBound41527.bound (LeftBound41527.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound41527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41543 .coefficient)
      LeftAuthority41540.bound (LeftAuthority41540.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority41540.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41527.bound, LeftAuthority41540.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41527.bound, LeftAuthority41540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41527.actual selector witness, LeftAuthority41540.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41544

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
