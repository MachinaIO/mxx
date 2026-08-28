import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard143

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound22446
def owner : Owner := ⟨.program ⟨214⟩, ⟨10154⟩⟩
def transferEvent : Nat := 22446
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22444 .coefficient) (.predecessor 1 22445 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22444 .coefficient)
      LeftBound22440.bound (LeftBound22440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22445 .coefficient)
      LeftBound7503.bound (LeftBound7503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22440.bound LeftBound7503.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22440.bound, LeftBound7503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22440.actual selector witness) * (LeftBound7503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22446

namespace LeftBound22447
def owner : Owner := ⟨.program ⟨214⟩, ⟨10154⟩⟩
def transferEvent : Nat := 22447
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩ [⟨.result 7500 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7500 .coefficient)
      LeftAuthority7499.bound (LeftAuthority7499.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7876⟩⟩) (rawTerms := some (Proof.Events029.exact7500RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7499.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7499.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7499.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22447

namespace LeftBound22448
def owner : Owner := ⟨.program ⟨214⟩, ⟨10154⟩⟩
def transferEvent : Nat := 22448
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22443 .summary) (.transfer 22447) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22443 .summary)
      LeftBound22441.bound (LeftBound22441.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10153⟩⟩) (rawTerms := some (Proof.Events087.exact22443RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22447)
      LeftBound22447.bound (LeftBound22447.actual selector witness) := by
  exact .transfer (LeftBound22447.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22441.bound LeftBound22447.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22441.bound, LeftBound22447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22441.actual selector witness) * (LeftBound22447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22448

namespace LeftBound22456
def owner : Owner := ⟨.program ⟨214⟩, ⟨12989⟩⟩
def transferEvent : Nat := 22456
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22454 .coefficient, .predecessor 1 22455 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22454 .coefficient)
      LeftBound22446.bound (LeftBound22446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22455 .coefficient)
      LeftBound22418.bound (LeftBound22418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22446.bound, LeftBound22418.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22446.bound, LeftBound22418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22446.actual selector witness, LeftBound22418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22456

namespace LeftBound22458
def owner : Owner := ⟨.program ⟨214⟩, ⟨12989⟩⟩
def transferEvent : Nat := 22458
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 22453 .summary, .result 22423 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22453 .summary)
      LeftBound22448.bound (LeftBound22448.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10154⟩⟩) (rawTerms := some (Proof.Events087.exact22453RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22423 .summary)
      LeftBound22420.bound (LeftBound22420.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12988⟩⟩) (rawTerms := some (Proof.Events087.exact22423RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22420.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22448.bound, LeftBound22420.bound]
def bound : CoeffClass := .finite ⟨95463680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22448.bound, LeftBound22420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22448.actual selector witness, LeftBound22420.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22458

namespace LeftBound22462
def owner : Owner := ⟨.program ⟨214⟩, ⟨25620⟩⟩
def transferEvent : Nat := 22462
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22460 .coefficient) (.predecessor 1 22461 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22460 .coefficient)
      LeftBound22456.bound (LeftBound22456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22461 .coefficient)
      LeftAuthority22394.bound (LeftAuthority22394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22394.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22394.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22456.bound LeftAuthority22394.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22456.bound, LeftAuthority22394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22456.actual selector witness) * (LeftAuthority22394.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22462

namespace LeftBound22463
def owner : Owner := ⟨.program ⟨214⟩, ⟨25620⟩⟩
def transferEvent : Nat := 22463
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25619⟩⟩]⟩ [⟨.result 22395 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22395 .coefficient)
      LeftAuthority22394.bound (LeftAuthority22394.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25619⟩⟩) (rawTerms := some (Proof.Events087.exact22395RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22394.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22394.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22394.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22394.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22463

namespace LeftBound22464
def owner : Owner := ⟨.program ⟨214⟩, ⟨25620⟩⟩
def transferEvent : Nat := 22464
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22459 .summary) (.transfer 22463) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22459 .summary)
      LeftBound22458.bound (LeftBound22458.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12989⟩⟩) (rawTerms := some (Proof.Events087.exact22459RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22463)
      LeftBound22463.bound (LeftBound22463.actual selector witness) := by
  exact .transfer (LeftBound22463.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22458.bound LeftBound22463.bound
def bound : CoeffClass := .finite ⟨350353233018880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22458.bound, LeftBound22463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22458.actual selector witness) * (LeftBound22463.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22464

namespace LeftBound22475
def owner : Owner := ⟨.program ⟨214⟩, ⟨20118⟩⟩
def transferEvent : Nat := 22475
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 22473 .coefficient) (.value (.predecessor 1 22474 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22473 .coefficient)
      LeftAuthority22471.bound (LeftAuthority22471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22471.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22474 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority22471.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22471.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22471.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22475

namespace LeftBound22479
def owner : Owner := ⟨.program ⟨214⟩, ⟨20119⟩⟩
def transferEvent : Nat := 22479
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22477 .coefficient) (.predecessor 1 22478 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22477 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22478 .coefficient)
      LeftBound22475.bound (LeftBound22475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound22475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound22475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound22475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22479

namespace LeftBound22480
def owner : Owner := ⟨.program ⟨214⟩, ⟨20119⟩⟩
def transferEvent : Nat := 22480
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20116⟩⟩]⟩ [⟨.result 22472 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22472 .coefficient)
      LeftAuthority22471.bound (LeftAuthority22471.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20116⟩⟩) (rawTerms := some (Proof.Events087.exact22472RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22471.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22471.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22471.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22471.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22480

namespace LeftBound22481
def owner : Owner := ⟨.program ⟨214⟩, ⟨20119⟩⟩
def transferEvent : Nat := 22481
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 22480) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22480)
      LeftBound22480.bound (LeftBound22480.actual selector witness) := by
  exact .transfer (LeftBound22480.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound22480.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound22480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound22480.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22481

namespace LeftBound22560
def owner : Owner := ⟨.program ⟨214⟩, ⟨12983⟩⟩
def transferEvent : Nat := 22560
def frameStart : Nat := 22531
def rule : BoundRule := .product (.predecessor 0 22558 .coefficient) (.predecessor 1 22559 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22558 .coefficient)
      LeftAuthority22556.bound (LeftAuthority22556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22556.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22559 .coefficient)
      LeftAuthority22553.bound (LeftAuthority22553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22553.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22553.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority22556.bound LeftAuthority22553.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22556.bound, LeftAuthority22553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority22556.actual selector witness) * (LeftAuthority22553.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22560

namespace LeftBound22564
def owner : Owner := ⟨.program ⟨214⟩, ⟨12984⟩⟩
def transferEvent : Nat := 22564
def frameStart : Nat := 22531
def rule : BoundRule := .identity (.predecessor 0 22563 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22563 .coefficient)
      LeftBound22560.bound (LeftBound22560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22560.derived selector witness)

def rawBound : CoeffClass := LeftBound22560.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound22560.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22564

namespace LeftBound22581
def owner : Owner := ⟨.program ⟨214⟩, ⟨13066⟩⟩
def transferEvent : Nat := 22581
def frameStart : Nat := 22531
def rule : BoundRule := .sum [.predecessor 0 22579 .coefficient, .predecessor 1 22580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22579 .coefficient)
      LeftBound22564.bound (LeftBound22564.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound22564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22580 .coefficient)
      LeftAuthority22577.bound (LeftAuthority22577.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority22577.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22564.bound, LeftAuthority22577.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22564.bound, LeftAuthority22577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22564.actual selector witness, LeftAuthority22577.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22581

namespace LeftBound22584
def owner : Owner := ⟨.program ⟨214⟩, ⟨13067⟩⟩
def transferEvent : Nat := 22584
def frameStart : Nat := 22531
def rule : BoundRule := .identity (.predecessor 0 22583 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22583 .coefficient)
      LeftBound22581.bound (LeftBound22581.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound22581.derived selector witness)

def rawBound : CoeffClass := LeftBound22581.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound22581.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22584

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
