import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard023
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard334

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50302
def owner : Owner := ⟨.program ⟨214⟩, ⟨7901⟩⟩
def transferEvent : Nat := 50302
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50297 .summary) (.transfer 50301) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50297 .summary)
      LeftBound50295.bound (LeftBound50295.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7724⟩⟩) (rawTerms := some (Proof.Events196.exact50297RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50301)
      LeftBound50301.bound (LeftBound50301.actual selector witness) := by
  exact .transfer (LeftBound50301.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50295.bound LeftBound50301.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50295.bound, LeftBound50301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50295.actual selector witness) * (LeftBound50301.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50302

namespace LeftBound50328
def owner : Owner := ⟨.program ⟨214⟩, ⟨30171⟩⟩
def transferEvent : Nat := 50328
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50326 .coefficient, .predecessor 1 50327 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50326 .coefficient)
      LeftBound50300.bound (LeftBound50300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50327 .coefficient)
      LeftBound50278.bound (LeftBound50278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50278.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50300.bound, LeftBound50278.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50300.bound, LeftBound50278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50300.actual selector witness, LeftBound50278.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50328

namespace LeftBound50348
def owner : Owner := ⟨.program ⟨214⟩, ⟨30171⟩⟩
def transferEvent : Nat := 50348
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50325 .summary, .result 50280 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50325 .summary)
      LeftBound50302.bound (LeftBound50302.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7901⟩⟩) (rawTerms := some (Proof.Events196.exact50325RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50280 .summary)
      LeftBound50279.bound (LeftBound50279.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30170⟩⟩) (rawTerms := some (Proof.Events196.exact50280RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50279.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50302.bound, LeftBound50279.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789483581492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50302.bound, LeftBound50279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50302.actual selector witness, LeftBound50279.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50348

namespace LeftBound50352
def owner : Owner := ⟨.program ⟨214⟩, ⟨30172⟩⟩
def transferEvent : Nat := 50352
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50350 .coefficient) (.predecessor 1 50351 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50350 .coefficient)
      LeftBound50328.bound (LeftBound50328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50351 .coefficient)
      LeftBound6040.bound (LeftBound6040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50328.bound LeftBound6040.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50328.bound, LeftBound6040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50328.actual selector witness) * (LeftBound6040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50352

namespace LeftBound50353
def owner : Owner := ⟨.program ⟨214⟩, ⟨30172⟩⟩
def transferEvent : Nat := 50353
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7823⟩⟩]⟩ [⟨.result 6037 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6037 .coefficient)
      LeftAuthority6036.bound (LeftAuthority6036.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7823⟩⟩) (rawTerms := some (Proof.Events023.exact6037RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6036.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6036.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6036.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6036.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50353

namespace LeftBound50354
def owner : Owner := ⟨.program ⟨214⟩, ⟨30172⟩⟩
def transferEvent : Nat := 50354
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50349 .summary) (.transfer 50353) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50349 .summary)
      LeftBound50348.bound (LeftBound50348.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30171⟩⟩) (rawTerms := some (Proof.Events196.exact50349RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50353)
      LeftBound50353.bound (LeftBound50353.actual selector witness) := by
  exact .transfer (LeftBound50353.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50348.bound LeftBound50353.bound
def bound : CoeffClass := .finite ⟨1149729608724517268372876178953375812943872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50348.bound, LeftBound50353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50348.actual selector witness) * (LeftBound50353.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50354

namespace LeftBound50416
def owner : Owner := ⟨.program ⟨214⟩, ⟨30173⟩⟩
def transferEvent : Nat := 50416
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50414 .coefficient, .predecessor 1 50415 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50414 .coefficient)
      LeftBound50352.bound (LeftBound50352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50415 .coefficient)
      LeftBound35933.bound (LeftBound35933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35933.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35933.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50352.bound, LeftBound35933.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50352.bound, LeftBound35933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50352.actual selector witness, LeftBound35933.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50416

namespace LeftBound50436
def owner : Owner := ⟨.program ⟨214⟩, ⟨30173⟩⟩
def transferEvent : Nat := 50436
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50413 .summary, .result 36010 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50413 .summary)
      LeftBound50354.bound (LeftBound50354.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30172⟩⟩) (rawTerms := some (Proof.Events196.exact50413RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36010 .summary)
      LeftBound35971.bound (LeftBound35971.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18876⟩⟩) (rawTerms := some (Proof.Events140.exact36010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50354.bound, LeftBound35971.bound]
def bound : CoeffClass := .finite ⟨1149729608724524008718218297164355856419136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50354.bound, LeftBound35971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50354.actual selector witness, LeftBound35971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50436

namespace LeftBound50440
def owner : Owner := ⟨.program ⟨214⟩, ⟨30174⟩⟩
def transferEvent : Nat := 50440
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50438 .coefficient) (.predecessor 1 50439 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50438 .coefficient)
      LeftBound50416.bound (LeftBound50416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50439 .coefficient)
      LeftBound6030.bound (LeftBound6030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6030.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50416.bound LeftBound6030.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50416.bound, LeftBound6030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50416.actual selector witness) * (LeftBound6030.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50440

namespace LeftBound50441
def owner : Owner := ⟨.program ⟨214⟩, ⟨30174⟩⟩
def transferEvent : Nat := 50441
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩ [⟨.result 6027 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6027 .coefficient)
      LeftAuthority6026.bound (LeftAuthority6026.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6653⟩⟩) (rawTerms := some (Proof.Events023.exact6027RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6026.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6026.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6026.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6026.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50441

namespace LeftBound50442
def owner : Owner := ⟨.program ⟨214⟩, ⟨30174⟩⟩
def transferEvent : Nat := 50442
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50437 .summary) (.transfer 50441) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50437 .summary)
      LeftBound50436.bound (LeftBound50436.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30173⟩⟩) (rawTerms := some (Proof.Events197.exact50437RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50436.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50441)
      LeftBound50441.bound (LeftBound50441.actual selector witness) := by
  exact .transfer (LeftBound50441.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50436.bound LeftBound50441.bound
def bound : CoeffClass := .finite ⟨4219526059692742704380000642085940622751931826176, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50436.bound, LeftBound50441.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50436.actual selector witness) * (LeftBound50441.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50442

namespace LeftBound50523
def owner : Owner := ⟨.program ⟨214⟩, ⟨5610⟩⟩
def transferEvent : Nat := 50523
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 50518 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50518 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50523

namespace LeftBound50527
def owner : Owner := ⟨.program ⟨214⟩, ⟨6580⟩⟩
def transferEvent : Nat := 50527
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50525 .coefficient) (.predecessor 1 50526 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50525 .coefficient)
      LeftBound50523.bound (LeftBound50523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50526 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50523.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50523.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50523.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50527

namespace LeftBound50539
def owner : Owner := ⟨.program ⟨214⟩, ⟨5545⟩⟩
def transferEvent : Nat := 50539
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 50534 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50534 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50539

namespace LeftBound50543
def owner : Owner := ⟨.program ⟨214⟩, ⟨7251⟩⟩
def transferEvent : Nat := 50543
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50541 .coefficient) (.predecessor 1 50542 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50541 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50542 .coefficient)
      LeftAuthority6073.bound (LeftAuthority6073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6073.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6073.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftAuthority6073.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftAuthority6073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftAuthority6073.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50543

namespace LeftBound50548
def owner : Owner := ⟨.program ⟨214⟩, ⟨7755⟩⟩
def transferEvent : Nat := 50548
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50546 .coefficient, .predecessor 1 50547 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50546 .coefficient)
      LeftBound50543.bound (LeftBound50543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50547 .coefficient)
      LeftBound50527.bound (LeftBound50527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50527.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50543.bound, LeftBound50527.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50543.bound, LeftBound50527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50543.actual selector witness, LeftBound50527.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50548

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
