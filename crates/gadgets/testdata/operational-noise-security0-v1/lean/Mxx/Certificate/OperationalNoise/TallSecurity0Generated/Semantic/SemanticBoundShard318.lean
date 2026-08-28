import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard268
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard317

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound47976
def owner : Owner := ⟨.program ⟨214⟩, ⟨16347⟩⟩
def transferEvent : Nat := 47976
def frameStart : Nat := 47911
def rule : BoundRule := .product (.predecessor 0 47974 .coefficient) (.predecessor 1 47975 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47974 .coefficient)
      LeftAuthority47972.bound (LeftAuthority47972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47972.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47972.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47975 .coefficient)
      LeftBound47970.bound (LeftBound47970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47970.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority47972.bound LeftBound47970.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47972.bound, LeftBound47970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority47972.actual selector witness) * (LeftBound47970.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47976

namespace LeftBound47984
def owner : Owner := ⟨.program ⟨214⟩, ⟨16348⟩⟩
def transferEvent : Nat := 47984
def frameStart : Nat := 47911
def rule : BoundRule := .sum [.predecessor 0 47982 .coefficient, .predecessor 1 47983 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47982 .coefficient)
      LeftAuthority47980.bound (LeftAuthority47980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47980.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47983 .coefficient)
      LeftBound47976.bound (LeftBound47976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47976.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47980.bound, LeftBound47976.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47980.bound, LeftBound47976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority47980.actual selector witness, LeftBound47976.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47984

namespace LeftBound47988
def owner : Owner := ⟨.program ⟨214⟩, ⟨28537⟩⟩
def transferEvent : Nat := 47988
def frameStart : Nat := 47911
def rule : BoundRule := .product (.predecessor 0 47986 .coefficient) (.predecessor 1 47987 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47986 .coefficient)
      LeftBound47984.bound (LeftBound47984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47987 .coefficient)
      LeftAuthority47961.bound (LeftAuthority47961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47961.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47984.bound LeftAuthority47961.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47984.bound, LeftAuthority47961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47984.actual selector witness) * (LeftAuthority47961.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47988

namespace LeftBound47999
def owner : Owner := ⟨.program ⟨214⟩, ⟨17616⟩⟩
def transferEvent : Nat := 47999
def frameStart : Nat := 47911
def rule : BoundRule := .product (.predecessor 0 47997 .coefficient) (.predecessor 1 47998 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47997 .coefficient)
      LeftAuthority47972.bound (LeftAuthority47972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47972.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47972.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47998 .coefficient)
      LeftAuthority47995.bound (LeftAuthority47995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47995.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47995.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority47972.bound LeftAuthority47995.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47972.bound, LeftAuthority47995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority47972.actual selector witness) * (LeftAuthority47995.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47999

namespace LeftBound48007
def owner : Owner := ⟨.program ⟨214⟩, ⟨17617⟩⟩
def transferEvent : Nat := 48007
def frameStart : Nat := 47911
def rule : BoundRule := .sum [.predecessor 0 48005 .coefficient, .predecessor 1 48006 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48005 .coefficient)
      LeftAuthority48003.bound (LeftAuthority48003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48006 .coefficient)
      LeftBound47999.bound (LeftBound47999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48003.bound, LeftBound47999.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48003.bound, LeftBound47999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48003.actual selector witness, LeftBound47999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48007

namespace LeftBound48011
def owner : Owner := ⟨.program ⟨214⟩, ⟨28542⟩⟩
def transferEvent : Nat := 48011
def frameStart : Nat := 47911
def rule : BoundRule := .sum [.predecessor 0 48009 .coefficient, .predecessor 1 48010 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48009 .coefficient)
      LeftBound48007.bound (LeftBound48007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48010 .coefficient)
      LeftBound47988.bound (LeftBound47988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact47993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47988.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48007.bound, LeftBound47988.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48007.bound, LeftBound47988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48007.actual selector witness, LeftBound47988.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48011

namespace LeftBound48024
def owner : Owner := ⟨.program ⟨214⟩, ⟨28539⟩⟩
def transferEvent : Nat := 48024
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 48022 .coefficient, .predecessor 1 48023 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48022 .coefficient)
      LeftBound47853.bound (LeftBound47853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48023 .coefficient)
      LeftBound47836.bound (LeftBound47836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47853.bound, LeftBound47836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47853.bound, LeftBound47836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47853.actual selector witness, LeftBound47836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48024

namespace LeftBound48027
def owner : Owner := ⟨.program ⟨214⟩, ⟨28539⟩⟩
def transferEvent : Nat := 48027
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 48021 .summary, .result 47843 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48021 .summary)
      LeftBound47855.bound (LeftBound47855.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21771⟩⟩) (rawTerms := some (Proof.Events187.exact48021RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47843 .summary)
      LeftBound47838.bound (LeftBound47838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28538⟩⟩) (rawTerms := some (Proof.Events186.exact47843RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47855.bound, LeftBound47838.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47855.bound, LeftBound47838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47855.actual selector witness, LeftBound47838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48027

namespace LeftBound48031
def owner : Owner := ⟨.program ⟨214⟩, ⟨28540⟩⟩
def transferEvent : Nat := 48031
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48029 .coefficient) (.predecessor 1 48030 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48029 .coefficient)
      LeftBound48024.bound (LeftBound48024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48024.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48030 .coefficient)
      LeftBound5658.bound (LeftBound5658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5658.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48024.bound LeftBound5658.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48024.bound, LeftBound5658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48024.actual selector witness) * (LeftBound5658.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48031

namespace LeftBound48032
def owner : Owner := ⟨.program ⟨214⟩, ⟨28540⟩⟩
def transferEvent : Nat := 48032
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩ [⟨.result 5655 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5655 .coefficient)
      LeftAuthority5654.bound (LeftAuthority5654.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6677⟩⟩) (rawTerms := some (Proof.Events022.exact5655RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5654.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5654.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5654.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48032

namespace LeftBound48033
def owner : Owner := ⟨.program ⟨214⟩, ⟨28540⟩⟩
def transferEvent : Nat := 48033
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 48028 .summary) (.transfer 48032) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48028 .summary)
      LeftBound48027.bound (LeftBound48027.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28539⟩⟩) (rawTerms := some (Proof.Events187.exact48028RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48032)
      LeftBound48032.bound (LeftBound48032.actual selector witness) := by
  exact .transfer (LeftBound48032.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48027.bound LeftBound48032.bound
def bound : CoeffClass := .finite ⟨4742405496644812892115304448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48027.bound, LeftBound48032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48027.actual selector witness) * (LeftBound48032.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48033

namespace LeftBound48048
def owner : Owner := ⟨.program ⟨214⟩, ⟨28321⟩⟩
def transferEvent : Nat := 48048
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48046 .coefficient) (.predecessor 1 48047 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48046 .coefficient)
      LeftBound40175.bound (LeftBound40175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48047 .coefficient)
      LeftAuthority48044.bound (LeftAuthority48044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48044.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40175.bound LeftAuthority48044.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40175.bound, LeftAuthority48044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40175.actual selector witness) * (LeftAuthority48044.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48048

namespace LeftBound48049
def owner : Owner := ⟨.program ⟨214⟩, ⟨28321⟩⟩
def transferEvent : Nat := 48049
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩ [⟨.result 48045 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48045 .coefficient)
      LeftAuthority48044.bound (LeftAuthority48044.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28319⟩⟩) (rawTerms := some (Proof.Events187.exact48045RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48044.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48044.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48044.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48049

namespace LeftBound48050
def owner : Owner := ⟨.program ⟨214⟩, ⟨28321⟩⟩
def transferEvent : Nat := 48050
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40179 .summary) (.transfer 48049) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40179 .summary)
      LeftBound40178.bound (LeftBound40178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26232⟩⟩) (rawTerms := some (Proof.Events156.exact40179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48049)
      LeftBound48049.bound (LeftBound48049.actual selector witness) := by
  exact .transfer (LeftBound48049.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40178.bound LeftBound48049.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40178.bound, LeftBound48049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40178.actual selector witness) * (LeftBound48049.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48050

namespace LeftBound48061
def owner : Owner := ⟨.program ⟨214⟩, ⟨21626⟩⟩
def transferEvent : Nat := 48061
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 48059 .coefficient) (.value (.predecessor 1 48060 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48059 .coefficient)
      LeftAuthority48057.bound (LeftAuthority48057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48060 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority48057.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48057.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48057.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound48061

namespace LeftBound48065
def owner : Owner := ⟨.program ⟨214⟩, ⟨21627⟩⟩
def transferEvent : Nat := 48065
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48063 .coefficient) (.predecessor 1 48064 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48063 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48064 .coefficient)
      LeftBound48061.bound (LeftBound48061.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48061.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48061.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound48061.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound48061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound48061.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48065

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
