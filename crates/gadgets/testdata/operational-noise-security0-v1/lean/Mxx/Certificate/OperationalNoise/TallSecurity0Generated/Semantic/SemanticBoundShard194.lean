import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard091
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard092
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard193

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound29151
def owner : Owner := ⟨.program ⟨214⟩, ⟨7343⟩⟩
def transferEvent : Nat := 29151
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29149 .coefficient) (.predecessor 1 29150 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29149 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29150 .coefficient)
      LeftBound14487.bound (LeftBound14487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14487.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound14487.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound14487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound14487.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29151

namespace LeftBound29156
def owner : Owner := ⟨.program ⟨214⟩, ⟨10704⟩⟩
def transferEvent : Nat := 29156
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29154 .coefficient, .predecessor 1 29155 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29154 .coefficient)
      LeftBound29151.bound (LeftBound29151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29155 .coefficient)
      LeftBound29146.bound (LeftBound29146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29146.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29146.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29151.bound, LeftBound29146.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29151.bound, LeftBound29146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29151.actual selector witness, LeftBound29146.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29156

namespace LeftBound29160
def owner : Owner := ⟨.program ⟨214⟩, ⟨10705⟩⟩
def transferEvent : Nat := 29160
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29158 .coefficient, .predecessor 1 29159 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29158 .coefficient)
      LeftBound29156.bound (LeftBound29156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29159 .coefficient)
      LeftBound14479.bound (LeftBound14479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29156.bound, LeftBound14479.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29156.bound, LeftBound14479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29156.actual selector witness, LeftBound14479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29160

namespace LeftBound29161
def owner : Owner := ⟨.program ⟨214⟩, ⟨10705⟩⟩
def transferEvent : Nat := 29161
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨87⟩⟩]⟩ [⟨.result 14480 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14480 .coefficient)
      LeftBound14479.bound (LeftBound14479.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨87⟩⟩) (rawTerms := some (Proof.Events056.exact14480RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14479.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14479.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29161

namespace LeftBound29166
def owner : Owner := ⟨.program ⟨214⟩, ⟨10706⟩⟩
def transferEvent : Nat := 29166
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29164 .coefficient) (.predecessor 1 29165 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29164 .coefficient)
      LeftBound29160.bound (LeftBound29160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29165 .coefficient)
      LeftAuthority1212.bound (LeftAuthority1212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1212.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1212.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound29160.bound LeftAuthority1212.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29160.bound, LeftAuthority1212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound29160.actual selector witness) * (LeftAuthority1212.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29166

namespace LeftBound29167
def owner : Owner := ⟨.program ⟨214⟩, ⟨10706⟩⟩
def transferEvent : Nat := 29167
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9520⟩⟩], []⟩ [⟨.result 1213 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1213 .coefficient)
      LeftAuthority1212.bound (LeftAuthority1212.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9520⟩⟩) (rawTerms := some (Proof.Events004.exact1213RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1212.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1212.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1212.bound []
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1212.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29167

namespace LeftBound29168
def owner : Owner := ⟨.program ⟨214⟩, ⟨10706⟩⟩
def transferEvent : Nat := 29168
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29163 .summary) (.transfer 29167) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29163 .summary)
      LeftBound29161.bound (LeftBound29161.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10705⟩⟩) (rawTerms := some (Proof.Events113.exact29163RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29161.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29167)
      LeftBound29167.bound (LeftBound29167.actual selector witness) := by
  exact .transfer (LeftBound29167.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound29161.bound LeftBound29167.bound
def bound : CoeffClass := .finite ⟨2496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29161.bound, LeftBound29167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound29161.actual selector witness) * (LeftBound29167.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29168

namespace LeftBound29174
def owner : Owner := ⟨.program ⟨214⟩, ⟨9521⟩⟩
def transferEvent : Nat := 29174
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 29172 .coefficient) (.predecessor 1 29173 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29172 .coefficient)
      LeftAuthority1212.bound (LeftAuthority1212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1212.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29173 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1212.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1212.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1212.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29174

namespace LeftBound29179
def owner : Owner := ⟨.program ⟨214⟩, ⟨7352⟩⟩
def transferEvent : Nat := 29179
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29177 .coefficient) (.predecessor 1 29178 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29177 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29178 .coefficient)
      LeftBound14528.bound (LeftBound14528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14528.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound14528.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound14528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound14528.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29179

namespace LeftBound29184
def owner : Owner := ⟨.program ⟨214⟩, ⟨9522⟩⟩
def transferEvent : Nat := 29184
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29182 .coefficient, .predecessor 1 29183 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29182 .coefficient)
      LeftBound29179.bound (LeftBound29179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29183 .coefficient)
      LeftBound29174.bound (LeftBound29174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29179.bound, LeftBound29174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29179.bound, LeftBound29174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29179.actual selector witness, LeftBound29174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29184

namespace LeftBound29188
def owner : Owner := ⟨.program ⟨214⟩, ⟨9523⟩⟩
def transferEvent : Nat := 29188
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29186 .coefficient, .predecessor 1 29187 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29186 .coefficient)
      LeftBound29184.bound (LeftBound29184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29187 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29184.bound, LeftBound14520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29184.bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29184.actual selector witness, LeftBound14520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29188

namespace LeftBound29189
def owner : Owner := ⟨.program ⟨214⟩, ⟨9523⟩⟩
def transferEvent : Nat := 29189
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨96⟩⟩]⟩ [⟨.result 14521 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14521 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨96⟩⟩) (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14520.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14520.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29189

namespace LeftBound29194
def owner : Owner := ⟨.program ⟨214⟩, ⟨9524⟩⟩
def transferEvent : Nat := 29194
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29192 .coefficient) (.predecessor 1 29193 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29192 .coefficient)
      LeftBound29188.bound (LeftBound29188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29193 .coefficient)
      LeftBound14517.bound (LeftBound14517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29188.bound LeftBound14517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29188.bound, LeftBound14517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29188.actual selector witness) * (LeftBound14517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29194

namespace LeftBound29195
def owner : Owner := ⟨.program ⟨214⟩, ⟨9524⟩⟩
def transferEvent : Nat := 29195
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩ [⟨.result 14514 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14514 .coefficient)
      LeftAuthority14513.bound (LeftAuthority14513.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7834⟩⟩) (rawTerms := some (Proof.Events056.exact14514RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14513.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14513.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14513.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29195

namespace LeftBound29196
def owner : Owner := ⟨.program ⟨214⟩, ⟨9524⟩⟩
def transferEvent : Nat := 29196
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29191 .summary) (.transfer 29195) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29191 .summary)
      LeftBound29189.bound (LeftBound29189.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9523⟩⟩) (rawTerms := some (Proof.Events114.exact29191RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29195)
      LeftBound29195.bound (LeftBound29195.actual selector witness) := by
  exact .transfer (LeftBound29195.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29189.bound LeftBound29195.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29189.bound, LeftBound29195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29189.actual selector witness) * (LeftBound29195.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29196

namespace LeftBound29204
def owner : Owner := ⟨.program ⟨214⟩, ⟨10707⟩⟩
def transferEvent : Nat := 29204
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29202 .coefficient, .predecessor 1 29203 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29202 .coefficient)
      LeftBound29194.bound (LeftBound29194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29203 .coefficient)
      LeftBound29166.bound (LeftBound29166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29194.bound, LeftBound29166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29194.bound, LeftBound29166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29194.actual selector witness, LeftBound29166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29204

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
