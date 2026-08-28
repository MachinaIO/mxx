import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard084
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard186

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28202
def owner : Owner := ⟨.program ⟨214⟩, ⟨12193⟩⟩
def transferEvent : Nat := 28202
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28200 .coefficient) (.predecessor 1 28201 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28200 .coefficient)
      LeftBound28196.bound (LeftBound28196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28201 .coefficient)
      LeftAuthority1166.bound (LeftAuthority1166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1166.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound28196.bound LeftAuthority1166.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28196.bound, LeftAuthority1166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound28196.actual selector witness) * (LeftAuthority1166.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28202

namespace LeftBound28203
def owner : Owner := ⟨.program ⟨214⟩, ⟨12193⟩⟩
def transferEvent : Nat := 28203
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩ [⟨.result 1167 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1167 .coefficient)
      LeftAuthority1166.bound (LeftAuthority1166.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨12190⟩⟩) (rawTerms := some (Proof.Events004.exact1167RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1166.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1166.bound []
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1166.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28203

namespace LeftBound28204
def owner : Owner := ⟨.program ⟨214⟩, ⟨12193⟩⟩
def transferEvent : Nat := 28204
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28199 .summary) (.transfer 28203) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28199 .summary)
      LeftBound28197.bound (LeftBound28197.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11148⟩⟩) (rawTerms := some (Proof.Events110.exact28199RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28203)
      LeftBound28203.bound (LeftBound28203.actual selector witness) := by
  exact .transfer (LeftBound28203.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound28197.bound LeftBound28203.bound
def bound : CoeffClass := .finite ⟨4992, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28197.bound, LeftBound28203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound28197.actual selector witness) * (LeftBound28203.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28204

namespace LeftBound28210
def owner : Owner := ⟨.program ⟨214⟩, ⟨12194⟩⟩
def transferEvent : Nat := 28210
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 28208 .coefficient) (.predecessor 1 28209 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28208 .coefficient)
      LeftAuthority1166.bound (LeftAuthority1166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28209 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1166.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1166.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1166.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28210

namespace LeftBound28215
def owner : Owner := ⟨.program ⟨214⟩, ⟨7362⟩⟩
def transferEvent : Nat := 28215
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28213 .coefficient) (.predecessor 1 28214 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28213 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28214 .coefficient)
      LeftBound13526.bound (LeftBound13526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13526.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound13526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound13526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound13526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28215

namespace LeftBound28220
def owner : Owner := ⟨.program ⟨214⟩, ⟨12195⟩⟩
def transferEvent : Nat := 28220
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28218 .coefficient, .predecessor 1 28219 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28218 .coefficient)
      LeftBound28215.bound (LeftBound28215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28219 .coefficient)
      LeftBound28210.bound (LeftBound28210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28215.bound, LeftBound28210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28215.bound, LeftBound28210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28215.actual selector witness, LeftBound28210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28220

namespace LeftBound28224
def owner : Owner := ⟨.program ⟨214⟩, ⟨12196⟩⟩
def transferEvent : Nat := 28224
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28222 .coefficient, .predecessor 1 28223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28222 .coefficient)
      LeftBound28220.bound (LeftBound28220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28223 .coefficient)
      LeftBound13518.bound (LeftBound13518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28220.bound, LeftBound13518.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28220.bound, LeftBound13518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28220.actual selector witness, LeftBound13518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28224

namespace LeftBound28225
def owner : Owner := ⟨.program ⟨214⟩, ⟨12196⟩⟩
def transferEvent : Nat := 28225
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩ [⟨.result 13519 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13519 .coefficient)
      LeftBound13518.bound (LeftBound13518.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨106⟩⟩) (rawTerms := some (Proof.Events052.exact13519RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13518.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13518.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28225

namespace LeftBound28230
def owner : Owner := ⟨.program ⟨214⟩, ⟨12197⟩⟩
def transferEvent : Nat := 28230
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28228 .coefficient) (.predecessor 1 28229 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28228 .coefficient)
      LeftBound28224.bound (LeftBound28224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28229 .coefficient)
      LeftBound13515.bound (LeftBound13515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28224.bound LeftBound13515.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28224.bound, LeftBound13515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28224.actual selector witness) * (LeftBound13515.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28230

namespace LeftBound28231
def owner : Owner := ⟨.program ⟨214⟩, ⟨12197⟩⟩
def transferEvent : Nat := 28231
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩ [⟨.result 13512 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13512 .coefficient)
      LeftAuthority13511.bound (LeftAuthority13511.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7840⟩⟩) (rawTerms := some (Proof.Events052.exact13512RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13511.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13511.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13511.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13511.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28231

namespace LeftBound28232
def owner : Owner := ⟨.program ⟨214⟩, ⟨12197⟩⟩
def transferEvent : Nat := 28232
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28227 .summary) (.transfer 28231) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28227 .summary)
      LeftBound28225.bound (LeftBound28225.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12196⟩⟩) (rawTerms := some (Proof.Events110.exact28227RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28231)
      LeftBound28231.bound (LeftBound28231.actual selector witness) := by
  exact .transfer (LeftBound28231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28225.bound LeftBound28231.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28225.bound, LeftBound28231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28225.actual selector witness) * (LeftBound28231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28232

namespace LeftBound28240
def owner : Owner := ⟨.program ⟨214⟩, ⟨12198⟩⟩
def transferEvent : Nat := 28240
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28238 .coefficient, .predecessor 1 28239 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28238 .coefficient)
      LeftBound28230.bound (LeftBound28230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28239 .coefficient)
      LeftBound28202.bound (LeftBound28202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28230.bound, LeftBound28202.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28230.bound, LeftBound28202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28230.actual selector witness, LeftBound28202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28240

namespace LeftBound28242
def owner : Owner := ⟨.program ⟨214⟩, ⟨12198⟩⟩
def transferEvent : Nat := 28242
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 28237 .summary, .result 28207 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28237 .summary)
      LeftBound28232.bound (LeftBound28232.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12197⟩⟩) (rawTerms := some (Proof.Events110.exact28237RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28207 .summary)
      LeftBound28204.bound (LeftBound28204.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12193⟩⟩) (rawTerms := some (Proof.Events110.exact28207RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28204.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28232.bound, LeftBound28204.bound]
def bound : CoeffClass := .finite ⟨95425408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28232.bound, LeftBound28204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28232.actual selector witness, LeftBound28204.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28242

namespace LeftBound28246
def owner : Owner := ⟨.program ⟨214⟩, ⟨25312⟩⟩
def transferEvent : Nat := 28246
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28244 .coefficient) (.predecessor 1 28245 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28244 .coefficient)
      LeftBound28240.bound (LeftBound28240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28240.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28240.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28245 .coefficient)
      LeftAuthority28178.bound (LeftAuthority28178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28178.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28178.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28240.bound LeftAuthority28178.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28240.bound, LeftAuthority28178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28240.actual selector witness) * (LeftAuthority28178.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28246

namespace LeftBound28247
def owner : Owner := ⟨.program ⟨214⟩, ⟨25312⟩⟩
def transferEvent : Nat := 28247
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩ [⟨.result 28179 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28179 .coefficient)
      LeftAuthority28178.bound (LeftAuthority28178.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25311⟩⟩) (rawTerms := some (Proof.Events110.exact28179RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28178.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28178.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28178.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28178.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28247

namespace LeftBound28248
def owner : Owner := ⟨.program ⟨214⟩, ⟨25312⟩⟩
def transferEvent : Nat := 28248
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28243 .summary) (.transfer 28247) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28243 .summary)
      LeftBound28242.bound (LeftBound28242.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12198⟩⟩) (rawTerms := some (Proof.Events110.exact28243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28247)
      LeftBound28247.bound (LeftBound28247.actual selector witness) := by
  exact .transfer (LeftBound28247.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28242.bound LeftBound28247.bound
def bound : CoeffClass := .finite ⟨350212774166528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28242.bound, LeftBound28247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28242.actual selector witness) * (LeftBound28247.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28248

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
