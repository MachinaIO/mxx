import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard080
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard588

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound86188
def owner : Owner := ⟨.program ⟨214⟩, ⟨11220⟩⟩
def transferEvent : Nat := 86188
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86186 .coefficient, .predecessor 1 86187 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86186 .coefficient)
      LeftBound86184.bound (LeftBound86184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86187 .coefficient)
      LeftBound12976.bound (LeftBound12976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86184.bound, LeftBound12976.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86184.bound, LeftBound12976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86184.actual selector witness, LeftBound12976.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86188

namespace LeftBound86189
def owner : Owner := ⟨.program ⟨214⟩, ⟨11220⟩⟩
def transferEvent : Nat := 86189
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩ [⟨.result 12977 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12977 .coefficient)
      LeftBound12976.bound (LeftBound12976.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨90⟩⟩) (rawTerms := some (Proof.Events050.exact12977RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12976.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12976.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86189

namespace LeftBound86194
def owner : Owner := ⟨.program ⟨214⟩, ⟨13559⟩⟩
def transferEvent : Nat := 86194
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86192 .coefficient) (.predecessor 1 86193 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86192 .coefficient)
      LeftBound86188.bound (LeftBound86188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86193 .coefficient)
      LeftAuthority4129.bound (LeftAuthority4129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4129.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4129.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound86188.bound LeftAuthority4129.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86188.bound, LeftAuthority4129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound86188.actual selector witness) * (LeftAuthority4129.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86194

namespace LeftBound86195
def owner : Owner := ⟨.program ⟨214⟩, ⟨13559⟩⟩
def transferEvent : Nat := 86195
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13556⟩⟩], []⟩ [⟨.result 4130 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4130 .coefficient)
      LeftAuthority4129.bound (LeftAuthority4129.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13556⟩⟩) (rawTerms := some (Proof.Events016.exact4130RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4129.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4129.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4129.bound []
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4129.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86195

namespace LeftBound86196
def owner : Owner := ⟨.program ⟨214⟩, ⟨13559⟩⟩
def transferEvent : Nat := 86196
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 86191 .summary) (.transfer 86195) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86191 .summary)
      LeftBound86189.bound (LeftBound86189.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11220⟩⟩) (rawTerms := some (Proof.Events336.exact86191RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86195)
      LeftBound86195.bound (LeftBound86195.actual selector witness) := by
  exact .transfer (LeftBound86195.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound86189.bound LeftBound86195.bound
def bound : CoeffClass := .finite ⟨8320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86189.bound, LeftBound86195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound86189.actual selector witness) * (LeftBound86195.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86196

namespace LeftBound86202
def owner : Owner := ⟨.program ⟨214⟩, ⟨13560⟩⟩
def transferEvent : Nat := 86202
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 86200 .coefficient) (.predecessor 1 86201 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86200 .coefficient)
      LeftAuthority4129.bound (LeftAuthority4129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4129.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86201 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4129.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4129.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4129.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86202

namespace LeftBound86207
def owner : Owner := ⟨.program ⟨214⟩, ⟨7249⟩⟩
def transferEvent : Nat := 86207
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86205 .coefficient) (.predecessor 1 86206 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86205 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86206 .coefficient)
      LeftBound13025.bound (LeftBound13025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13025.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound13025.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound13025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound13025.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86207

namespace LeftBound86212
def owner : Owner := ⟨.program ⟨214⟩, ⟨13561⟩⟩
def transferEvent : Nat := 86212
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86210 .coefficient, .predecessor 1 86211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86210 .coefficient)
      LeftBound86207.bound (LeftBound86207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86211 .coefficient)
      LeftBound86202.bound (LeftBound86202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86207.bound, LeftBound86202.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86207.bound, LeftBound86202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86207.actual selector witness, LeftBound86202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86212

namespace LeftBound86216
def owner : Owner := ⟨.program ⟨214⟩, ⟨13562⟩⟩
def transferEvent : Nat := 86216
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86214 .coefficient, .predecessor 1 86215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86214 .coefficient)
      LeftBound86212.bound (LeftBound86212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86215 .coefficient)
      LeftBound13017.bound (LeftBound13017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86212.bound, LeftBound13017.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86212.bound, LeftBound13017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86212.actual selector witness, LeftBound13017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86216

namespace LeftBound86217
def owner : Owner := ⟨.program ⟨214⟩, ⟨13562⟩⟩
def transferEvent : Nat := 86217
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩ [⟨.result 13018 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13018 .coefficient)
      LeftBound13017.bound (LeftBound13017.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨107⟩⟩) (rawTerms := some (Proof.Events050.exact13018RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13017.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13017.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86217

namespace LeftBound86222
def owner : Owner := ⟨.program ⟨214⟩, ⟨13563⟩⟩
def transferEvent : Nat := 86222
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86220 .coefficient) (.predecessor 1 86221 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86220 .coefficient)
      LeftBound86216.bound (LeftBound86216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86221 .coefficient)
      LeftBound13014.bound (LeftBound13014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13014.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86216.bound LeftBound13014.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86216.bound, LeftBound13014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86216.actual selector witness) * (LeftBound13014.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86222

namespace LeftBound86223
def owner : Owner := ⟨.program ⟨214⟩, ⟨13563⟩⟩
def transferEvent : Nat := 86223
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩ [⟨.result 13011 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13011 .coefficient)
      LeftAuthority13010.bound (LeftAuthority13010.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7843⟩⟩) (rawTerms := some (Proof.Events050.exact13011RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13010.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13010.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13010.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86223

namespace LeftBound86224
def owner : Owner := ⟨.program ⟨214⟩, ⟨13563⟩⟩
def transferEvent : Nat := 86224
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 86219 .summary) (.transfer 86223) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86219 .summary)
      LeftBound86217.bound (LeftBound86217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13562⟩⟩) (rawTerms := some (Proof.Events336.exact86219RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86223)
      LeftBound86223.bound (LeftBound86223.actual selector witness) := by
  exact .transfer (LeftBound86223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86217.bound LeftBound86223.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86217.bound, LeftBound86223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86217.actual selector witness) * (LeftBound86223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86224

namespace LeftBound86232
def owner : Owner := ⟨.program ⟨214⟩, ⟨13564⟩⟩
def transferEvent : Nat := 86232
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86230 .coefficient, .predecessor 1 86231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86230 .coefficient)
      LeftBound86222.bound (LeftBound86222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86222.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86231 .coefficient)
      LeftBound86194.bound (LeftBound86194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86222.bound, LeftBound86194.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86222.bound, LeftBound86194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86222.actual selector witness, LeftBound86194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86232

namespace LeftBound86234
def owner : Owner := ⟨.program ⟨214⟩, ⟨13564⟩⟩
def transferEvent : Nat := 86234
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 86229 .summary, .result 86199 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86229 .summary)
      LeftBound86224.bound (LeftBound86224.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13563⟩⟩) (rawTerms := some (Proof.Events336.exact86229RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86199 .summary)
      LeftBound86196.bound (LeftBound86196.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13559⟩⟩) (rawTerms := some (Proof.Events336.exact86199RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86196.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86224.bound, LeftBound86196.bound]
def bound : CoeffClass := .finite ⟨95428736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86224.bound, LeftBound86196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86224.actual selector witness, LeftBound86196.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86234

namespace LeftBound86238
def owner : Owner := ⟨.program ⟨214⟩, ⟨25836⟩⟩
def transferEvent : Nat := 86238
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86236 .coefficient) (.predecessor 1 86237 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86236 .coefficient)
      LeftBound86232.bound (LeftBound86232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86237 .coefficient)
      LeftAuthority86170.bound (LeftAuthority86170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86170.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86170.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86232.bound LeftAuthority86170.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86232.bound, LeftAuthority86170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86232.actual selector witness) * (LeftAuthority86170.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86238

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
