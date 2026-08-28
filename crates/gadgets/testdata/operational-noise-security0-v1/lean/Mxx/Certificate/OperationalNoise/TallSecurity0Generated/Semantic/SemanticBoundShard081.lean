import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard080

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound13038
def owner : Owner := ⟨.program ⟨214⟩, ⟨13598⟩⟩
def transferEvent : Nat := 13038
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13036 .coefficient, .predecessor 1 13037 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13036 .coefficient)
      LeftBound13034.bound (LeftBound13034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13034.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13037 .coefficient)
      LeftBound13017.bound (LeftBound13017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13034.bound, LeftBound13017.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13034.bound, LeftBound13017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13034.actual selector witness, LeftBound13017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13038

namespace LeftBound13039
def owner : Owner := ⟨.program ⟨214⟩, ⟨13598⟩⟩
def transferEvent : Nat := 13039
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
end LeftBound13039

namespace LeftBound13044
def owner : Owner := ⟨.program ⟨214⟩, ⟨13599⟩⟩
def transferEvent : Nat := 13044
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13042 .coefficient) (.predecessor 1 13043 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13042 .coefficient)
      LeftBound13038.bound (LeftBound13038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13038.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13043 .coefficient)
      LeftBound13014.bound (LeftBound13014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13014.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13038.bound LeftBound13014.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13038.bound, LeftBound13014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13038.actual selector witness) * (LeftBound13014.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13044

namespace LeftBound13045
def owner : Owner := ⟨.program ⟨214⟩, ⟨13599⟩⟩
def transferEvent : Nat := 13045
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
end LeftBound13045

namespace LeftBound13046
def owner : Owner := ⟨.program ⟨214⟩, ⟨13599⟩⟩
def transferEvent : Nat := 13046
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 13041 .summary) (.transfer 13045) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13041 .summary)
      LeftBound13039.bound (LeftBound13039.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13598⟩⟩) (rawTerms := some (Proof.Events050.exact13041RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13039.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13045)
      LeftBound13045.bound (LeftBound13045.actual selector witness) := by
  exact .transfer (LeftBound13045.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13039.bound LeftBound13045.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13039.bound, LeftBound13045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13039.actual selector witness) * (LeftBound13045.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13046

namespace LeftBound13054
def owner : Owner := ⟨.program ⟨214⟩, ⟨13600⟩⟩
def transferEvent : Nat := 13054
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13052 .coefficient, .predecessor 1 13053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13052 .coefficient)
      LeftBound13044.bound (LeftBound13044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13044.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13053 .coefficient)
      LeftBound13003.bound (LeftBound13003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13044.bound, LeftBound13003.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13044.bound, LeftBound13003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13044.actual selector witness, LeftBound13003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13054

namespace LeftBound13056
def owner : Owner := ⟨.program ⟨214⟩, ⟨13600⟩⟩
def transferEvent : Nat := 13056
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 13051 .summary, .result 13008 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13051 .summary)
      LeftBound13046.bound (LeftBound13046.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13599⟩⟩) (rawTerms := some (Proof.Events050.exact13051RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13008 .summary)
      LeftBound13005.bound (LeftBound13005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13595⟩⟩) (rawTerms := some (Proof.Events050.exact13008RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13046.bound, LeftBound13005.bound]
def bound : CoeffClass := .finite ⟨95428736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13046.bound, LeftBound13005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13046.actual selector witness, LeftBound13005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13056

namespace LeftBound13060
def owner : Owner := ⟨.program ⟨214⟩, ⟨25856⟩⟩
def transferEvent : Nat := 13060
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13058 .coefficient) (.predecessor 1 13059 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13058 .coefficient)
      LeftBound13054.bound (LeftBound13054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13059 .coefficient)
      LeftAuthority12973.bound (LeftAuthority12973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12973.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13054.bound LeftAuthority12973.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13054.bound, LeftAuthority12973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13054.actual selector witness) * (LeftAuthority12973.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13060

namespace LeftBound13061
def owner : Owner := ⟨.program ⟨214⟩, ⟨25856⟩⟩
def transferEvent : Nat := 13061
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25855⟩⟩]⟩ [⟨.result 12974 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12974 .coefficient)
      LeftAuthority12973.bound (LeftAuthority12973.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25855⟩⟩) (rawTerms := some (Proof.Events050.exact12974RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12973.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12973.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12973.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13061

namespace LeftBound13062
def owner : Owner := ⟨.program ⟨214⟩, ⟨25856⟩⟩
def transferEvent : Nat := 13062
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 13057 .summary) (.transfer 13061) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13057 .summary)
      LeftBound13056.bound (LeftBound13056.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13600⟩⟩) (rawTerms := some (Proof.Events051.exact13057RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13061)
      LeftBound13061.bound (LeftBound13061.actual selector witness) := by
  exact .transfer (LeftBound13061.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13056.bound LeftBound13061.bound
def bound : CoeffClass := .finite ⟨350224987979776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13056.bound, LeftBound13061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13056.actual selector witness) * (LeftBound13061.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13062

namespace LeftBound13073
def owner : Owner := ⟨.program ⟨214⟩, ⟨19330⟩⟩
def transferEvent : Nat := 13073
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 13071 .coefficient) (.value (.predecessor 1 13072 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13071 .coefficient)
      LeftAuthority13069.bound (LeftAuthority13069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13069.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13072 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority13069.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13069.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13069.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound13073

namespace LeftBound13077
def owner : Owner := ⟨.program ⟨214⟩, ⟨19331⟩⟩
def transferEvent : Nat := 13077
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13075 .coefficient) (.predecessor 1 13076 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13075 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13076 .coefficient)
      LeftBound13073.bound (LeftBound13073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13073.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound13073.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound13073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound13073.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13077

namespace LeftBound13078
def owner : Owner := ⟨.program ⟨214⟩, ⟨19331⟩⟩
def transferEvent : Nat := 13078
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19328⟩⟩]⟩ [⟨.result 13070 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13070 .coefficient)
      LeftAuthority13069.bound (LeftAuthority13069.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19328⟩⟩) (rawTerms := some (Proof.Events051.exact13070RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13069.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13069.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13069.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13069.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13078

namespace LeftBound13079
def owner : Owner := ⟨.program ⟨214⟩, ⟨19331⟩⟩
def transferEvent : Nat := 13079
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 13078) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13078)
      LeftBound13078.bound (LeftBound13078.actual selector witness) := by
  exact .transfer (LeftBound13078.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound13078.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound13078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound13078.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13079

namespace LeftBound13158
def owner : Owner := ⟨.program ⟨214⟩, ⟨13593⟩⟩
def transferEvent : Nat := 13158
def frameStart : Nat := 13129
def rule : BoundRule := .product (.predecessor 0 13156 .coefficient) (.predecessor 1 13157 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13156 .coefficient)
      LeftAuthority13154.bound (LeftAuthority13154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13157 .coefficient)
      LeftAuthority13151.bound (LeftAuthority13151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13151.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13154.bound LeftAuthority13151.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13154.bound, LeftAuthority13151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority13154.actual selector witness) * (LeftAuthority13151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13158

namespace LeftBound13162
def owner : Owner := ⟨.program ⟨214⟩, ⟨13594⟩⟩
def transferEvent : Nat := 13162
def frameStart : Nat := 13129
def rule : BoundRule := .identity (.predecessor 0 13161 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13161 .coefficient)
      LeftBound13158.bound (LeftBound13158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13158.derived selector witness)

def rawBound : CoeffClass := LeftBound13158.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound13158.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13162

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
