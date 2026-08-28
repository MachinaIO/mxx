import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard096

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound15050
def owner : Owner := ⟨.program ⟨214⟩, ⟨9424⟩⟩
def transferEvent : Nat := 15050
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 15045 .summary) (.transfer 15049) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15045 .summary)
      LeftBound15043.bound (LeftBound15043.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9423⟩⟩) (rawTerms := some (Proof.Events058.exact15045RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 15049)
      LeftBound15049.bound (LeftBound15049.actual selector witness) := by
  exact .transfer (LeftBound15049.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15043.bound LeftBound15049.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15043.bound, LeftBound15049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15043.actual selector witness) * (LeftBound15049.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15050

namespace LeftBound15058
def owner : Owner := ⟨.program ⟨214⟩, ⟨10519⟩⟩
def transferEvent : Nat := 15058
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15056 .coefficient, .predecessor 1 15057 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15056 .coefficient)
      LeftBound15048.bound (LeftBound15048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15057 .coefficient)
      LeftBound15007.bound (LeftBound15007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15048.bound, LeftBound15007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15048.bound, LeftBound15007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15048.actual selector witness, LeftBound15007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15058

namespace LeftBound15060
def owner : Owner := ⟨.program ⟨214⟩, ⟨10519⟩⟩
def transferEvent : Nat := 15060
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15055 .summary, .result 15012 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15055 .summary)
      LeftBound15050.bound (LeftBound15050.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9424⟩⟩) (rawTerms := some (Proof.Events058.exact15055RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15012 .summary)
      LeftBound15009.bound (LeftBound15009.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10518⟩⟩) (rawTerms := some (Proof.Events058.exact15012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15050.bound, LeftBound15009.bound]
def bound : CoeffClass := .finite ⟨95422080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15050.bound, LeftBound15009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15050.actual selector witness, LeftBound15009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15060

namespace LeftBound15064
def owner : Owner := ⟨.program ⟨214⟩, ⟨24932⟩⟩
def transferEvent : Nat := 15064
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 15062 .coefficient) (.predecessor 1 15063 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15062 .coefficient)
      LeftBound15058.bound (LeftBound15058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15063 .coefficient)
      LeftAuthority14977.bound (LeftAuthority14977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14977.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15058.bound LeftAuthority14977.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15058.bound, LeftAuthority14977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15058.actual selector witness) * (LeftAuthority14977.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15064

namespace LeftBound15065
def owner : Owner := ⟨.program ⟨214⟩, ⟨24932⟩⟩
def transferEvent : Nat := 15065
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24931⟩⟩]⟩ [⟨.result 14978 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14978 .coefficient)
      LeftAuthority14977.bound (LeftAuthority14977.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24931⟩⟩) (rawTerms := some (Proof.Events058.exact14978RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14977.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14977.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14977.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound15065

namespace LeftBound15066
def owner : Owner := ⟨.program ⟨214⟩, ⟨24932⟩⟩
def transferEvent : Nat := 15066
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 15061 .summary) (.transfer 15065) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15061 .summary)
      LeftBound15060.bound (LeftBound15060.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10519⟩⟩) (rawTerms := some (Proof.Events058.exact15061RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15060.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 15065)
      LeftBound15065.bound (LeftBound15065.actual selector witness) := by
  exact .transfer (LeftBound15065.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15060.bound LeftBound15065.bound
def bound : CoeffClass := .finite ⟨350200560353280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15060.bound, LeftBound15065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15060.actual selector witness) * (LeftBound15065.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15066

namespace LeftBound15077
def owner : Owner := ⟨.program ⟨214⟩, ⟨19042⟩⟩
def transferEvent : Nat := 15077
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 15075 .coefficient) (.value (.predecessor 1 15076 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15075 .coefficient)
      LeftAuthority15073.bound (LeftAuthority15073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15073.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15076 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority15073.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15073.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15073.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound15077

namespace LeftBound15081
def owner : Owner := ⟨.program ⟨214⟩, ⟨19043⟩⟩
def transferEvent : Nat := 15081
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 15079 .coefficient) (.predecessor 1 15080 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15079 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15080 .coefficient)
      LeftBound15077.bound (LeftBound15077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15077.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound15077.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound15077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound15077.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15081

namespace LeftBound15082
def owner : Owner := ⟨.program ⟨214⟩, ⟨19043⟩⟩
def transferEvent : Nat := 15082
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19040⟩⟩]⟩ [⟨.result 15074 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15074 .coefficient)
      LeftAuthority15073.bound (LeftAuthority15073.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19040⟩⟩) (rawTerms := some (Proof.Events058.exact15074RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15073.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15073.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15073.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15073.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound15082

namespace LeftBound15083
def owner : Owner := ⟨.program ⟨214⟩, ⟨19043⟩⟩
def transferEvent : Nat := 15083
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 15082) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 15082)
      LeftBound15082.bound (LeftBound15082.actual selector witness) := by
  exact .transfer (LeftBound15082.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound15082.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound15082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound15082.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15083

namespace LeftBound15162
def owner : Owner := ⟨.program ⟨214⟩, ⟨10513⟩⟩
def transferEvent : Nat := 15162
def frameStart : Nat := 15133
def rule : BoundRule := .product (.predecessor 0 15160 .coefficient) (.predecessor 1 15161 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15160 .coefficient)
      LeftAuthority15158.bound (LeftAuthority15158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15158.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15161 .coefficient)
      LeftAuthority15155.bound (LeftAuthority15155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15155.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority15158.bound LeftAuthority15155.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15158.bound, LeftAuthority15155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority15158.actual selector witness) * (LeftAuthority15155.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15162

namespace LeftBound15166
def owner : Owner := ⟨.program ⟨214⟩, ⟨10514⟩⟩
def transferEvent : Nat := 15166
def frameStart : Nat := 15133
def rule : BoundRule := .identity (.predecessor 0 15165 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15165 .coefficient)
      LeftBound15162.bound (LeftBound15162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15162.derived selector witness)

def rawBound : CoeffClass := LeftBound15162.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound15162.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound15166

namespace LeftBound15183
def owner : Owner := ⟨.program ⟨214⟩, ⟨10592⟩⟩
def transferEvent : Nat := 15183
def frameStart : Nat := 15133
def rule : BoundRule := .sum [.predecessor 0 15181 .coefficient, .predecessor 1 15182 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15181 .coefficient)
      LeftBound15166.bound (LeftBound15166.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound15166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15182 .coefficient)
      LeftAuthority15179.bound (LeftAuthority15179.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority15179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15166.bound, LeftAuthority15179.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15166.bound, LeftAuthority15179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15166.actual selector witness, LeftAuthority15179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15183

namespace LeftBound15186
def owner : Owner := ⟨.program ⟨214⟩, ⟨10593⟩⟩
def transferEvent : Nat := 15186
def frameStart : Nat := 15133
def rule : BoundRule := .identity (.predecessor 0 15185 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15185 .coefficient)
      LeftBound15183.bound (LeftBound15183.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound15183.derived selector witness)

def rawBound : CoeffClass := LeftBound15183.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound15183.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound15186

namespace LeftBound15192
def owner : Owner := ⟨.program ⟨214⟩, ⟨10594⟩⟩
def transferEvent : Nat := 15192
def frameStart : Nat := 15133
def rule : BoundRule := .product (.predecessor 0 15190 .coefficient) (.predecessor 1 15191 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15190 .coefficient)
      LeftAuthority15188.bound (LeftAuthority15188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15191 .coefficient)
      LeftBound15186.bound (LeftBound15186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15186.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority15188.bound LeftBound15186.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15188.bound, LeftBound15186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority15188.actual selector witness) * (LeftBound15186.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15192

namespace LeftBound15208
def owner : Owner := ⟨.program ⟨214⟩, ⟨7832⟩⟩
def transferEvent : Nat := 15208
def frameStart : Nat := 15133
def rule : BoundRule := .scale (.predecessor 0 15206 .coefficient) (.value (.predecessor 1 15207 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15206 .coefficient)
      LeftAuthority15204.bound (LeftAuthority15204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15204.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15207 .coefficient)
      LeftAuthority15195.bound (LeftAuthority15195.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority15195.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority15204.bound LeftAuthority15195.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15204.bound, LeftAuthority15195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15204.actual selector witness) * (LeftAuthority15195.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound15208

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
