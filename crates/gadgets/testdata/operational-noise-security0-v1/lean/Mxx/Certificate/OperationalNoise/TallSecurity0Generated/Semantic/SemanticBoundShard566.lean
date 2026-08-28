import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard565

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound83080
def owner : Owner := ⟨.program ⟨214⟩, ⟨28736⟩⟩
def transferEvent : Nat := 83080
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩ [⟨.result 82801 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82801 .coefficient)
      LeftAuthority82800.bound (LeftAuthority82800.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28734⟩⟩) (rawTerms := some (Proof.Events323.exact82801RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82800.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority82800.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82800.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83080

namespace LeftBound83081
def owner : Owner := ⟨.program ⟨214⟩, ⟨28736⟩⟩
def transferEvent : Nat := 83081
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83076 .summary) (.transfer 83080) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83076 .summary)
      LeftBound83075.bound (LeftBound83075.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25221⟩⟩) (rawTerms := some (Proof.Events324.exact83076RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83080)
      LeftBound83080.bound (LeftBound83080.actual selector witness) := by
  exact .transfer (LeftBound83080.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83075.bound LeftBound83080.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83075.bound, LeftBound83080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83075.actual selector witness) * (LeftBound83080.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83081

namespace LeftBound83092
def owner : Owner := ⟨.program ⟨214⟩, ⟨21978⟩⟩
def transferEvent : Nat := 83092
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 83090 .coefficient) (.value (.predecessor 1 83091 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83090 .coefficient)
      LeftAuthority83088.bound (LeftAuthority83088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83088.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83088.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83091 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority83088.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83088.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83088.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83092

namespace LeftBound83096
def owner : Owner := ⟨.program ⟨214⟩, ⟨21979⟩⟩
def transferEvent : Nat := 83096
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83094 .coefficient) (.predecessor 1 83095 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83094 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83095 .coefficient)
      LeftBound83092.bound (LeftBound83092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83092.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound83092.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound83092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound83092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83096

namespace LeftBound83097
def owner : Owner := ⟨.program ⟨214⟩, ⟨21979⟩⟩
def transferEvent : Nat := 83097
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩ [⟨.result 83089 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83089 .coefficient)
      LeftAuthority83088.bound (LeftAuthority83088.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21976⟩⟩) (rawTerms := some (Proof.Events324.exact83089RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83088.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83088.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority83088.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83088.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83097

namespace LeftBound83098
def owner : Owner := ⟨.program ⟨214⟩, ⟨21979⟩⟩
def transferEvent : Nat := 83098
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 83097) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83097)
      LeftBound83097.bound (LeftBound83097.actual selector witness) := by
  exact .transfer (LeftBound83097.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound83097.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound83097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound83097.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83098

namespace LeftBound83193
def owner : Owner := ⟨.program ⟨214⟩, ⟨16382⟩⟩
def transferEvent : Nat := 83193
def frameStart : Nat := 83154
def rule : BoundRule := .identity (.predecessor 0 83192 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83192 .coefficient)
      LeftAuthority83190.bound (LeftAuthority83190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83190.derived selector witness)

def rawBound : CoeffClass := LeftAuthority83190.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority83190.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83193

namespace LeftBound83210
def owner : Owner := ⟨.program ⟨214⟩, ⟨16421⟩⟩
def transferEvent : Nat := 83210
def frameStart : Nat := 83154
def rule : BoundRule := .sum [.predecessor 0 83208 .coefficient, .predecessor 1 83209 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83208 .coefficient)
      LeftBound83193.bound (LeftBound83193.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83209 .coefficient)
      LeftAuthority83206.bound (LeftAuthority83206.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority83206.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83193.bound, LeftAuthority83206.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83193.bound, LeftAuthority83206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83193.actual selector witness, LeftAuthority83206.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83210

namespace LeftBound83213
def owner : Owner := ⟨.program ⟨214⟩, ⟨16422⟩⟩
def transferEvent : Nat := 83213
def frameStart : Nat := 83154
def rule : BoundRule := .identity (.predecessor 0 83212 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83212 .coefficient)
      LeftBound83210.bound (LeftBound83210.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83210.derived selector witness)

def rawBound : CoeffClass := LeftBound83210.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound83210.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83213

namespace LeftBound83219
def owner : Owner := ⟨.program ⟨214⟩, ⟨16423⟩⟩
def transferEvent : Nat := 83219
def frameStart : Nat := 83154
def rule : BoundRule := .product (.predecessor 0 83217 .coefficient) (.predecessor 1 83218 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83217 .coefficient)
      LeftAuthority83215.bound (LeftAuthority83215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83215.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83218 .coefficient)
      LeftBound83213.bound (LeftBound83213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83213.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority83215.bound LeftBound83213.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83215.bound, LeftBound83213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority83215.actual selector witness) * (LeftBound83213.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83219

namespace LeftBound83227
def owner : Owner := ⟨.program ⟨214⟩, ⟨16424⟩⟩
def transferEvent : Nat := 83227
def frameStart : Nat := 83154
def rule : BoundRule := .sum [.predecessor 0 83225 .coefficient, .predecessor 1 83226 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83225 .coefficient)
      LeftAuthority83223.bound (LeftAuthority83223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83226 .coefficient)
      LeftBound83219.bound (LeftBound83219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83219.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83223.bound, LeftBound83219.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83223.bound, LeftBound83219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority83223.actual selector witness, LeftBound83219.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83227

namespace LeftBound83231
def owner : Owner := ⟨.program ⟨214⟩, ⟨28735⟩⟩
def transferEvent : Nat := 83231
def frameStart : Nat := 83154
def rule : BoundRule := .product (.predecessor 0 83229 .coefficient) (.predecessor 1 83230 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83229 .coefficient)
      LeftBound83227.bound (LeftBound83227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83230 .coefficient)
      LeftAuthority83204.bound (LeftAuthority83204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83204.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83204.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83227.bound LeftAuthority83204.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83227.bound, LeftAuthority83204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83227.actual selector witness) * (LeftAuthority83204.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83231

namespace LeftBound83242
def owner : Owner := ⟨.program ⟨214⟩, ⟨17121⟩⟩
def transferEvent : Nat := 83242
def frameStart : Nat := 83154
def rule : BoundRule := .product (.predecessor 0 83240 .coefficient) (.predecessor 1 83241 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83240 .coefficient)
      LeftAuthority83215.bound (LeftAuthority83215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83215.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83241 .coefficient)
      LeftAuthority83238.bound (LeftAuthority83238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83238.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83238.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83215.bound LeftAuthority83238.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83215.bound, LeftAuthority83238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority83215.actual selector witness) * (LeftAuthority83238.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83242

namespace LeftBound83250
def owner : Owner := ⟨.program ⟨214⟩, ⟨17122⟩⟩
def transferEvent : Nat := 83250
def frameStart : Nat := 83154
def rule : BoundRule := .sum [.predecessor 0 83248 .coefficient, .predecessor 1 83249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83248 .coefficient)
      LeftAuthority83246.bound (LeftAuthority83246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83249 .coefficient)
      LeftBound83242.bound (LeftBound83242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83242.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83246.bound, LeftBound83242.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83246.bound, LeftBound83242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority83246.actual selector witness, LeftBound83242.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83250

namespace LeftBound83254
def owner : Owner := ⟨.program ⟨214⟩, ⟨28739⟩⟩
def transferEvent : Nat := 83254
def frameStart : Nat := 83154
def rule : BoundRule := .sum [.predecessor 0 83252 .coefficient, .predecessor 1 83253 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83252 .coefficient)
      LeftBound83250.bound (LeftBound83250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83250.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83250.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83253 .coefficient)
      LeftBound83231.bound (LeftBound83231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83231.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83250.bound, LeftBound83231.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83250.bound, LeftBound83231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83250.actual selector witness, LeftBound83231.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83254

namespace LeftBound83267
def owner : Owner := ⟨.program ⟨214⟩, ⟨28737⟩⟩
def transferEvent : Nat := 83267
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83265 .coefficient, .predecessor 1 83266 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83265 .coefficient)
      LeftBound83096.bound (LeftBound83096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83266 .coefficient)
      LeftBound83079.bound (LeftBound83079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83096.bound, LeftBound83079.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83096.bound, LeftBound83079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83096.actual selector witness, LeftBound83079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83267

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
