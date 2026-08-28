import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard348
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard411

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound61380
def owner : Owner := ⟨.program ⟨214⟩, ⟨29828⟩⟩
def transferEvent : Nat := 61380
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 61374 .summary, .result 61196 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61374 .summary)
      LeftBound61208.bound (LeftBound61208.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22631⟩⟩) (rawTerms := some (Proof.Events239.exact61374RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61196 .summary)
      LeftBound61191.bound (LeftBound61191.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29827⟩⟩) (rawTerms := some (Proof.Events239.exact61196RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61208.bound, LeftBound61191.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61208.bound, LeftBound61191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61208.actual selector witness, LeftBound61191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61380

namespace LeftBound61384
def owner : Owner := ⟨.program ⟨214⟩, ⟨29829⟩⟩
def transferEvent : Nat := 61384
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61382 .coefficient) (.predecessor 1 61383 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61382 .coefficient)
      LeftBound61377.bound (LeftBound61377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61377.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61383 .coefficient)
      LeftBound5538.bound (LeftBound5538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61377.bound LeftBound5538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61377.bound, LeftBound5538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61377.actual selector witness) * (LeftBound5538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61384

namespace LeftBound61385
def owner : Owner := ⟨.program ⟨214⟩, ⟨29829⟩⟩
def transferEvent : Nat := 61385
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩ [⟨.result 5535 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5535 .coefficient)
      LeftAuthority5534.bound (LeftAuthority5534.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6659⟩⟩) (rawTerms := some (Proof.Events021.exact5535RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5534.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5534.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5534.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5534.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61385

namespace LeftBound61386
def owner : Owner := ⟨.program ⟨214⟩, ⟨29829⟩⟩
def transferEvent : Nat := 61386
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 61381 .summary) (.transfer 61385) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61381 .summary)
      LeftBound61380.bound (LeftBound61380.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29828⟩⟩) (rawTerms := some (Proof.Events239.exact61381RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61385)
      LeftBound61385.bound (LeftBound61385.actual selector witness) := by
  exact .transfer (LeftBound61385.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61380.bound LeftBound61385.bound
def bound : CoeffClass := .finite ⟨4743557053090358284584484864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61380.bound, LeftBound61385.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61380.actual selector witness) * (LeftBound61385.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61386

namespace LeftBound61401
def owner : Owner := ⟨.program ⟨214⟩, ⟨29610⟩⟩
def transferEvent : Nat := 61401
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61399 .coefficient) (.predecessor 1 61400 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61399 .coefficient)
      LeftBound51908.bound (LeftBound51908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61400 .coefficient)
      LeftAuthority61397.bound (LeftAuthority61397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51908.bound LeftAuthority61397.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51908.bound, LeftAuthority61397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51908.actual selector witness) * (LeftAuthority61397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61401

namespace LeftBound61402
def owner : Owner := ⟨.program ⟨214⟩, ⟨29610⟩⟩
def transferEvent : Nat := 61402
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29608⟩⟩]⟩ [⟨.result 61398 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61398 .coefficient)
      LeftAuthority61397.bound (LeftAuthority61397.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29608⟩⟩) (rawTerms := some (Proof.Events239.exact61398RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61397.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority61397.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61397.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61402

namespace LeftBound61403
def owner : Owner := ⟨.program ⟨214⟩, ⟨29610⟩⟩
def transferEvent : Nat := 61403
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51912 .summary) (.transfer 61402) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51912 .summary)
      LeftBound51911.bound (LeftBound51911.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25611⟩⟩) (rawTerms := some (Proof.Events202.exact51912RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61402)
      LeftBound61402.bound (LeftBound61402.actual selector witness) := by
  exact .transfer (LeftBound61402.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51911.bound LeftBound61402.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51911.bound, LeftBound61402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51911.actual selector witness) * (LeftBound61402.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61403

namespace LeftBound61414
def owner : Owner := ⟨.program ⟨214⟩, ⟨22486⟩⟩
def transferEvent : Nat := 61414
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 61412 .coefficient) (.value (.predecessor 1 61413 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61412 .coefficient)
      LeftAuthority61410.bound (LeftAuthority61410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61413 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority61410.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61410.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61410.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound61414

namespace LeftBound61418
def owner : Owner := ⟨.program ⟨214⟩, ⟨22487⟩⟩
def transferEvent : Nat := 61418
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61416 .coefficient) (.predecessor 1 61417 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61416 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61417 .coefficient)
      LeftBound61414.bound (LeftBound61414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound61414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound61414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound61414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61418

namespace LeftBound61419
def owner : Owner := ⟨.program ⟨214⟩, ⟨22487⟩⟩
def transferEvent : Nat := 61419
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22484⟩⟩]⟩ [⟨.result 61411 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61411 .coefficient)
      LeftAuthority61410.bound (LeftAuthority61410.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22484⟩⟩) (rawTerms := some (Proof.Events239.exact61411RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61410.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority61410.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61410.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61419

namespace LeftBound61420
def owner : Owner := ⟨.program ⟨214⟩, ⟨22487⟩⟩
def transferEvent : Nat := 61420
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 61419) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61419)
      LeftBound61419.bound (LeftBound61419.actual selector witness) := by
  exact .transfer (LeftBound61419.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound61419.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound61419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound61419.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61420

namespace LeftBound61515
def owner : Owner := ⟨.program ⟨214⟩, ⟨16757⟩⟩
def transferEvent : Nat := 61515
def frameStart : Nat := 61476
def rule : BoundRule := .identity (.predecessor 0 61514 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61514 .coefficient)
      LeftAuthority61512.bound (LeftAuthority61512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61512.derived selector witness)

def rawBound : CoeffClass := LeftAuthority61512.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority61512.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61515

namespace LeftBound61532
def owner : Owner := ⟨.program ⟨214⟩, ⟨16831⟩⟩
def transferEvent : Nat := 61532
def frameStart : Nat := 61476
def rule : BoundRule := .sum [.predecessor 0 61530 .coefficient, .predecessor 1 61531 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61530 .coefficient)
      LeftBound61515.bound (LeftBound61515.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61531 .coefficient)
      LeftAuthority61528.bound (LeftAuthority61528.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority61528.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61515.bound, LeftAuthority61528.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61515.bound, LeftAuthority61528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61515.actual selector witness, LeftAuthority61528.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61532

namespace LeftBound61535
def owner : Owner := ⟨.program ⟨214⟩, ⟨16832⟩⟩
def transferEvent : Nat := 61535
def frameStart : Nat := 61476
def rule : BoundRule := .identity (.predecessor 0 61534 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61534 .coefficient)
      LeftBound61532.bound (LeftBound61532.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61532.derived selector witness)

def rawBound : CoeffClass := LeftBound61532.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound61532.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61535

namespace LeftBound61541
def owner : Owner := ⟨.program ⟨214⟩, ⟨16833⟩⟩
def transferEvent : Nat := 61541
def frameStart : Nat := 61476
def rule : BoundRule := .product (.predecessor 0 61539 .coefficient) (.predecessor 1 61540 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61539 .coefficient)
      LeftAuthority61537.bound (LeftAuthority61537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61540 .coefficient)
      LeftBound61535.bound (LeftBound61535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61535.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61535.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority61537.bound LeftBound61535.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61537.bound, LeftBound61535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority61537.actual selector witness) * (LeftBound61535.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61541

namespace LeftBound61549
def owner : Owner := ⟨.program ⟨214⟩, ⟨16834⟩⟩
def transferEvent : Nat := 61549
def frameStart : Nat := 61476
def rule : BoundRule := .sum [.predecessor 0 61547 .coefficient, .predecessor 1 61548 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61547 .coefficient)
      LeftAuthority61545.bound (LeftAuthority61545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61545.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61548 .coefficient)
      LeftBound61541.bound (LeftBound61541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61545.bound, LeftBound61541.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61545.bound, LeftBound61541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61545.actual selector witness, LeftBound61541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61549

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
