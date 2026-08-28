import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard373

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55290
def owner : Owner := ⟨.program ⟨214⟩, ⟨28098⟩⟩
def transferEvent : Nat := 55290
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩ [⟨.result 55009 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55009 .coefficient)
      LeftAuthority55008.bound (LeftAuthority55008.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28096⟩⟩) (rawTerms := some (Proof.Events214.exact55009RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55008.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55008.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55008.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55290

namespace LeftBound55291
def owner : Owner := ⟨.program ⟨214⟩, ⟨28098⟩⟩
def transferEvent : Nat := 55291
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55286 .summary) (.transfer 55290) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55286 .summary)
      LeftBound55285.bound (LeftBound55285.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26150⟩⟩) (rawTerms := some (Proof.Events215.exact55286RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55290)
      LeftBound55290.bound (LeftBound55290.actual selector witness) := by
  exact .transfer (LeftBound55290.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55285.bound LeftBound55290.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55285.bound, LeftBound55290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55285.actual selector witness) * (LeftBound55290.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55291

namespace LeftBound55302
def owner : Owner := ⟨.program ⟨214⟩, ⟨21550⟩⟩
def transferEvent : Nat := 55302
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 55300 .coefficient) (.value (.predecessor 1 55301 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55300 .coefficient)
      LeftAuthority55298.bound (LeftAuthority55298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55298.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55298.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55301 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority55298.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55298.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55298.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55302

namespace LeftBound55306
def owner : Owner := ⟨.program ⟨214⟩, ⟨21551⟩⟩
def transferEvent : Nat := 55306
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55304 .coefficient) (.predecessor 1 55305 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55304 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55305 .coefficient)
      LeftBound55302.bound (LeftBound55302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55302.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound55302.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound55302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound55302.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55306

namespace LeftBound55307
def owner : Owner := ⟨.program ⟨214⟩, ⟨21551⟩⟩
def transferEvent : Nat := 55307
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩ [⟨.result 55299 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55299 .coefficient)
      LeftAuthority55298.bound (LeftAuthority55298.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21548⟩⟩) (rawTerms := some (Proof.Events216.exact55299RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55298.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55298.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55298.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55298.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55307

namespace LeftBound55308
def owner : Owner := ⟨.program ⟨214⟩, ⟨21551⟩⟩
def transferEvent : Nat := 55308
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 55307) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55307)
      LeftBound55307.bound (LeftBound55307.actual selector witness) := by
  exact .transfer (LeftBound55307.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound55307.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound55307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound55307.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55308

namespace LeftBound55403
def owner : Owner := ⟨.program ⟨214⟩, ⟨16064⟩⟩
def transferEvent : Nat := 55403
def frameStart : Nat := 55364
def rule : BoundRule := .identity (.predecessor 0 55402 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55402 .coefficient)
      LeftAuthority55400.bound (LeftAuthority55400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55400.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55400.derived selector witness)

def rawBound : CoeffClass := LeftAuthority55400.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority55400.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55403

namespace LeftBound55420
def owner : Owner := ⟨.program ⟨214⟩, ⟨16138⟩⟩
def transferEvent : Nat := 55420
def frameStart : Nat := 55364
def rule : BoundRule := .sum [.predecessor 0 55418 .coefficient, .predecessor 1 55419 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55418 .coefficient)
      LeftBound55403.bound (LeftBound55403.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55419 .coefficient)
      LeftAuthority55416.bound (LeftAuthority55416.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority55416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55403.bound, LeftAuthority55416.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55403.bound, LeftAuthority55416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55403.actual selector witness, LeftAuthority55416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55420

namespace LeftBound55423
def owner : Owner := ⟨.program ⟨214⟩, ⟨16139⟩⟩
def transferEvent : Nat := 55423
def frameStart : Nat := 55364
def rule : BoundRule := .identity (.predecessor 0 55422 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55422 .coefficient)
      LeftBound55420.bound (LeftBound55420.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55420.derived selector witness)

def rawBound : CoeffClass := LeftBound55420.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound55420.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55423

namespace LeftBound55429
def owner : Owner := ⟨.program ⟨214⟩, ⟨16140⟩⟩
def transferEvent : Nat := 55429
def frameStart : Nat := 55364
def rule : BoundRule := .product (.predecessor 0 55427 .coefficient) (.predecessor 1 55428 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55427 .coefficient)
      LeftAuthority55425.bound (LeftAuthority55425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55425.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55428 .coefficient)
      LeftBound55423.bound (LeftBound55423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55423.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority55425.bound LeftBound55423.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55425.bound, LeftBound55423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority55425.actual selector witness) * (LeftBound55423.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55429

namespace LeftBound55437
def owner : Owner := ⟨.program ⟨214⟩, ⟨16141⟩⟩
def transferEvent : Nat := 55437
def frameStart : Nat := 55364
def rule : BoundRule := .sum [.predecessor 0 55435 .coefficient, .predecessor 1 55436 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55435 .coefficient)
      LeftAuthority55433.bound (LeftAuthority55433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55433.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55433.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55436 .coefficient)
      LeftBound55429.bound (LeftBound55429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55429.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55433.bound, LeftBound55429.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55433.bound, LeftBound55429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority55433.actual selector witness, LeftBound55429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55437

namespace LeftBound55441
def owner : Owner := ⟨.program ⟨214⟩, ⟨28097⟩⟩
def transferEvent : Nat := 55441
def frameStart : Nat := 55364
def rule : BoundRule := .product (.predecessor 0 55439 .coefficient) (.predecessor 1 55440 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55439 .coefficient)
      LeftBound55437.bound (LeftBound55437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55437.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55440 .coefficient)
      LeftAuthority55414.bound (LeftAuthority55414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55437.bound LeftAuthority55414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55437.bound, LeftAuthority55414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55437.actual selector witness) * (LeftAuthority55414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55441

namespace LeftBound55452
def owner : Owner := ⟨.program ⟨214⟩, ⟨16109⟩⟩
def transferEvent : Nat := 55452
def frameStart : Nat := 55364
def rule : BoundRule := .product (.predecessor 0 55450 .coefficient) (.predecessor 1 55451 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55450 .coefficient)
      LeftAuthority55425.bound (LeftAuthority55425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55425.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55451 .coefficient)
      LeftAuthority55448.bound (LeftAuthority55448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55448.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55448.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority55425.bound LeftAuthority55448.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55425.bound, LeftAuthority55448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority55425.actual selector witness) * (LeftAuthority55448.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55452

namespace LeftBound55460
def owner : Owner := ⟨.program ⟨214⟩, ⟨16110⟩⟩
def transferEvent : Nat := 55460
def frameStart : Nat := 55364
def rule : BoundRule := .sum [.predecessor 0 55458 .coefficient, .predecessor 1 55459 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55458 .coefficient)
      LeftAuthority55456.bound (LeftAuthority55456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55459 .coefficient)
      LeftBound55452.bound (LeftBound55452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55456.bound, LeftBound55452.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55456.bound, LeftBound55452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority55456.actual selector witness, LeftBound55452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55460

namespace LeftBound55464
def owner : Owner := ⟨.program ⟨214⟩, ⟨28101⟩⟩
def transferEvent : Nat := 55464
def frameStart : Nat := 55364
def rule : BoundRule := .sum [.predecessor 0 55462 .coefficient, .predecessor 1 55463 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55462 .coefficient)
      LeftBound55460.bound (LeftBound55460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55463 .coefficient)
      LeftBound55441.bound (LeftBound55441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55441.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55460.bound, LeftBound55441.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55460.bound, LeftBound55441.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55460.actual selector witness, LeftBound55441.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55464

namespace LeftBound55477
def owner : Owner := ⟨.program ⟨214⟩, ⟨28099⟩⟩
def transferEvent : Nat := 55477
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55475 .coefficient, .predecessor 1 55476 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55475 .coefficient)
      LeftBound55306.bound (LeftBound55306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55476 .coefficient)
      LeftBound55289.bound (LeftBound55289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55289.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55306.bound, LeftBound55289.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55306.bound, LeftBound55289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55306.actual selector witness, LeftBound55289.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55477

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
