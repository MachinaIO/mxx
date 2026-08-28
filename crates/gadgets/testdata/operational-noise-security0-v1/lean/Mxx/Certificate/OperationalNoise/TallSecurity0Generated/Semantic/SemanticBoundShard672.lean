import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard671

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98117
def owner : Owner := ⟨.program ⟨214⟩, ⟨21680⟩⟩
def transferEvent : Nat := 98117
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩ [⟨.result 98109 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98109 .coefficient)
      LeftAuthority98108.bound (LeftAuthority98108.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21677⟩⟩) (rawTerms := some (Proof.Events383.exact98109RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98108.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98108.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98108.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98108.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98117

namespace LeftBound98118
def owner : Owner := ⟨.program ⟨214⟩, ⟨21680⟩⟩
def transferEvent : Nat := 98118
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 98117) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98117)
      LeftBound98117.bound (LeftBound98117.actual selector witness) := by
  exact .transfer (LeftBound98117.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound98117.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound98117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound98117.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98118

namespace LeftBound98189
def owner : Owner := ⟨.program ⟨214⟩, ⟨16169⟩⟩
def transferEvent : Nat := 98189
def frameStart : Nat := 98162
def rule : BoundRule := .identity (.predecessor 0 98188 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98188 .coefficient)
      LeftAuthority98186.bound (LeftAuthority98186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98186.derived selector witness)

def rawBound : CoeffClass := LeftAuthority98186.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority98186.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98189

namespace LeftBound98206
def owner : Owner := ⟨.program ⟨214⟩, ⟨16210⟩⟩
def transferEvent : Nat := 98206
def frameStart : Nat := 98162
def rule : BoundRule := .sum [.predecessor 0 98204 .coefficient, .predecessor 1 98205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98204 .coefficient)
      LeftBound98189.bound (LeftBound98189.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98205 .coefficient)
      LeftAuthority98202.bound (LeftAuthority98202.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98189.bound, LeftAuthority98202.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98189.bound, LeftAuthority98202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98189.actual selector witness, LeftAuthority98202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98206

namespace LeftBound98209
def owner : Owner := ⟨.program ⟨214⟩, ⟨16211⟩⟩
def transferEvent : Nat := 98209
def frameStart : Nat := 98162
def rule : BoundRule := .identity (.predecessor 0 98208 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98208 .coefficient)
      LeftBound98206.bound (LeftBound98206.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98206.derived selector witness)

def rawBound : CoeffClass := LeftBound98206.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound98206.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98209

namespace LeftBound98215
def owner : Owner := ⟨.program ⟨214⟩, ⟨16212⟩⟩
def transferEvent : Nat := 98215
def frameStart : Nat := 98162
def rule : BoundRule := .product (.predecessor 0 98213 .coefficient) (.predecessor 1 98214 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98213 .coefficient)
      LeftAuthority98211.bound (LeftAuthority98211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98214 .coefficient)
      LeftBound98209.bound (LeftBound98209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98209.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority98211.bound LeftBound98209.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98211.bound, LeftBound98209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority98211.actual selector witness) * (LeftBound98209.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98215

namespace LeftBound98223
def owner : Owner := ⟨.program ⟨214⟩, ⟨16213⟩⟩
def transferEvent : Nat := 98223
def frameStart : Nat := 98162
def rule : BoundRule := .sum [.predecessor 0 98221 .coefficient, .predecessor 1 98222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98221 .coefficient)
      LeftAuthority98219.bound (LeftAuthority98219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98222 .coefficient)
      LeftBound98215.bound (LeftBound98215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98215.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority98219.bound, LeftBound98215.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98219.bound, LeftBound98215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority98219.actual selector witness, LeftBound98215.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98223

namespace LeftBound98227
def owner : Owner := ⟨.program ⟨214⟩, ⟨28266⟩⟩
def transferEvent : Nat := 98227
def frameStart : Nat := 98162
def rule : BoundRule := .product (.predecessor 0 98225 .coefficient) (.predecessor 1 98226 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98225 .coefficient)
      LeftBound98223.bound (LeftBound98223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98226 .coefficient)
      LeftAuthority98200.bound (LeftAuthority98200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98223.bound LeftAuthority98200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98223.bound, LeftAuthority98200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98223.actual selector witness) * (LeftAuthority98200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98227

namespace LeftBound98238
def owner : Owner := ⟨.program ⟨214⟩, ⟨18314⟩⟩
def transferEvent : Nat := 98238
def frameStart : Nat := 98162
def rule : BoundRule := .product (.predecessor 0 98236 .coefficient) (.predecessor 1 98237 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98236 .coefficient)
      LeftAuthority98211.bound (LeftAuthority98211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98237 .coefficient)
      LeftAuthority98234.bound (LeftAuthority98234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98234.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98234.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority98211.bound LeftAuthority98234.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98211.bound, LeftAuthority98234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority98211.actual selector witness) * (LeftAuthority98234.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98238

namespace LeftBound98246
def owner : Owner := ⟨.program ⟨214⟩, ⟨18315⟩⟩
def transferEvent : Nat := 98246
def frameStart : Nat := 98162
def rule : BoundRule := .sum [.predecessor 0 98244 .coefficient, .predecessor 1 98245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98244 .coefficient)
      LeftAuthority98242.bound (LeftAuthority98242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98242.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98245 .coefficient)
      LeftBound98238.bound (LeftBound98238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98238.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority98242.bound, LeftBound98238.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98242.bound, LeftBound98238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority98242.actual selector witness, LeftBound98238.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98246

namespace LeftBound98250
def owner : Owner := ⟨.program ⟨214⟩, ⟨28270⟩⟩
def transferEvent : Nat := 98250
def frameStart : Nat := 98162
def rule : BoundRule := .sum [.predecessor 0 98248 .coefficient, .predecessor 1 98249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98248 .coefficient)
      LeftBound98246.bound (LeftBound98246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98249 .coefficient)
      LeftBound98227.bound (LeftBound98227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98227.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98246.bound, LeftBound98227.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98246.bound, LeftBound98227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98246.actual selector witness, LeftBound98227.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98250

namespace LeftBound98263
def owner : Owner := ⟨.program ⟨214⟩, ⟨28268⟩⟩
def transferEvent : Nat := 98263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98261 .coefficient, .predecessor 1 98262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98261 .coefficient)
      LeftBound98116.bound (LeftBound98116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98262 .coefficient)
      LeftBound98099.bound (LeftBound98099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98099.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98116.bound, LeftBound98099.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98116.bound, LeftBound98099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98116.actual selector witness, LeftBound98099.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98263

namespace LeftBound98266
def owner : Owner := ⟨.program ⟨214⟩, ⟨28268⟩⟩
def transferEvent : Nat := 98266
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 98260 .summary, .result 98106 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98260 .summary)
      LeftBound98118.bound (LeftBound98118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21680⟩⟩) (rawTerms := some (Proof.Events383.exact98260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98106 .summary)
      LeftBound98101.bound (LeftBound98101.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28267⟩⟩) (rawTerms := some (Proof.Events383.exact98106RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98101.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98118.bound, LeftBound98101.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98118.bound, LeftBound98101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98118.actual selector witness, LeftBound98101.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98266

namespace LeftBound98290
def owner : Owner := ⟨.program ⟨214⟩, ⟨11542⟩⟩
def transferEvent : Nat := 98290
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 98288 .coefficient) (.predecessor 1 98289 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98288 .coefficient)
      LeftAuthority4772.bound (LeftAuthority4772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98289 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4772.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4772.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4772.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98290

namespace LeftBound98295
def owner : Owner := ⟨.program ⟨214⟩, ⟨7117⟩⟩
def transferEvent : Nat := 98295
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98293 .coefficient) (.predecessor 1 98294 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98293 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98294 .coefficient)
      LeftBound10980.bound (LeftBound10980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound10980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound10980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound10980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98295

namespace LeftBound98300
def owner : Owner := ⟨.program ⟨214⟩, ⟨11543⟩⟩
def transferEvent : Nat := 98300
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98298 .coefficient, .predecessor 1 98299 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98298 .coefficient)
      LeftBound98295.bound (LeftBound98295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98299 .coefficient)
      LeftBound98290.bound (LeftBound98290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98290.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98295.bound, LeftBound98290.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98295.bound, LeftBound98290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98295.actual selector witness, LeftBound98290.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98300

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
