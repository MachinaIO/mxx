import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard034

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound7195
def owner : Owner := ⟨.program ⟨214⟩, ⟨6769⟩⟩
def transferEvent : Nat := 7195
def frameStart : Nat := 7117
def rule : BoundRule := .identity (.predecessor 0 7194 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7194 .coefficient)
      LeftAuthority7182.bound (LeftAuthority7182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7182.derived selector witness)

def rawBound : CoeffClass := LeftAuthority7182.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority7182.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7195

namespace LeftBound7199
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def transferEvent : Nat := 7199
def frameStart : Nat := 7117
def rule : BoundRule := .product (.predecessor 0 7197 .coefficient) (.predecessor 1 7198 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7197 .coefficient)
      LeftBound7195.bound (LeftBound7195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7195.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7195.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7198 .coefficient)
      LeftBound7192.bound (LeftBound7192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7192.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7195.bound LeftBound7192.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7195.bound, LeftBound7192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7195.actual selector witness) * (LeftBound7192.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7199

namespace LeftBound7204
def owner : Owner := ⟨.program ⟨214⟩, ⟨13269⟩⟩
def transferEvent : Nat := 7204
def frameStart : Nat := 7117
def rule : BoundRule := .sum [.predecessor 0 7202 .coefficient, .predecessor 1 7203 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7202 .coefficient)
      LeftBound7199.bound (LeftBound7199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7199.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7203 .coefficient)
      LeftBound7176.bound (LeftBound7176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7199.bound, LeftBound7176.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7199.bound, LeftBound7176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7199.actual selector witness, LeftBound7176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7204

namespace LeftBound7208
def owner : Owner := ⟨.program ⟨214⟩, ⟨25704⟩⟩
def transferEvent : Nat := 7208
def frameStart : Nat := 7117
def rule : BoundRule := .product (.predecessor 0 7206 .coefficient) (.predecessor 1 7207 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7206 .coefficient)
      LeftBound7204.bound (LeftBound7204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7207 .coefficient)
      LeftAuthority7161.bound (LeftAuthority7161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7161.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7204.bound LeftAuthority7161.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7204.bound, LeftAuthority7161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7204.actual selector witness) * (LeftAuthority7161.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7208

namespace LeftBound7219
def owner : Owner := ⟨.program ⟨214⟩, ⟨16889⟩⟩
def transferEvent : Nat := 7219
def frameStart : Nat := 7117
def rule : BoundRule := .product (.predecessor 0 7217 .coefficient) (.predecessor 1 7218 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7217 .coefficient)
      LeftAuthority7172.bound (LeftAuthority7172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7172.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7218 .coefficient)
      LeftAuthority7215.bound (LeftAuthority7215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7215.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7215.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority7172.bound LeftAuthority7215.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7172.bound, LeftAuthority7215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority7172.actual selector witness) * (LeftAuthority7215.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7219

namespace LeftBound7227
def owner : Owner := ⟨.program ⟨214⟩, ⟨16890⟩⟩
def transferEvent : Nat := 7227
def frameStart : Nat := 7117
def rule : BoundRule := .sum [.predecessor 0 7225 .coefficient, .predecessor 1 7226 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7225 .coefficient)
      LeftAuthority7223.bound (LeftAuthority7223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7226 .coefficient)
      LeftBound7219.bound (LeftBound7219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7219.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority7223.bound, LeftBound7219.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7223.bound, LeftBound7219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority7223.actual selector witness, LeftBound7219.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7227

namespace LeftBound7231
def owner : Owner := ⟨.program ⟨214⟩, ⟨25705⟩⟩
def transferEvent : Nat := 7231
def frameStart : Nat := 7117
def rule : BoundRule := .sum [.predecessor 0 7229 .coefficient, .predecessor 1 7230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7229 .coefficient)
      LeftBound7227.bound (LeftBound7227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7230 .coefficient)
      LeftBound7208.bound (LeftBound7208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7208.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7227.bound, LeftBound7208.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7227.bound, LeftBound7208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7227.actual selector witness, LeftBound7208.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7231

namespace LeftBound7244
def owner : Owner := ⟨.program ⟨214⟩, ⟨25703⟩⟩
def transferEvent : Nat := 7244
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 7242 .coefficient, .predecessor 1 7243 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7242 .coefficient)
      LeftBound7065.bound (LeftBound7065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7065.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7065.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7243 .coefficient)
      LeftBound7048.bound (LeftBound7048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7048.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7065.bound, LeftBound7048.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7065.bound, LeftBound7048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7065.actual selector witness, LeftBound7048.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7244

namespace LeftBound7247
def owner : Owner := ⟨.program ⟨214⟩, ⟨25703⟩⟩
def transferEvent : Nat := 7247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 7241 .summary, .result 7055 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7241 .summary)
      LeftBound7067.bound (LeftBound7067.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20195⟩⟩) (rawTerms := some (Proof.Events028.exact7241RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7055 .summary)
      LeftBound7050.bound (LeftBound7050.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25702⟩⟩) (rawTerms := some (Proof.Events027.exact7055RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7067.bound, LeftBound7050.bound]
def bound : CoeffClass := .finite ⟨352182857248768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7067.bound, LeftBound7050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7067.actual selector witness, LeftBound7050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7247

namespace LeftBound7251
def owner : Owner := ⟨.program ⟨214⟩, ⟨29873⟩⟩
def transferEvent : Nat := 7251
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7249 .coefficient) (.predecessor 1 7250 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7249 .coefficient)
      LeftBound7244.bound (LeftBound7244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7250 .coefficient)
      LeftAuthority6951.bound (LeftAuthority6951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6951.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6951.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7244.bound LeftAuthority6951.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7244.bound, LeftAuthority6951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7244.actual selector witness) * (LeftAuthority6951.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7251

namespace LeftBound7252
def owner : Owner := ⟨.program ⟨214⟩, ⟨29873⟩⟩
def transferEvent : Nat := 7252
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29871⟩⟩]⟩ [⟨.result 6952 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6952 .coefficient)
      LeftAuthority6951.bound (LeftAuthority6951.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29871⟩⟩) (rawTerms := some (Proof.Events027.exact6952RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6951.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6951.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6951.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6951.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound7252

namespace LeftBound7253
def owner : Owner := ⟨.program ⟨214⟩, ⟨29873⟩⟩
def transferEvent : Nat := 7253
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 7248 .summary) (.transfer 7252) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7248 .summary)
      LeftBound7247.bound (LeftBound7247.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25703⟩⟩) (rawTerms := some (Proof.Events028.exact7248RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7252)
      LeftBound7252.bound (LeftBound7252.actual selector witness) := by
  exact .transfer (LeftBound7252.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7247.bound LeftBound7252.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7247.bound, LeftBound7252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7247.actual selector witness) * (LeftBound7252.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7253

namespace LeftBound7264
def owner : Owner := ⟨.program ⟨214⟩, ⟨22714⟩⟩
def transferEvent : Nat := 7264
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 7262 .coefficient) (.value (.predecessor 1 7263 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7262 .coefficient)
      LeftAuthority7260.bound (LeftAuthority7260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7263 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority7260.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7260.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7260.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound7264

namespace LeftBound7268
def owner : Owner := ⟨.program ⟨214⟩, ⟨22715⟩⟩
def transferEvent : Nat := 7268
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7266 .coefficient) (.predecessor 1 7267 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7266 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7267 .coefficient)
      LeftBound7264.bound (LeftBound7264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7264.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound7264.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound7264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound7264.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7268

namespace LeftBound7269
def owner : Owner := ⟨.program ⟨214⟩, ⟨22715⟩⟩
def transferEvent : Nat := 7269
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22712⟩⟩]⟩ [⟨.result 7261 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7261 .coefficient)
      LeftAuthority7260.bound (LeftAuthority7260.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22712⟩⟩) (rawTerms := some (Proof.Events028.exact7261RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7260.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7260.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7260.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound7269

namespace LeftBound7270
def owner : Owner := ⟨.program ⟨214⟩, ⟨22715⟩⟩
def transferEvent : Nat := 7270
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 7269) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7269)
      LeftBound7269.bound (LeftBound7269.actual selector witness) := by
  exact .transfer (LeftBound7269.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound7269.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound7269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound7269.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7270

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
