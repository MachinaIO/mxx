import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard580
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard624

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound92247
def owner : Owner := ⟨.program ⟨214⟩, ⟨16137⟩⟩
def transferEvent : Nat := 92247
def frameStart : Nat := 92174
def rule : BoundRule := .sum [.predecessor 0 92245 .coefficient, .predecessor 1 92246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92245 .coefficient)
      LeftAuthority92243.bound (LeftAuthority92243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92246 .coefficient)
      LeftBound92239.bound (LeftBound92239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92239.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92243.bound, LeftBound92239.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92243.bound, LeftBound92239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92243.actual selector witness, LeftBound92239.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92247

namespace LeftBound92251
def owner : Owner := ⟨.program ⟨214⟩, ⟨28077⟩⟩
def transferEvent : Nat := 92251
def frameStart : Nat := 92174
def rule : BoundRule := .product (.predecessor 0 92249 .coefficient) (.predecessor 1 92250 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92249 .coefficient)
      LeftBound92247.bound (LeftBound92247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92250 .coefficient)
      LeftAuthority92224.bound (LeftAuthority92224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92224.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92224.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92247.bound LeftAuthority92224.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92247.bound, LeftAuthority92224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92247.actual selector witness) * (LeftAuthority92224.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92251

namespace LeftBound92262
def owner : Owner := ⟨.program ⟨214⟩, ⟨18040⟩⟩
def transferEvent : Nat := 92262
def frameStart : Nat := 92174
def rule : BoundRule := .product (.predecessor 0 92260 .coefficient) (.predecessor 1 92261 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92260 .coefficient)
      LeftAuthority92235.bound (LeftAuthority92235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92261 .coefficient)
      LeftAuthority92258.bound (LeftAuthority92258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92258.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92258.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority92235.bound LeftAuthority92258.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92235.bound, LeftAuthority92258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority92235.actual selector witness) * (LeftAuthority92258.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92262

namespace LeftBound92270
def owner : Owner := ⟨.program ⟨214⟩, ⟨18041⟩⟩
def transferEvent : Nat := 92270
def frameStart : Nat := 92174
def rule : BoundRule := .sum [.predecessor 0 92268 .coefficient, .predecessor 1 92269 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92268 .coefficient)
      LeftAuthority92266.bound (LeftAuthority92266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92269 .coefficient)
      LeftBound92262.bound (LeftBound92262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92266.bound, LeftBound92262.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92266.bound, LeftBound92262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92266.actual selector witness, LeftBound92262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92270

namespace LeftBound92274
def owner : Owner := ⟨.program ⟨214⟩, ⟨28082⟩⟩
def transferEvent : Nat := 92274
def frameStart : Nat := 92174
def rule : BoundRule := .sum [.predecessor 0 92272 .coefficient, .predecessor 1 92273 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92272 .coefficient)
      LeftBound92270.bound (LeftBound92270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92273 .coefficient)
      LeftBound92251.bound (LeftBound92251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92251.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92270.bound, LeftBound92251.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92270.bound, LeftBound92251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92270.actual selector witness, LeftBound92251.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92274

namespace LeftBound92287
def owner : Owner := ⟨.program ⟨214⟩, ⟨28079⟩⟩
def transferEvent : Nat := 92287
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92285 .coefficient, .predecessor 1 92286 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92285 .coefficient)
      LeftBound92116.bound (LeftBound92116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92286 .coefficient)
      LeftBound92099.bound (LeftBound92099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92099.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92116.bound, LeftBound92099.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92116.bound, LeftBound92099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92116.actual selector witness, LeftBound92099.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92287

namespace LeftBound92290
def owner : Owner := ⟨.program ⟨214⟩, ⟨28079⟩⟩
def transferEvent : Nat := 92290
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 92284 .summary, .result 92106 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92284 .summary)
      LeftBound92118.bound (LeftBound92118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21475⟩⟩) (rawTerms := some (Proof.Events360.exact92284RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92106 .summary)
      LeftBound92101.bound (LeftBound92101.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28078⟩⟩) (rawTerms := some (Proof.Events359.exact92106RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92101.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92118.bound, LeftBound92101.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92118.bound, LeftBound92101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92118.actual selector witness, LeftBound92101.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92290

namespace LeftBound92294
def owner : Owner := ⟨.program ⟨214⟩, ⟨28080⟩⟩
def transferEvent : Nat := 92294
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92292 .coefficient) (.predecessor 1 92293 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92292 .coefficient)
      LeftBound92287.bound (LeftBound92287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92293 .coefficient)
      LeftBound5698.bound (LeftBound5698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92287.bound LeftBound5698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92287.bound, LeftBound5698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92287.actual selector witness) * (LeftBound5698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92294

namespace LeftBound92295
def owner : Owner := ⟨.program ⟨214⟩, ⟨28080⟩⟩
def transferEvent : Nat := 92295
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩ [⟨.result 5695 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5695 .coefficient)
      LeftAuthority5694.bound (LeftAuthority5694.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6637⟩⟩) (rawTerms := some (Proof.Events022.exact5695RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5694.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5694.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5694.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5694.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92295

namespace LeftBound92296
def owner : Owner := ⟨.program ⟨214⟩, ⟨28080⟩⟩
def transferEvent : Nat := 92296
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 92291 .summary) (.transfer 92295) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92291 .summary)
      LeftBound92290.bound (LeftBound92290.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28079⟩⟩) (rawTerms := some (Proof.Events360.exact92291RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92295)
      LeftBound92295.bound (LeftBound92295.actual selector witness) := by
  exact .transfer (LeftBound92295.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92290.bound LeftBound92295.bound
def bound : CoeffClass := .finite ⟨4742076480517514208552681472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92290.bound, LeftBound92295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92290.actual selector witness) * (LeftBound92295.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92296

namespace LeftBound92311
def owner : Owner := ⟨.program ⟨214⟩, ⟨27861⟩⟩
def transferEvent : Nat := 92311
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92309 .coefficient) (.predecessor 1 92310 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92309 .coefficient)
      LeftBound84992.bound (LeftBound84992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact84996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92310 .coefficient)
      LeftAuthority92307.bound (LeftAuthority92307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92307.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84992.bound LeftAuthority92307.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84992.bound, LeftAuthority92307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84992.actual selector witness) * (LeftAuthority92307.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92311

namespace LeftBound92312
def owner : Owner := ⟨.program ⟨214⟩, ⟨27861⟩⟩
def transferEvent : Nat := 92312
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27859⟩⟩]⟩ [⟨.result 92308 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92308 .coefficient)
      LeftAuthority92307.bound (LeftAuthority92307.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27859⟩⟩) (rawTerms := some (Proof.Events360.exact92308RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92307.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92307.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92307.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92312

namespace LeftBound92313
def owner : Owner := ⟨.program ⟨214⟩, ⟨27861⟩⟩
def transferEvent : Nat := 92313
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84996 .summary) (.transfer 92312) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84996 .summary)
      LeftBound84995.bound (LeftBound84995.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26068⟩⟩) (rawTerms := some (Proof.Events332.exact84996RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92312)
      LeftBound92312.bound (LeftBound92312.actual selector witness) := by
  exact .transfer (LeftBound92312.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84995.bound LeftBound92312.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84995.bound, LeftBound92312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84995.actual selector witness) * (LeftBound92312.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92313

namespace LeftBound92324
def owner : Owner := ⟨.program ⟨214⟩, ⟨21330⟩⟩
def transferEvent : Nat := 92324
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 92322 .coefficient) (.value (.predecessor 1 92323 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92322 .coefficient)
      LeftAuthority92320.bound (LeftAuthority92320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92323 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority92320.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92320.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92320.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound92324

namespace LeftBound92328
def owner : Owner := ⟨.program ⟨214⟩, ⟨21331⟩⟩
def transferEvent : Nat := 92328
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92326 .coefficient) (.predecessor 1 92327 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92326 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92327 .coefficient)
      LeftBound92324.bound (LeftBound92324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound92324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound92324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound92324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92328

namespace LeftBound92329
def owner : Owner := ⟨.program ⟨214⟩, ⟨21331⟩⟩
def transferEvent : Nat := 92329
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21328⟩⟩]⟩ [⟨.result 92321 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92321 .coefficient)
      LeftAuthority92320.bound (LeftAuthority92320.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21328⟩⟩) (rawTerms := some (Proof.Events360.exact92321RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92320.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92320.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92320.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92329

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
