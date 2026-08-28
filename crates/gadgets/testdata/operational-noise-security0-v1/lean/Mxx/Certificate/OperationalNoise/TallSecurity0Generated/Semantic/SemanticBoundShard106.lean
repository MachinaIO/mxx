import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard035
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard105

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound17251
def owner : Owner := ⟨.program ⟨214⟩, ⟨30199⟩⟩
def transferEvent : Nat := 17251
def frameStart : Nat := 17174
def rule : BoundRule := .product (.predecessor 0 17249 .coefficient) (.predecessor 1 17250 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17249 .coefficient)
      LeftBound17247.bound (LeftBound17247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17250 .coefficient)
      LeftAuthority17224.bound (LeftAuthority17224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17224.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17224.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17247.bound LeftAuthority17224.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17247.bound, LeftAuthority17224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17247.actual selector witness) * (LeftAuthority17224.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17251

namespace LeftBound17262
def owner : Owner := ⟨.program ⟨214⟩, ⟨18142⟩⟩
def transferEvent : Nat := 17262
def frameStart : Nat := 17174
def rule : BoundRule := .product (.predecessor 0 17260 .coefficient) (.predecessor 1 17261 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17260 .coefficient)
      LeftAuthority17235.bound (LeftAuthority17235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17261 .coefficient)
      LeftAuthority17258.bound (LeftAuthority17258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17258.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17258.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority17235.bound LeftAuthority17258.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17235.bound, LeftAuthority17258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority17235.actual selector witness) * (LeftAuthority17258.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17262

namespace LeftBound17270
def owner : Owner := ⟨.program ⟨214⟩, ⟨18143⟩⟩
def transferEvent : Nat := 17270
def frameStart : Nat := 17174
def rule : BoundRule := .sum [.predecessor 0 17268 .coefficient, .predecessor 1 17269 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17268 .coefficient)
      LeftAuthority17266.bound (LeftAuthority17266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17269 .coefficient)
      LeftBound17262.bound (LeftBound17262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17266.bound, LeftBound17262.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17266.bound, LeftBound17262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority17266.actual selector witness, LeftBound17262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17270

namespace LeftBound17274
def owner : Owner := ⟨.program ⟨214⟩, ⟨30204⟩⟩
def transferEvent : Nat := 17274
def frameStart : Nat := 17174
def rule : BoundRule := .sum [.predecessor 0 17272 .coefficient, .predecessor 1 17273 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17272 .coefficient)
      LeftBound17270.bound (LeftBound17270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17273 .coefficient)
      LeftBound17251.bound (LeftBound17251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17251.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17270.bound, LeftBound17251.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17270.bound, LeftBound17251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17270.actual selector witness, LeftBound17251.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17274

namespace LeftBound17287
def owner : Owner := ⟨.program ⟨214⟩, ⟨30201⟩⟩
def transferEvent : Nat := 17287
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 17285 .coefficient, .predecessor 1 17286 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17285 .coefficient)
      LeftBound17116.bound (LeftBound17116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17286 .coefficient)
      LeftBound17099.bound (LeftBound17099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17099.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17116.bound, LeftBound17099.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17116.bound, LeftBound17099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17116.actual selector witness, LeftBound17099.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17287

namespace LeftBound17290
def owner : Owner := ⟨.program ⟨214⟩, ⟨30201⟩⟩
def transferEvent : Nat := 17290
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 17284 .summary, .result 17106 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17284 .summary)
      LeftBound17118.bound (LeftBound17118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22787⟩⟩) (rawTerms := some (Proof.Events067.exact17284RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17106 .summary)
      LeftBound17101.bound (LeftBound17101.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30200⟩⟩) (rawTerms := some (Proof.Events066.exact17106RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17101.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17118.bound, LeftBound17101.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17118.bound, LeftBound17101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17118.actual selector witness, LeftBound17101.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17290

namespace LeftBound17294
def owner : Owner := ⟨.program ⟨214⟩, ⟨30202⟩⟩
def transferEvent : Nat := 17294
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17292 .coefficient) (.predecessor 1 17293 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17292 .coefficient)
      LeftBound17287.bound (LeftBound17287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17293 .coefficient)
      LeftBound5518.bound (LeftBound5518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17287.bound LeftBound5518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17287.bound, LeftBound5518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17287.actual selector witness) * (LeftBound5518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17294

namespace LeftBound17295
def owner : Owner := ⟨.program ⟨214⟩, ⟨30202⟩⟩
def transferEvent : Nat := 17295
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩ [⟨.result 5515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5515 .coefficient)
      LeftAuthority5514.bound (LeftAuthority5514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6657⟩⟩) (rawTerms := some (Proof.Events021.exact5515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5514.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17295

namespace LeftBound17296
def owner : Owner := ⟨.program ⟨214⟩, ⟨30202⟩⟩
def transferEvent : Nat := 17296
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17291 .summary) (.transfer 17295) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17291 .summary)
      LeftBound17290.bound (LeftBound17290.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30201⟩⟩) (rawTerms := some (Proof.Events067.exact17291RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17295)
      LeftBound17295.bound (LeftBound17295.actual selector witness) := by
  exact .transfer (LeftBound17295.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17290.bound LeftBound17295.bound
def bound : CoeffClass := .finite ⟨4743639307122182955475140608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17290.bound, LeftBound17295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17290.actual selector witness) * (LeftBound17295.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17296

namespace LeftBound17311
def owner : Owner := ⟨.program ⟨214⟩, ⟨29866⟩⟩
def transferEvent : Nat := 17311
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17309 .coefficient) (.predecessor 1 17310 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17309 .coefficient)
      LeftBound7244.bound (LeftBound7244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17310 .coefficient)
      LeftAuthority17307.bound (LeftAuthority17307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17307.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7244.bound LeftAuthority17307.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7244.bound, LeftAuthority17307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7244.actual selector witness) * (LeftAuthority17307.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17311

namespace LeftBound17312
def owner : Owner := ⟨.program ⟨214⟩, ⟨29866⟩⟩
def transferEvent : Nat := 17312
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩ [⟨.result 17308 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17308 .coefficient)
      LeftAuthority17307.bound (LeftAuthority17307.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29864⟩⟩) (rawTerms := some (Proof.Events067.exact17308RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17307.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17307.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17307.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17312

namespace LeftBound17313
def owner : Owner := ⟨.program ⟨214⟩, ⟨29866⟩⟩
def transferEvent : Nat := 17313
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 7248 .summary) (.transfer 17312) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7248 .summary)
      LeftBound7247.bound (LeftBound7247.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25703⟩⟩) (rawTerms := some (Proof.Events028.exact7248RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17312)
      LeftBound17312.bound (LeftBound17312.actual selector witness) := by
  exact .transfer (LeftBound17312.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7247.bound LeftBound17312.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7247.bound, LeftBound17312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7247.actual selector witness) * (LeftBound17312.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17313

namespace LeftBound17324
def owner : Owner := ⟨.program ⟨214⟩, ⟨22642⟩⟩
def transferEvent : Nat := 17324
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 17322 .coefficient) (.value (.predecessor 1 17323 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17322 .coefficient)
      LeftAuthority17320.bound (LeftAuthority17320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17323 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority17320.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17320.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17320.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound17324

namespace LeftBound17328
def owner : Owner := ⟨.program ⟨214⟩, ⟨22643⟩⟩
def transferEvent : Nat := 17328
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17326 .coefficient) (.predecessor 1 17327 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17326 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17327 .coefficient)
      LeftBound17324.bound (LeftBound17324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound17324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound17324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound17324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17328

namespace LeftBound17329
def owner : Owner := ⟨.program ⟨214⟩, ⟨22643⟩⟩
def transferEvent : Nat := 17329
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22640⟩⟩]⟩ [⟨.result 17321 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17321 .coefficient)
      LeftAuthority17320.bound (LeftAuthority17320.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22640⟩⟩) (rawTerms := some (Proof.Events067.exact17321RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17320.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17320.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17320.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17329

namespace LeftBound17330
def owner : Owner := ⟨.program ⟨214⟩, ⟨22643⟩⟩
def transferEvent : Nat := 17330
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 17329) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17329)
      LeftBound17329.bound (LeftBound17329.actual selector witness) := by
  exact .transfer (LeftBound17329.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound17329.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound17329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound17329.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17330

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
