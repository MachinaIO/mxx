import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard594
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard630

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound93161
def owner : Owner := ⟨.program ⟨214⟩, ⟨26993⟩⟩
def transferEvent : Nat := 93161
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 86916 .summary) (.transfer 93160) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86916 .summary)
      LeftBound86915.bound (LeftBound86915.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25298⟩⟩) (rawTerms := some (Proof.Events339.exact86916RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93160)
      LeftBound93160.bound (LeftBound93160.actual selector witness) := by
  exact .transfer (LeftBound93160.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86915.bound LeftBound93160.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86915.bound, LeftBound93160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86915.actual selector witness) * (LeftBound93160.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93161

namespace LeftBound93172
def owner : Owner := ⟨.program ⟨214⟩, ⟨20754⟩⟩
def transferEvent : Nat := 93172
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 93170 .coefficient) (.value (.predecessor 1 93171 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93170 .coefficient)
      LeftAuthority93168.bound (LeftAuthority93168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93171 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority93168.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93168.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93168.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound93172

namespace LeftBound93176
def owner : Owner := ⟨.program ⟨214⟩, ⟨20755⟩⟩
def transferEvent : Nat := 93176
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93174 .coefficient) (.predecessor 1 93175 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93174 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93175 .coefficient)
      LeftBound93172.bound (LeftBound93172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93172.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound93172.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound93172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound93172.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93176

namespace LeftBound93177
def owner : Owner := ⟨.program ⟨214⟩, ⟨20755⟩⟩
def transferEvent : Nat := 93177
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20752⟩⟩]⟩ [⟨.result 93169 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93169 .coefficient)
      LeftAuthority93168.bound (LeftAuthority93168.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20752⟩⟩) (rawTerms := some (Proof.Events363.exact93169RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93168.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority93168.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93168.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93177

namespace LeftBound93178
def owner : Owner := ⟨.program ⟨214⟩, ⟨20755⟩⟩
def transferEvent : Nat := 93178
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 93177) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93177)
      LeftBound93177.bound (LeftBound93177.actual selector witness) := by
  exact .transfer (LeftBound93177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound93177.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound93177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound93177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93178

namespace LeftBound93273
def owner : Owner := ⟨.program ⟨214⟩, ⟨15423⟩⟩
def transferEvent : Nat := 93273
def frameStart : Nat := 93234
def rule : BoundRule := .identity (.predecessor 0 93272 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93272 .coefficient)
      LeftAuthority93270.bound (LeftAuthority93270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93270.derived selector witness)

def rawBound : CoeffClass := LeftAuthority93270.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority93270.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93273

namespace LeftBound93290
def owner : Owner := ⟨.program ⟨214⟩, ⟨15462⟩⟩
def transferEvent : Nat := 93290
def frameStart : Nat := 93234
def rule : BoundRule := .sum [.predecessor 0 93288 .coefficient, .predecessor 1 93289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93288 .coefficient)
      LeftBound93273.bound (LeftBound93273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93289 .coefficient)
      LeftAuthority93286.bound (LeftAuthority93286.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority93286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93273.bound, LeftAuthority93286.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93273.bound, LeftAuthority93286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93273.actual selector witness, LeftAuthority93286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93290

namespace LeftBound93293
def owner : Owner := ⟨.program ⟨214⟩, ⟨15463⟩⟩
def transferEvent : Nat := 93293
def frameStart : Nat := 93234
def rule : BoundRule := .identity (.predecessor 0 93292 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93292 .coefficient)
      LeftBound93290.bound (LeftBound93290.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93290.derived selector witness)

def rawBound : CoeffClass := LeftBound93290.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound93290.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93293

namespace LeftBound93299
def owner : Owner := ⟨.program ⟨214⟩, ⟨15464⟩⟩
def transferEvent : Nat := 93299
def frameStart : Nat := 93234
def rule : BoundRule := .product (.predecessor 0 93297 .coefficient) (.predecessor 1 93298 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93297 .coefficient)
      LeftAuthority93295.bound (LeftAuthority93295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93298 .coefficient)
      LeftBound93293.bound (LeftBound93293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority93295.bound LeftBound93293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93295.bound, LeftBound93293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority93295.actual selector witness) * (LeftBound93293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93299

namespace LeftBound93307
def owner : Owner := ⟨.program ⟨214⟩, ⟨15465⟩⟩
def transferEvent : Nat := 93307
def frameStart : Nat := 93234
def rule : BoundRule := .sum [.predecessor 0 93305 .coefficient, .predecessor 1 93306 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93305 .coefficient)
      LeftAuthority93303.bound (LeftAuthority93303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93303.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93306 .coefficient)
      LeftBound93299.bound (LeftBound93299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93303.bound, LeftBound93299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93303.bound, LeftBound93299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93303.actual selector witness, LeftBound93299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93307

namespace LeftBound93311
def owner : Owner := ⟨.program ⟨214⟩, ⟨26992⟩⟩
def transferEvent : Nat := 93311
def frameStart : Nat := 93234
def rule : BoundRule := .product (.predecessor 0 93309 .coefficient) (.predecessor 1 93310 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93309 .coefficient)
      LeftBound93307.bound (LeftBound93307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93310 .coefficient)
      LeftAuthority93284.bound (LeftAuthority93284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93284.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93284.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93307.bound LeftAuthority93284.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93307.bound, LeftAuthority93284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93307.actual selector witness) * (LeftAuthority93284.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93311

namespace LeftBound93322
def owner : Owner := ⟨.program ⟨214⟩, ⟨15519⟩⟩
def transferEvent : Nat := 93322
def frameStart : Nat := 93234
def rule : BoundRule := .product (.predecessor 0 93320 .coefficient) (.predecessor 1 93321 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93320 .coefficient)
      LeftAuthority93295.bound (LeftAuthority93295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93321 .coefficient)
      LeftAuthority93318.bound (LeftAuthority93318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93318.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority93295.bound LeftAuthority93318.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93295.bound, LeftAuthority93318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority93295.actual selector witness) * (LeftAuthority93318.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93322

namespace LeftBound93330
def owner : Owner := ⟨.program ⟨214⟩, ⟨15520⟩⟩
def transferEvent : Nat := 93330
def frameStart : Nat := 93234
def rule : BoundRule := .sum [.predecessor 0 93328 .coefficient, .predecessor 1 93329 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93328 .coefficient)
      LeftAuthority93326.bound (LeftAuthority93326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93329 .coefficient)
      LeftBound93322.bound (LeftBound93322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93326.bound, LeftBound93322.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93326.bound, LeftBound93322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93326.actual selector witness, LeftBound93322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93330

namespace LeftBound93334
def owner : Owner := ⟨.program ⟨214⟩, ⟨26997⟩⟩
def transferEvent : Nat := 93334
def frameStart : Nat := 93234
def rule : BoundRule := .sum [.predecessor 0 93332 .coefficient, .predecessor 1 93333 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93332 .coefficient)
      LeftBound93330.bound (LeftBound93330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93333 .coefficient)
      LeftBound93311.bound (LeftBound93311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93330.bound, LeftBound93311.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93330.bound, LeftBound93311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93330.actual selector witness, LeftBound93311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93334

namespace LeftBound93347
def owner : Owner := ⟨.program ⟨214⟩, ⟨26994⟩⟩
def transferEvent : Nat := 93347
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 93345 .coefficient, .predecessor 1 93346 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93345 .coefficient)
      LeftBound93176.bound (LeftBound93176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93346 .coefficient)
      LeftBound93159.bound (LeftBound93159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93176.bound, LeftBound93159.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93176.bound, LeftBound93159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93176.actual selector witness, LeftBound93159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93347

namespace LeftBound93350
def owner : Owner := ⟨.program ⟨214⟩, ⟨26994⟩⟩
def transferEvent : Nat := 93350
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 93344 .summary, .result 93166 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93344 .summary)
      LeftBound93178.bound (LeftBound93178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20755⟩⟩) (rawTerms := some (Proof.Events364.exact93344RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93166 .summary)
      LeftBound93161.bound (LeftBound93161.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26993⟩⟩) (rawTerms := some (Proof.Events363.exact93166RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93178.bound, LeftBound93161.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93178.bound, LeftBound93161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93178.actual selector witness, LeftBound93161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93350

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
