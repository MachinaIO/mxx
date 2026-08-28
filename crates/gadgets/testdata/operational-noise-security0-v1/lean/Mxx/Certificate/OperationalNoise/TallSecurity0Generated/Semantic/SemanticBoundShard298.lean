import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard297

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44040
def owner : Owner := ⟨.program ⟨214⟩, ⟨26592⟩⟩
def transferEvent : Nat := 44040
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 44035 .summary) (.transfer 44039) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44035 .summary)
      LeftBound44034.bound (LeftBound44034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25000⟩⟩) (rawTerms := some (Proof.Events172.exact44035RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 44039)
      LeftBound44039.bound (LeftBound44039.actual selector witness) := by
  exact .transfer (LeftBound44039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44034.bound LeftBound44039.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44034.bound, LeftBound44039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44034.actual selector witness) * (LeftBound44039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44040

namespace LeftBound44051
def owner : Owner := ⟨.program ⟨214⟩, ⟨20546⟩⟩
def transferEvent : Nat := 44051
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 44049 .coefficient) (.value (.predecessor 1 44050 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44049 .coefficient)
      LeftAuthority44047.bound (LeftAuthority44047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44047.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44050 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority44047.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44047.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44047.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound44051

namespace LeftBound44055
def owner : Owner := ⟨.program ⟨214⟩, ⟨20547⟩⟩
def transferEvent : Nat := 44055
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44053 .coefficient) (.predecessor 1 44054 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44053 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44054 .coefficient)
      LeftBound44051.bound (LeftBound44051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound44051.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound44051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound44051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44055

namespace LeftBound44056
def owner : Owner := ⟨.program ⟨214⟩, ⟨20547⟩⟩
def transferEvent : Nat := 44056
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩ [⟨.result 44048 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44048 .coefficient)
      LeftAuthority44047.bound (LeftAuthority44047.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20544⟩⟩) (rawTerms := some (Proof.Events172.exact44048RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44047.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44047.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority44047.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44047.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44056

namespace LeftBound44057
def owner : Owner := ⟨.program ⟨214⟩, ⟨20547⟩⟩
def transferEvent : Nat := 44057
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 44056) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 44056)
      LeftBound44056.bound (LeftBound44056.actual selector witness) := by
  exact .transfer (LeftBound44056.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound44056.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound44056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound44056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44057

namespace LeftBound44152
def owner : Owner := ⟨.program ⟨214⟩, ⟨14962⟩⟩
def transferEvent : Nat := 44152
def frameStart : Nat := 44113
def rule : BoundRule := .identity (.predecessor 0 44151 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44151 .coefficient)
      LeftAuthority44149.bound (LeftAuthority44149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44149.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44149.derived selector witness)

def rawBound : CoeffClass := LeftAuthority44149.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority44149.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44152

namespace LeftBound44169
def owner : Owner := ⟨.program ⟨214⟩, ⟨15001⟩⟩
def transferEvent : Nat := 44169
def frameStart : Nat := 44113
def rule : BoundRule := .sum [.predecessor 0 44167 .coefficient, .predecessor 1 44168 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44167 .coefficient)
      LeftBound44152.bound (LeftBound44152.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound44152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44168 .coefficient)
      LeftAuthority44165.bound (LeftAuthority44165.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority44165.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44152.bound, LeftAuthority44165.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44152.bound, LeftAuthority44165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44152.actual selector witness, LeftAuthority44165.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44169

namespace LeftBound44172
def owner : Owner := ⟨.program ⟨214⟩, ⟨15002⟩⟩
def transferEvent : Nat := 44172
def frameStart : Nat := 44113
def rule : BoundRule := .identity (.predecessor 0 44171 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44171 .coefficient)
      LeftBound44169.bound (LeftBound44169.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound44169.derived selector witness)

def rawBound : CoeffClass := LeftBound44169.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound44169.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44172

namespace LeftBound44178
def owner : Owner := ⟨.program ⟨214⟩, ⟨15003⟩⟩
def transferEvent : Nat := 44178
def frameStart : Nat := 44113
def rule : BoundRule := .product (.predecessor 0 44176 .coefficient) (.predecessor 1 44177 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44176 .coefficient)
      LeftAuthority44174.bound (LeftAuthority44174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44177 .coefficient)
      LeftBound44172.bound (LeftBound44172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44172.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority44174.bound LeftBound44172.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44174.bound, LeftBound44172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority44174.actual selector witness) * (LeftBound44172.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44178

namespace LeftBound44186
def owner : Owner := ⟨.program ⟨214⟩, ⟨15004⟩⟩
def transferEvent : Nat := 44186
def frameStart : Nat := 44113
def rule : BoundRule := .sum [.predecessor 0 44184 .coefficient, .predecessor 1 44185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44184 .coefficient)
      LeftAuthority44182.bound (LeftAuthority44182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44185 .coefficient)
      LeftBound44178.bound (LeftBound44178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority44182.bound, LeftBound44178.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44182.bound, LeftBound44178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority44182.actual selector witness, LeftBound44178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44186

namespace LeftBound44190
def owner : Owner := ⟨.program ⟨214⟩, ⟨26591⟩⟩
def transferEvent : Nat := 44190
def frameStart : Nat := 44113
def rule : BoundRule := .product (.predecessor 0 44188 .coefficient) (.predecessor 1 44189 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44188 .coefficient)
      LeftBound44186.bound (LeftBound44186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44189 .coefficient)
      LeftAuthority44163.bound (LeftAuthority44163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44163.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44163.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44186.bound LeftAuthority44163.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44186.bound, LeftAuthority44163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44186.actual selector witness) * (LeftAuthority44163.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44190

namespace LeftBound44201
def owner : Owner := ⟨.program ⟨214⟩, ⟨15320⟩⟩
def transferEvent : Nat := 44201
def frameStart : Nat := 44113
def rule : BoundRule := .product (.predecessor 0 44199 .coefficient) (.predecessor 1 44200 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44199 .coefficient)
      LeftAuthority44174.bound (LeftAuthority44174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44200 .coefficient)
      LeftAuthority44197.bound (LeftAuthority44197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44197.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44197.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority44174.bound LeftAuthority44197.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44174.bound, LeftAuthority44197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority44174.actual selector witness) * (LeftAuthority44197.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44201

namespace LeftBound44209
def owner : Owner := ⟨.program ⟨214⟩, ⟨15321⟩⟩
def transferEvent : Nat := 44209
def frameStart : Nat := 44113
def rule : BoundRule := .sum [.predecessor 0 44207 .coefficient, .predecessor 1 44208 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44207 .coefficient)
      LeftAuthority44205.bound (LeftAuthority44205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44205.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44208 .coefficient)
      LeftBound44201.bound (LeftBound44201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority44205.bound, LeftBound44201.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44205.bound, LeftBound44201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority44205.actual selector witness, LeftBound44201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44209

namespace LeftBound44213
def owner : Owner := ⟨.program ⟨214⟩, ⟨26595⟩⟩
def transferEvent : Nat := 44213
def frameStart : Nat := 44113
def rule : BoundRule := .sum [.predecessor 0 44211 .coefficient, .predecessor 1 44212 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44211 .coefficient)
      LeftBound44209.bound (LeftBound44209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44212 .coefficient)
      LeftBound44190.bound (LeftBound44190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44190.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44190.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44209.bound, LeftBound44190.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44209.bound, LeftBound44190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44209.actual selector witness, LeftBound44190.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44213

namespace LeftBound44226
def owner : Owner := ⟨.program ⟨214⟩, ⟨26593⟩⟩
def transferEvent : Nat := 44226
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44224 .coefficient, .predecessor 1 44225 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44224 .coefficient)
      LeftBound44055.bound (LeftBound44055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44225 .coefficient)
      LeftBound44038.bound (LeftBound44038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44038.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44038.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44055.bound, LeftBound44038.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44055.bound, LeftBound44038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44055.actual selector witness, LeftBound44038.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44226

namespace LeftBound44229
def owner : Owner := ⟨.program ⟨214⟩, ⟨26593⟩⟩
def transferEvent : Nat := 44229
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44223 .summary, .result 44045 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44223 .summary)
      LeftBound44057.bound (LeftBound44057.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20547⟩⟩) (rawTerms := some (Proof.Events172.exact44223RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44045 .summary)
      LeftBound44040.bound (LeftBound44040.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26592⟩⟩) (rawTerms := some (Proof.Events172.exact44045RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44040.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44057.bound, LeftBound44040.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44057.bound, LeftBound44040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44057.actual selector witness, LeftBound44040.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44229

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
