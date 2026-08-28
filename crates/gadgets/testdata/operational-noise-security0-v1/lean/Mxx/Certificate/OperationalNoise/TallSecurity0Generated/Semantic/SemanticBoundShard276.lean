import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard274
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard275

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound41139
def owner : Owner := ⟨.program ⟨214⟩, ⟨26078⟩⟩
def transferEvent : Nat := 41139
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41137 .coefficient, .predecessor 1 41138 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41137 .coefficient)
      LeftBound40960.bound (LeftBound40960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41138 .coefficient)
      LeftBound40943.bound (LeftBound40943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40960.bound, LeftBound40943.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40960.bound, LeftBound40943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40960.actual selector witness, LeftBound40943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41139

namespace LeftBound41142
def owner : Owner := ⟨.program ⟨214⟩, ⟨26078⟩⟩
def transferEvent : Nat := 41142
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 41136 .summary, .result 40950 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41136 .summary)
      LeftBound40962.bound (LeftBound40962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19539⟩⟩) (rawTerms := some (Proof.Events160.exact41136RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40950 .summary)
      LeftBound40945.bound (LeftBound40945.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26077⟩⟩) (rawTerms := some (Proof.Events159.exact40950RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40962.bound, LeftBound40945.bound]
def bound : CoeffClass := .finite ⟨352060719116288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40962.bound, LeftBound40945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40962.actual selector witness, LeftBound40945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41142

namespace LeftBound41146
def owner : Owner := ⟨.program ⟨214⟩, ⟨27894⟩⟩
def transferEvent : Nat := 41146
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41144 .coefficient) (.predecessor 1 41145 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41144 .coefficient)
      LeftBound41139.bound (LeftBound41139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41145 .coefficient)
      LeftAuthority40865.bound (LeftAuthority40865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40865.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41139.bound LeftAuthority40865.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41139.bound, LeftAuthority40865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41139.actual selector witness) * (LeftAuthority40865.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41146

namespace LeftBound41147
def owner : Owner := ⟨.program ⟨214⟩, ⟨27894⟩⟩
def transferEvent : Nat := 41147
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27892⟩⟩]⟩ [⟨.result 40866 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40866 .coefficient)
      LeftAuthority40865.bound (LeftAuthority40865.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27892⟩⟩) (rawTerms := some (Proof.Events159.exact40866RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40865.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority40865.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40865.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41147

namespace LeftBound41148
def owner : Owner := ⟨.program ⟨214⟩, ⟨27894⟩⟩
def transferEvent : Nat := 41148
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41143 .summary) (.transfer 41147) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41143 .summary)
      LeftBound41142.bound (LeftBound41142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26078⟩⟩) (rawTerms := some (Proof.Events160.exact41143RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41142.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41147)
      LeftBound41147.bound (LeftBound41147.actual selector witness) := by
  exact .transfer (LeftBound41147.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41142.bound LeftBound41147.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41142.bound, LeftBound41147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41142.actual selector witness) * (LeftBound41147.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41148

namespace LeftBound41159
def owner : Owner := ⟨.program ⟨214⟩, ⟨21410⟩⟩
def transferEvent : Nat := 41159
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 41157 .coefficient) (.value (.predecessor 1 41158 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41157 .coefficient)
      LeftAuthority41155.bound (LeftAuthority41155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41158 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority41155.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41155.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41155.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound41159

namespace LeftBound41163
def owner : Owner := ⟨.program ⟨214⟩, ⟨21411⟩⟩
def transferEvent : Nat := 41163
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41161 .coefficient) (.predecessor 1 41162 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41161 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41162 .coefficient)
      LeftBound41159.bound (LeftBound41159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound41159.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound41159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound41159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41163

namespace LeftBound41164
def owner : Owner := ⟨.program ⟨214⟩, ⟨21411⟩⟩
def transferEvent : Nat := 41164
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21408⟩⟩]⟩ [⟨.result 41156 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41156 .coefficient)
      LeftAuthority41155.bound (LeftAuthority41155.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21408⟩⟩) (rawTerms := some (Proof.Events160.exact41156RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41155.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority41155.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41155.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41164

namespace LeftBound41165
def owner : Owner := ⟨.program ⟨214⟩, ⟨21411⟩⟩
def transferEvent : Nat := 41165
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 41164) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41164)
      LeftBound41164.bound (LeftBound41164.actual selector witness) := by
  exact .transfer (LeftBound41164.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound41164.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound41164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound41164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41165

namespace LeftBound41260
def owner : Owner := ⟨.program ⟨214⟩, ⟨15949⟩⟩
def transferEvent : Nat := 41260
def frameStart : Nat := 41221
def rule : BoundRule := .identity (.predecessor 0 41259 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41259 .coefficient)
      LeftAuthority41257.bound (LeftAuthority41257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41257.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41257.derived selector witness)

def rawBound : CoeffClass := LeftAuthority41257.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority41257.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41260

namespace LeftBound41277
def owner : Owner := ⟨.program ⟨214⟩, ⟨16023⟩⟩
def transferEvent : Nat := 41277
def frameStart : Nat := 41221
def rule : BoundRule := .sum [.predecessor 0 41275 .coefficient, .predecessor 1 41276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41275 .coefficient)
      LeftBound41260.bound (LeftBound41260.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound41260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41276 .coefficient)
      LeftAuthority41273.bound (LeftAuthority41273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority41273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41260.bound, LeftAuthority41273.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41260.bound, LeftAuthority41273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41260.actual selector witness, LeftAuthority41273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41277

namespace LeftBound41280
def owner : Owner := ⟨.program ⟨214⟩, ⟨16024⟩⟩
def transferEvent : Nat := 41280
def frameStart : Nat := 41221
def rule : BoundRule := .identity (.predecessor 0 41279 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41279 .coefficient)
      LeftBound41277.bound (LeftBound41277.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound41277.derived selector witness)

def rawBound : CoeffClass := LeftBound41277.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound41277.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41280

namespace LeftBound41286
def owner : Owner := ⟨.program ⟨214⟩, ⟨16025⟩⟩
def transferEvent : Nat := 41286
def frameStart : Nat := 41221
def rule : BoundRule := .product (.predecessor 0 41284 .coefficient) (.predecessor 1 41285 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41284 .coefficient)
      LeftAuthority41282.bound (LeftAuthority41282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41285 .coefficient)
      LeftBound41280.bound (LeftBound41280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority41282.bound LeftBound41280.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41282.bound, LeftBound41280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority41282.actual selector witness) * (LeftBound41280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41286

namespace LeftBound41294
def owner : Owner := ⟨.program ⟨214⟩, ⟨16026⟩⟩
def transferEvent : Nat := 41294
def frameStart : Nat := 41221
def rule : BoundRule := .sum [.predecessor 0 41292 .coefficient, .predecessor 1 41293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41292 .coefficient)
      LeftAuthority41290.bound (LeftAuthority41290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41290.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41293 .coefficient)
      LeftBound41286.bound (LeftBound41286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority41290.bound, LeftBound41286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41290.bound, LeftBound41286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority41290.actual selector witness, LeftBound41286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41294

namespace LeftBound41298
def owner : Owner := ⟨.program ⟨214⟩, ⟨27893⟩⟩
def transferEvent : Nat := 41298
def frameStart : Nat := 41221
def rule : BoundRule := .product (.predecessor 0 41296 .coefficient) (.predecessor 1 41297 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41296 .coefficient)
      LeftBound41294.bound (LeftBound41294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41297 .coefficient)
      LeftAuthority41271.bound (LeftAuthority41271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41271.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41271.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41294.bound LeftAuthority41271.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41294.bound, LeftAuthority41271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41294.actual selector witness) * (LeftAuthority41271.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41298

namespace LeftBound41309
def owner : Owner := ⟨.program ⟨214⟩, ⟨15993⟩⟩
def transferEvent : Nat := 41309
def frameStart : Nat := 41221
def rule : BoundRule := .product (.predecessor 0 41307 .coefficient) (.predecessor 1 41308 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41307 .coefficient)
      LeftAuthority41282.bound (LeftAuthority41282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41308 .coefficient)
      LeftAuthority41305.bound (LeftAuthority41305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41305.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41305.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority41282.bound LeftAuthority41305.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41282.bound, LeftAuthority41305.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority41282.actual selector witness) * (LeftAuthority41305.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41309

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
