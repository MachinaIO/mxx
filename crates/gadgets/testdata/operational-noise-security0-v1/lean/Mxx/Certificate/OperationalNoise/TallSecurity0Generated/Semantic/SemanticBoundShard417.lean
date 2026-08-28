import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard362
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard363
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard416

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound62200
def owner : Owner := ⟨.program ⟨214⟩, ⟨17556⟩⟩
def transferEvent : Nat := 62200
def frameStart : Nat := 62112
def rule : BoundRule := .product (.predecessor 0 62198 .coefficient) (.predecessor 1 62199 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62198 .coefficient)
      LeftAuthority62173.bound (LeftAuthority62173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact62174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62199 .coefficient)
      LeftAuthority62196.bound (LeftAuthority62196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact62197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62196.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62196.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority62173.bound LeftAuthority62196.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62173.bound, LeftAuthority62196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority62173.actual selector witness) * (LeftAuthority62196.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62200

namespace LeftBound62208
def owner : Owner := ⟨.program ⟨214⟩, ⟨17557⟩⟩
def transferEvent : Nat := 62208
def frameStart : Nat := 62112
def rule : BoundRule := .sum [.predecessor 0 62206 .coefficient, .predecessor 1 62207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62206 .coefficient)
      LeftAuthority62204.bound (LeftAuthority62204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact62205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62204.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62207 .coefficient)
      LeftBound62200.bound (LeftBound62200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact62202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62200.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority62204.bound, LeftBound62200.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62204.bound, LeftBound62200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority62204.actual selector witness, LeftBound62200.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62208

namespace LeftBound62212
def owner : Owner := ⟨.program ⟨214⟩, ⟨28963⟩⟩
def transferEvent : Nat := 62212
def frameStart : Nat := 62112
def rule : BoundRule := .sum [.predecessor 0 62210 .coefficient, .predecessor 1 62211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62210 .coefficient)
      LeftBound62208.bound (LeftBound62208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62211 .coefficient)
      LeftBound62189.bound (LeftBound62189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact62194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62189.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62208.bound, LeftBound62189.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62208.bound, LeftBound62189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62208.actual selector witness, LeftBound62189.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62212

namespace LeftBound62225
def owner : Owner := ⟨.program ⟨214⟩, ⟨28960⟩⟩
def transferEvent : Nat := 62225
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 62223 .coefficient, .predecessor 1 62224 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62223 .coefficient)
      LeftBound62054.bound (LeftBound62054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62224 .coefficient)
      LeftBound62037.bound (LeftBound62037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact62044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62037.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62037.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62054.bound, LeftBound62037.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62054.bound, LeftBound62037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62054.actual selector witness, LeftBound62037.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62225

namespace LeftBound62228
def owner : Owner := ⟨.program ⟨214⟩, ⟨28960⟩⟩
def transferEvent : Nat := 62228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 62222 .summary, .result 62044 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62222 .summary)
      LeftBound62056.bound (LeftBound62056.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22055⟩⟩) (rawTerms := some (Proof.Events243.exact62222RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62044 .summary)
      LeftBound62039.bound (LeftBound62039.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28959⟩⟩) (rawTerms := some (Proof.Events242.exact62044RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62039.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62056.bound, LeftBound62039.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62056.bound, LeftBound62039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62056.actual selector witness, LeftBound62039.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62228

namespace LeftBound62232
def owner : Owner := ⟨.program ⟨214⟩, ⟨28961⟩⟩
def transferEvent : Nat := 62232
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62230 .coefficient) (.predecessor 1 62231 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62230 .coefficient)
      LeftBound62225.bound (LeftBound62225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62225.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62231 .coefficient)
      LeftBound5618.bound (LeftBound5618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62225.bound LeftBound5618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62225.bound, LeftBound5618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62225.actual selector witness) * (LeftBound5618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62232

namespace LeftBound62233
def owner : Owner := ⟨.program ⟨214⟩, ⟨28961⟩⟩
def transferEvent : Nat := 62233
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩ [⟨.result 5615 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5615 .coefficient)
      LeftAuthority5614.bound (LeftAuthority5614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6669⟩⟩) (rawTerms := some (Proof.Events021.exact5615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5614.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62233

namespace LeftBound62234
def owner : Owner := ⟨.program ⟨214⟩, ⟨28961⟩⟩
def transferEvent : Nat := 62234
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 62229 .summary) (.transfer 62233) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62229 .summary)
      LeftBound62228.bound (LeftBound62228.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28960⟩⟩) (rawTerms := some (Proof.Events243.exact62229RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62233)
      LeftBound62233.bound (LeftBound62233.actual selector witness) := by
  exact .transfer (LeftBound62233.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62228.bound LeftBound62233.bound
def bound : CoeffClass := .finite ⟨4742816766803936246568583168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62228.bound, LeftBound62233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62228.actual selector witness) * (LeftBound62233.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62234

namespace LeftBound62249
def owner : Owner := ⟨.program ⟨214⟩, ⟨28742⟩⟩
def transferEvent : Nat := 62249
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62247 .coefficient) (.predecessor 1 62248 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62247 .coefficient)
      LeftBound53836.bound (LeftBound53836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62248 .coefficient)
      LeftAuthority62245.bound (LeftAuthority62245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62245.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62245.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53836.bound LeftAuthority62245.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53836.bound, LeftAuthority62245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53836.actual selector witness) * (LeftAuthority62245.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62249

namespace LeftBound62250
def owner : Owner := ⟨.program ⟨214⟩, ⟨28742⟩⟩
def transferEvent : Nat := 62250
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28740⟩⟩]⟩ [⟨.result 62246 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62246 .coefficient)
      LeftAuthority62245.bound (LeftAuthority62245.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28740⟩⟩) (rawTerms := some (Proof.Events243.exact62246RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62245.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62245.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62245.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62245.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62250

namespace LeftBound62251
def owner : Owner := ⟨.program ⟨214⟩, ⟨28742⟩⟩
def transferEvent : Nat := 62251
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 53840 .summary) (.transfer 62250) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53840 .summary)
      LeftBound53839.bound (LeftBound53839.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25226⟩⟩) (rawTerms := some (Proof.Events210.exact53840RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62250)
      LeftBound62250.bound (LeftBound62250.actual selector witness) := by
  exact .transfer (LeftBound62250.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53839.bound LeftBound62250.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53839.bound, LeftBound62250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53839.actual selector witness) * (LeftBound62250.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62251

namespace LeftBound62262
def owner : Owner := ⟨.program ⟨214⟩, ⟨21910⟩⟩
def transferEvent : Nat := 62262
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 62260 .coefficient) (.value (.predecessor 1 62261 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62260 .coefficient)
      LeftAuthority62258.bound (LeftAuthority62258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62258.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62261 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority62258.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62258.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62258.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound62262

namespace LeftBound62266
def owner : Owner := ⟨.program ⟨214⟩, ⟨21911⟩⟩
def transferEvent : Nat := 62266
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62264 .coefficient) (.predecessor 1 62265 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62264 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62265 .coefficient)
      LeftBound62262.bound (LeftBound62262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62262.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound62262.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound62262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound62262.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62266

namespace LeftBound62267
def owner : Owner := ⟨.program ⟨214⟩, ⟨21911⟩⟩
def transferEvent : Nat := 62267
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21908⟩⟩]⟩ [⟨.result 62259 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62259 .coefficient)
      LeftAuthority62258.bound (LeftAuthority62258.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21908⟩⟩) (rawTerms := some (Proof.Events243.exact62259RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62258.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62258.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62258.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62258.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62267

namespace LeftBound62268
def owner : Owner := ⟨.program ⟨214⟩, ⟨21911⟩⟩
def transferEvent : Nat := 62268
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 62267) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62267)
      LeftBound62267.bound (LeftBound62267.actual selector witness) := by
  exact .transfer (LeftBound62267.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound62267.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound62267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound62267.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62268

namespace LeftBound62363
def owner : Owner := ⟨.program ⟨214⟩, ⟨16386⟩⟩
def transferEvent : Nat := 62363
def frameStart : Nat := 62324
def rule : BoundRule := .identity (.predecessor 0 62362 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62362 .coefficient)
      LeftAuthority62360.bound (LeftAuthority62360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62360.derived selector witness)

def rawBound : CoeffClass := LeftAuthority62360.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority62360.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62363

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
