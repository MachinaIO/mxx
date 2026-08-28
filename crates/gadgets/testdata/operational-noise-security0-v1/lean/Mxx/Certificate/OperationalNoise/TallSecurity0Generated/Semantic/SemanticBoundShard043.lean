import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard042

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound8206
def owner : Owner := ⟨.program ⟨214⟩, ⟨12877⟩⟩
def transferEvent : Nat := 8206
def frameStart : Nat := 8119
def rule : BoundRule := .sum [.predecessor 0 8204 .coefficient, .predecessor 1 8205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8204 .coefficient)
      LeftBound8201.bound (LeftBound8201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8205 .coefficient)
      LeftBound8178.bound (LeftBound8178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8201.bound, LeftBound8178.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8201.bound, LeftBound8178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8201.actual selector witness, LeftBound8178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8206

namespace LeftBound8210
def owner : Owner := ⟨.program ⟨214⟩, ⟨25550⟩⟩
def transferEvent : Nat := 8210
def frameStart : Nat := 8119
def rule : BoundRule := .product (.predecessor 0 8208 .coefficient) (.predecessor 1 8209 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8208 .coefficient)
      LeftBound8206.bound (LeftBound8206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8209 .coefficient)
      LeftAuthority8163.bound (LeftAuthority8163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8163.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8163.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8206.bound LeftAuthority8163.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8206.bound, LeftAuthority8163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8206.actual selector witness) * (LeftAuthority8163.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8210

namespace LeftBound8221
def owner : Owner := ⟨.program ⟨214⟩, ⟨16651⟩⟩
def transferEvent : Nat := 8221
def frameStart : Nat := 8119
def rule : BoundRule := .product (.predecessor 0 8219 .coefficient) (.predecessor 1 8220 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8219 .coefficient)
      LeftAuthority8174.bound (LeftAuthority8174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8220 .coefficient)
      LeftAuthority8217.bound (LeftAuthority8217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8217.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority8174.bound LeftAuthority8217.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8174.bound, LeftAuthority8217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority8174.actual selector witness) * (LeftAuthority8217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8221

namespace LeftBound8229
def owner : Owner := ⟨.program ⟨214⟩, ⟨16652⟩⟩
def transferEvent : Nat := 8229
def frameStart : Nat := 8119
def rule : BoundRule := .sum [.predecessor 0 8227 .coefficient, .predecessor 1 8228 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8227 .coefficient)
      LeftAuthority8225.bound (LeftAuthority8225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8225.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8228 .coefficient)
      LeftBound8221.bound (LeftBound8221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority8225.bound, LeftBound8221.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8225.bound, LeftBound8221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority8225.actual selector witness, LeftBound8221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8229

namespace LeftBound8233
def owner : Owner := ⟨.program ⟨214⟩, ⟨25551⟩⟩
def transferEvent : Nat := 8233
def frameStart : Nat := 8119
def rule : BoundRule := .sum [.predecessor 0 8231 .coefficient, .predecessor 1 8232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8231 .coefficient)
      LeftBound8229.bound (LeftBound8229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8232 .coefficient)
      LeftBound8210.bound (LeftBound8210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8229.bound, LeftBound8210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8229.bound, LeftBound8210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8229.actual selector witness, LeftBound8210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8233

namespace LeftBound8246
def owner : Owner := ⟨.program ⟨214⟩, ⟨25549⟩⟩
def transferEvent : Nat := 8246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8244 .coefficient, .predecessor 1 8245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8244 .coefficient)
      LeftBound8067.bound (LeftBound8067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8245 .coefficient)
      LeftBound8050.bound (LeftBound8050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8067.bound, LeftBound8050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8067.bound, LeftBound8050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8067.actual selector witness, LeftBound8050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8246

namespace LeftBound8249
def owner : Owner := ⟨.program ⟨214⟩, ⟨25549⟩⟩
def transferEvent : Nat := 8249
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 8243 .summary, .result 8057 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8243 .summary)
      LeftBound8069.bound (LeftBound8069.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20051⟩⟩) (rawTerms := some (Proof.Events032.exact8243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8057 .summary)
      LeftBound8052.bound (LeftBound8052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25548⟩⟩) (rawTerms := some (Proof.Events031.exact8057RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8069.bound, LeftBound8052.bound]
def bound : CoeffClass := .finite ⟨352146215809024, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8069.bound, LeftBound8052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8069.actual selector witness, LeftBound8052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8249

namespace LeftBound8253
def owner : Owner := ⟨.program ⟨214⟩, ⟨29439⟩⟩
def transferEvent : Nat := 8253
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8251 .coefficient) (.predecessor 1 8252 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8251 .coefficient)
      LeftBound8246.bound (LeftBound8246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8252 .coefficient)
      LeftAuthority7953.bound (LeftAuthority7953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7953.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8246.bound LeftAuthority7953.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8246.bound, LeftAuthority7953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8246.actual selector witness) * (LeftAuthority7953.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8253

namespace LeftBound8254
def owner : Owner := ⟨.program ⟨214⟩, ⟨29439⟩⟩
def transferEvent : Nat := 8254
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩ [⟨.result 7954 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7954 .coefficient)
      LeftAuthority7953.bound (LeftAuthority7953.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29437⟩⟩) (rawTerms := some (Proof.Events031.exact7954RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7953.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7953.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7953.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound8254

namespace LeftBound8255
def owner : Owner := ⟨.program ⟨214⟩, ⟨29439⟩⟩
def transferEvent : Nat := 8255
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 8250 .summary) (.transfer 8254) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8250 .summary)
      LeftBound8249.bound (LeftBound8249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25549⟩⟩) (rawTerms := some (Proof.Events032.exact8250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 8254)
      LeftBound8254.bound (LeftBound8254.actual selector witness) := by
  exact .transfer (LeftBound8254.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8249.bound LeftBound8254.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8249.bound, LeftBound8254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8249.actual selector witness) * (LeftBound8254.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8255

namespace LeftBound8266
def owner : Owner := ⟨.program ⟨214⟩, ⟨22426⟩⟩
def transferEvent : Nat := 8266
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 8264 .coefficient) (.value (.predecessor 1 8265 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8264 .coefficient)
      LeftAuthority8262.bound (LeftAuthority8262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8265 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority8262.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8262.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8262.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound8266

namespace LeftBound8270
def owner : Owner := ⟨.program ⟨214⟩, ⟨22427⟩⟩
def transferEvent : Nat := 8270
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8268 .coefficient) (.predecessor 1 8269 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8268 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8269 .coefficient)
      LeftBound8266.bound (LeftBound8266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound8266.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound8266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound8266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8270

namespace LeftBound8271
def owner : Owner := ⟨.program ⟨214⟩, ⟨22427⟩⟩
def transferEvent : Nat := 8271
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22424⟩⟩]⟩ [⟨.result 8263 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8263 .coefficient)
      LeftAuthority8262.bound (LeftAuthority8262.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22424⟩⟩) (rawTerms := some (Proof.Events032.exact8263RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8262.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8262.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8262.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound8271

namespace LeftBound8272
def owner : Owner := ⟨.program ⟨214⟩, ⟨22427⟩⟩
def transferEvent : Nat := 8272
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 8271) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 8271)
      LeftBound8271.bound (LeftBound8271.actual selector witness) := by
  exact .transfer (LeftBound8271.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound8271.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound8271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound8271.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8272

namespace LeftBound8367
def owner : Owner := ⟨.program ⟨214⟩, ⟨16650⟩⟩
def transferEvent : Nat := 8367
def frameStart : Nat := 8328
def rule : BoundRule := .identity (.predecessor 0 8366 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8366 .coefficient)
      LeftAuthority8364.bound (LeftAuthority8364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8364.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8364.derived selector witness)

def rawBound : CoeffClass := LeftAuthority8364.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority8364.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound8367

namespace LeftBound8384
def owner : Owner := ⟨.program ⟨214⟩, ⟨16724⟩⟩
def transferEvent : Nat := 8384
def frameStart : Nat := 8328
def rule : BoundRule := .sum [.predecessor 0 8382 .coefficient, .predecessor 1 8383 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8382 .coefficient)
      LeftBound8367.bound (LeftBound8367.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound8367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8383 .coefficient)
      LeftAuthority8380.bound (LeftAuthority8380.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority8380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8367.bound, LeftAuthority8380.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8367.bound, LeftAuthority8380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8367.actual selector witness, LeftAuthority8380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8384

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
