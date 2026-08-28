import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard387

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound57170
def owner : Owner := ⟨.program ⟨214⟩, ⟨13670⟩⟩
def transferEvent : Nat := 57170
def frameStart : Nat := 57083
def rule : BoundRule := .sum [.predecessor 0 57168 .coefficient, .predecessor 1 57169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57168 .coefficient)
      LeftBound57165.bound (LeftBound57165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57169 .coefficient)
      LeftBound57142.bound (LeftBound57142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57165.bound, LeftBound57142.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57165.bound, LeftBound57142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57165.actual selector witness, LeftBound57142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57170

namespace LeftBound57174
def owner : Owner := ⟨.program ⟨214⟩, ⟨25843⟩⟩
def transferEvent : Nat := 57174
def frameStart : Nat := 57083
def rule : BoundRule := .product (.predecessor 0 57172 .coefficient) (.predecessor 1 57173 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57172 .coefficient)
      LeftBound57170.bound (LeftBound57170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57173 .coefficient)
      LeftAuthority57127.bound (LeftAuthority57127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57127.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57127.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57170.bound LeftAuthority57127.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57170.bound, LeftAuthority57127.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57170.actual selector witness) * (LeftAuthority57127.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57174

namespace LeftBound57185
def owner : Owner := ⟨.program ⟨214⟩, ⟨15589⟩⟩
def transferEvent : Nat := 57185
def frameStart : Nat := 57083
def rule : BoundRule := .product (.predecessor 0 57183 .coefficient) (.predecessor 1 57184 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57183 .coefficient)
      LeftAuthority57138.bound (LeftAuthority57138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57184 .coefficient)
      LeftAuthority57181.bound (LeftAuthority57181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57181.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57181.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority57138.bound LeftAuthority57181.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57138.bound, LeftAuthority57181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority57138.actual selector witness) * (LeftAuthority57181.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57185

namespace LeftBound57193
def owner : Owner := ⟨.program ⟨214⟩, ⟨15590⟩⟩
def transferEvent : Nat := 57193
def frameStart : Nat := 57083
def rule : BoundRule := .sum [.predecessor 0 57191 .coefficient, .predecessor 1 57192 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57191 .coefficient)
      LeftAuthority57189.bound (LeftAuthority57189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57192 .coefficient)
      LeftBound57185.bound (LeftBound57185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57185.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57185.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57189.bound, LeftBound57185.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57189.bound, LeftBound57185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority57189.actual selector witness, LeftBound57185.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57193

namespace LeftBound57197
def owner : Owner := ⟨.program ⟨214⟩, ⟨25844⟩⟩
def transferEvent : Nat := 57197
def frameStart : Nat := 57083
def rule : BoundRule := .sum [.predecessor 0 57195 .coefficient, .predecessor 1 57196 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57195 .coefficient)
      LeftBound57193.bound (LeftBound57193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57196 .coefficient)
      LeftBound57174.bound (LeftBound57174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57193.bound, LeftBound57174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57193.bound, LeftBound57174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57193.actual selector witness, LeftBound57174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57197

namespace LeftBound57210
def owner : Owner := ⟨.program ⟨214⟩, ⟨25842⟩⟩
def transferEvent : Nat := 57210
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57208 .coefficient, .predecessor 1 57209 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57208 .coefficient)
      LeftBound57031.bound (LeftBound57031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57209 .coefficient)
      LeftBound57014.bound (LeftBound57014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact57021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57031.bound, LeftBound57014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57031.bound, LeftBound57014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57031.actual selector witness, LeftBound57014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57210

namespace LeftBound57213
def owner : Owner := ⟨.program ⟨214⟩, ⟨25842⟩⟩
def transferEvent : Nat := 57213
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 57207 .summary, .result 57021 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57207 .summary)
      LeftBound57033.bound (LeftBound57033.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19319⟩⟩) (rawTerms := some (Proof.Events223.exact57207RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57021 .summary)
      LeftBound57016.bound (LeftBound57016.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25841⟩⟩) (rawTerms := some (Proof.Events222.exact57021RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57033.bound, LeftBound57016.bound]
def bound : CoeffClass := .finite ⟨352036291489792, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57033.bound, LeftBound57016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57033.actual selector witness, LeftBound57016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57213

namespace LeftBound57217
def owner : Owner := ⟨.program ⟨214⟩, ⟨27230⟩⟩
def transferEvent : Nat := 57217
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57215 .coefficient) (.predecessor 1 57216 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57215 .coefficient)
      LeftBound57210.bound (LeftBound57210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57216 .coefficient)
      LeftAuthority56936.bound (LeftAuthority56936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56936.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57210.bound LeftAuthority56936.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57210.bound, LeftAuthority56936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57210.actual selector witness) * (LeftAuthority56936.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57217

namespace LeftBound57218
def owner : Owner := ⟨.program ⟨214⟩, ⟨27230⟩⟩
def transferEvent : Nat := 57218
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩ [⟨.result 56937 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56937 .coefficient)
      LeftAuthority56936.bound (LeftAuthority56936.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27228⟩⟩) (rawTerms := some (Proof.Events222.exact56937RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56936.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority56936.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56936.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57218

namespace LeftBound57219
def owner : Owner := ⟨.program ⟨214⟩, ⟨27230⟩⟩
def transferEvent : Nat := 57219
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 57214 .summary) (.transfer 57218) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57214 .summary)
      LeftBound57213.bound (LeftBound57213.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25842⟩⟩) (rawTerms := some (Proof.Events223.exact57214RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 57218)
      LeftBound57218.bound (LeftBound57218.actual selector witness) := by
  exact .transfer (LeftBound57218.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57213.bound LeftBound57218.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57213.bound, LeftBound57218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57213.actual selector witness) * (LeftBound57218.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57219

namespace LeftBound57230
def owner : Owner := ⟨.program ⟨214⟩, ⟨20974⟩⟩
def transferEvent : Nat := 57230
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 57228 .coefficient) (.value (.predecessor 1 57229 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57228 .coefficient)
      LeftAuthority57226.bound (LeftAuthority57226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57226.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57229 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority57226.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57226.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57226.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound57230

namespace LeftBound57234
def owner : Owner := ⟨.program ⟨214⟩, ⟨20975⟩⟩
def transferEvent : Nat := 57234
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57232 .coefficient) (.predecessor 1 57233 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57232 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57233 .coefficient)
      LeftBound57230.bound (LeftBound57230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57230.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound57230.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound57230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound57230.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57234

namespace LeftBound57235
def owner : Owner := ⟨.program ⟨214⟩, ⟨20975⟩⟩
def transferEvent : Nat := 57235
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20972⟩⟩]⟩ [⟨.result 57227 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57227 .coefficient)
      LeftAuthority57226.bound (LeftAuthority57226.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20972⟩⟩) (rawTerms := some (Proof.Events223.exact57227RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57226.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57226.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority57226.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57226.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57235

namespace LeftBound57236
def owner : Owner := ⟨.program ⟨214⟩, ⟨20975⟩⟩
def transferEvent : Nat := 57236
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 57235) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 57235)
      LeftBound57235.bound (LeftBound57235.actual selector witness) := by
  exact .transfer (LeftBound57235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound57235.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound57235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound57235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57236

namespace LeftBound57331
def owner : Owner := ⟨.program ⟨214⟩, ⟨15588⟩⟩
def transferEvent : Nat := 57331
def frameStart : Nat := 57292
def rule : BoundRule := .identity (.predecessor 0 57330 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57330 .coefficient)
      LeftAuthority57328.bound (LeftAuthority57328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57328.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57328.derived selector witness)

def rawBound : CoeffClass := LeftAuthority57328.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority57328.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57331

namespace LeftBound57348
def owner : Owner := ⟨.program ⟨214⟩, ⟨15662⟩⟩
def transferEvent : Nat := 57348
def frameStart : Nat := 57292
def rule : BoundRule := .sum [.predecessor 0 57346 .coefficient, .predecessor 1 57347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57346 .coefficient)
      LeftBound57331.bound (LeftBound57331.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57347 .coefficient)
      LeftAuthority57344.bound (LeftAuthority57344.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority57344.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57331.bound, LeftAuthority57344.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57331.bound, LeftAuthority57344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57331.actual selector witness, LeftAuthority57344.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57348

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
