import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard139
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard143
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard146
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard202

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound30161
def owner : Owner := ⟨.program ⟨214⟩, ⟨29645⟩⟩
def transferEvent : Nat := 30161
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30157 .summary, .result 22857 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30157 .summary)
      LeftBound30156.bound (LeftBound30156.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29428⟩⟩) (rawTerms := some (Proof.Events117.exact30157RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22857 .summary)
      LeftBound22856.bound (LeftBound22856.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29644⟩⟩) (rawTerms := some (Proof.Events089.exact22857RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22856.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30156.bound, LeftBound22856.bound]
def bound : CoeffClass := .finite ⟨20673980874611694436352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30156.bound, LeftBound22856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30156.actual selector witness, LeftBound22856.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30161

namespace LeftBound30165
def owner : Owner := ⟨.program ⟨214⟩, ⟨29862⟩⟩
def transferEvent : Nat := 30165
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30163 .coefficient, .predecessor 1 30164 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30163 .coefficient)
      LeftBound30160.bound (LeftBound30160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30164 .coefficient)
      LeftBound22371.bound (LeftBound22371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22371.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30160.bound, LeftBound22371.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30160.bound, LeftBound22371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30160.actual selector witness, LeftBound22371.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30165

namespace LeftBound30166
def owner : Owner := ⟨.program ⟨214⟩, ⟨29862⟩⟩
def transferEvent : Nat := 30166
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30162 .summary, .result 22375 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30162 .summary)
      LeftBound30161.bound (LeftBound30161.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29645⟩⟩) (rawTerms := some (Proof.Events117.exact30162RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30161.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22375 .summary)
      LeftBound22374.bound (LeftBound22374.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29861⟩⟩) (rawTerms := some (Proof.Events087.exact22375RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22374.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30161.bound, LeftBound22374.bound]
def bound : CoeffClass := .finite ⟨21966497597451692486656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30161.bound, LeftBound22374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30161.actual selector witness, LeftBound22374.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30166

namespace LeftBound30170
def owner : Owner := ⟨.program ⟨214⟩, ⟨30187⟩⟩
def transferEvent : Nat := 30170
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30168 .coefficient, .predecessor 1 30169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30168 .coefficient)
      LeftBound30165.bound (LeftBound30165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30169 .coefficient)
      LeftBound21889.bound (LeftBound21889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30165.bound, LeftBound21889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30165.bound, LeftBound21889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30165.actual selector witness, LeftBound21889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30170

namespace LeftBound30171
def owner : Owner := ⟨.program ⟨214⟩, ⟨30187⟩⟩
def transferEvent : Nat := 30171
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30167 .summary, .result 21893 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30167 .summary)
      LeftBound30166.bound (LeftBound30166.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29862⟩⟩) (rawTerms := some (Proof.Events117.exact30167RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21893 .summary)
      LeftBound21892.bound (LeftBound21892.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30186⟩⟩) (rawTerms := some (Proof.Events085.exact21893RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30166.bound, LeftBound21892.bound]
def bound : CoeffClass := .finite ⟨23259036732736711122944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30166.bound, LeftBound21892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30166.actual selector witness, LeftBound21892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30171

namespace LeftBound30175
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def transferEvent : Nat := 30175
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 30173 .coefficient) (.predecessor 1 30174 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30173 .coefficient)
      LeftBound30170.bound (LeftBound30170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30174 .coefficient)
      LeftAuthority21394.bound (LeftAuthority21394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21394.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21394.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound30170.bound LeftAuthority21394.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30170.bound, LeftAuthority21394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound30170.actual selector witness) * (LeftAuthority21394.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound30175

namespace LeftBound30176
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def transferEvent : Nat := 30176
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩ [⟨.result 21395 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21395 .coefficient)
      LeftAuthority21394.bound (LeftAuthority21394.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18690⟩⟩) (rawTerms := some (Proof.Events083.exact21395RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21394.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21394.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21394.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21394.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound30176

namespace LeftBound30177
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def transferEvent : Nat := 30177
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 30172 .summary) (.transfer 30176) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30172 .summary)
      LeftBound30171.bound (LeftBound30171.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30187⟩⟩) (rawTerms := some (Proof.Events117.exact30172RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 30176)
      LeftBound30176.bound (LeftBound30176.actual selector witness) := by
  exact .transfer (LeftBound30176.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound30171.bound LeftBound30176.bound
def bound : CoeffClass := .finite ⟨85361036953731453608582447104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30171.bound, LeftBound30176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound30171.actual selector witness) * (LeftBound30176.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound30177

namespace LeftBound30256
def owner : Owner := ⟨.program ⟨214⟩, ⟨18573⟩⟩
def transferEvent : Nat := 30256
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 30254 .coefficient) (.value (.predecessor 1 30255 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30254 .coefficient)
      LeftAuthority30252.bound (LeftAuthority30252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events118.exact30253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30255 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority30252.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority30252.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority30252.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound30256

namespace LeftBound30260
def owner : Owner := ⟨.program ⟨214⟩, ⟨18574⟩⟩
def transferEvent : Nat := 30260
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 30258 .coefficient) (.predecessor 1 30259 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30258 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30259 .coefficient)
      LeftBound30256.bound (LeftBound30256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events118.exact30257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30256.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound30256.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound30256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound30256.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound30260

namespace LeftBound30261
def owner : Owner := ⟨.program ⟨214⟩, ⟨18574⟩⟩
def transferEvent : Nat := 30261
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18571⟩⟩]⟩ [⟨.result 30253 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30253 .coefficient)
      LeftAuthority30252.bound (LeftAuthority30252.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18571⟩⟩) (rawTerms := some (Proof.Events118.exact30253RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30252.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority30252.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority30252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority30252.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound30261

namespace LeftBound30262
def owner : Owner := ⟨.program ⟨214⟩, ⟨18574⟩⟩
def transferEvent : Nat := 30262
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 30261) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 30261)
      LeftBound30261.bound (LeftBound30261.actual selector witness) := by
  exact .transfer (LeftBound30261.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound30261.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound30261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound30261.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound30262

namespace LeftBound31290
def owner : Owner := ⟨.program ⟨214⟩, ⟨15323⟩⟩
def transferEvent : Nat := 31290
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31288 .coefficient, .predecessor 1 31289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31288 .coefficient)
      LeftAuthority31286.bound (LeftAuthority31286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31289 .coefficient)
      LeftAuthority31263.bound (LeftAuthority31263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31263.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31286.bound, LeftAuthority31263.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31286.bound, LeftAuthority31263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority31286.actual selector witness, LeftAuthority31263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31290

namespace LeftBound31294
def owner : Owner := ⟨.program ⟨214⟩, ⟨15379⟩⟩
def transferEvent : Nat := 31294
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31292 .coefficient, .predecessor 1 31293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31292 .coefficient)
      LeftBound31290.bound (LeftBound31290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31293 .coefficient)
      LeftAuthority31240.bound (LeftAuthority31240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31240.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31290.bound, LeftAuthority31240.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31290.bound, LeftAuthority31240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31290.actual selector witness, LeftAuthority31240.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31294

namespace LeftBound31298
def owner : Owner := ⟨.program ⟨214⟩, ⟨17355⟩⟩
def transferEvent : Nat := 31298
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31296 .coefficient, .predecessor 1 31297 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31296 .coefficient)
      LeftBound31294.bound (LeftBound31294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31297 .coefficient)
      LeftAuthority31217.bound (LeftAuthority31217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31217.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31294.bound, LeftAuthority31217.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31294.bound, LeftAuthority31217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31294.actual selector witness, LeftAuthority31217.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31298

namespace LeftBound31302
def owner : Owner := ⟨.program ⟨214⟩, ⟨17356⟩⟩
def transferEvent : Nat := 31302
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31300 .coefficient, .predecessor 1 31301 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31300 .coefficient)
      LeftBound31298.bound (LeftBound31298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31298.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31301 .coefficient)
      LeftAuthority31194.bound (LeftAuthority31194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31298.bound, LeftAuthority31194.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31298.bound, LeftAuthority31194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31298.actual selector witness, LeftAuthority31194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31302

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
