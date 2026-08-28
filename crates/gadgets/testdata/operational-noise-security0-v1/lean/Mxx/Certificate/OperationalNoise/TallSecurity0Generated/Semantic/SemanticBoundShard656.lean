import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96197
def owner : Owner := ⟨.program ⟨214⟩, ⟨19951⟩⟩
def transferEvent : Nat := 96197
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 96195 .coefficient) (.value (.predecessor 1 96196 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96195 .coefficient)
      LeftAuthority96193.bound (LeftAuthority96193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96193.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96196 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority96193.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96193.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96193.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96197

namespace LeftBound96201
def owner : Owner := ⟨.program ⟨214⟩, ⟨19952⟩⟩
def transferEvent : Nat := 96201
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96199 .coefficient) (.predecessor 1 96200 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96199 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96200 .coefficient)
      LeftBound96197.bound (LeftBound96197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96197.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound96197.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound96197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound96197.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96201

namespace LeftBound96202
def owner : Owner := ⟨.program ⟨214⟩, ⟨19952⟩⟩
def transferEvent : Nat := 96202
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19949⟩⟩]⟩ [⟨.result 96194 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96194 .coefficient)
      LeftAuthority96193.bound (LeftAuthority96193.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19949⟩⟩) (rawTerms := some (Proof.Events375.exact96194RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96193.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96193.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96193.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96193.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96202

namespace LeftBound96203
def owner : Owner := ⟨.program ⟨214⟩, ⟨19952⟩⟩
def transferEvent : Nat := 96203
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 96202) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96202)
      LeftBound96202.bound (LeftBound96202.actual selector witness) := by
  exact .transfer (LeftBound96202.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound96202.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound96202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound96202.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96203

namespace LeftBound96258
def owner : Owner := ⟨.program ⟨214⟩, ⟨12543⟩⟩
def transferEvent : Nat := 96258
def frameStart : Nat := 96241
def rule : BoundRule := .product (.predecessor 0 96256 .coefficient) (.predecessor 1 96257 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96256 .coefficient)
      LeftAuthority96254.bound (LeftAuthority96254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96254.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96257 .coefficient)
      LeftAuthority96251.bound (LeftAuthority96251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96251.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96251.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority96254.bound LeftAuthority96251.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96254.bound, LeftAuthority96251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority96254.actual selector witness) * (LeftAuthority96251.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96258

namespace LeftBound96262
def owner : Owner := ⟨.program ⟨214⟩, ⟨12544⟩⟩
def transferEvent : Nat := 96262
def frameStart : Nat := 96241
def rule : BoundRule := .identity (.predecessor 0 96261 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96261 .coefficient)
      LeftBound96258.bound (LeftBound96258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96258.derived selector witness)

def rawBound : CoeffClass := LeftBound96258.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound96258.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96262

namespace LeftBound96279
def owner : Owner := ⟨.program ⟨214⟩, ⟨12654⟩⟩
def transferEvent : Nat := 96279
def frameStart : Nat := 96241
def rule : BoundRule := .sum [.predecessor 0 96277 .coefficient, .predecessor 1 96278 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96277 .coefficient)
      LeftBound96262.bound (LeftBound96262.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96278 .coefficient)
      LeftAuthority96275.bound (LeftAuthority96275.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96275.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96262.bound, LeftAuthority96275.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96262.bound, LeftAuthority96275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96262.actual selector witness, LeftAuthority96275.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96279

namespace LeftBound96282
def owner : Owner := ⟨.program ⟨214⟩, ⟨12655⟩⟩
def transferEvent : Nat := 96282
def frameStart : Nat := 96241
def rule : BoundRule := .identity (.predecessor 0 96281 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96281 .coefficient)
      LeftBound96279.bound (LeftBound96279.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96279.derived selector witness)

def rawBound : CoeffClass := LeftBound96279.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound96279.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96282

namespace LeftBound96288
def owner : Owner := ⟨.program ⟨214⟩, ⟨12656⟩⟩
def transferEvent : Nat := 96288
def frameStart : Nat := 96241
def rule : BoundRule := .product (.predecessor 0 96286 .coefficient) (.predecessor 1 96287 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96286 .coefficient)
      LeftAuthority96284.bound (LeftAuthority96284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96284.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96287 .coefficient)
      LeftBound96282.bound (LeftBound96282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96282.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority96284.bound LeftBound96282.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96284.bound, LeftBound96282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority96284.actual selector witness) * (LeftBound96282.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96288

namespace LeftBound96304
def owner : Owner := ⟨.program ⟨214⟩, ⟨7871⟩⟩
def transferEvent : Nat := 96304
def frameStart : Nat := 96241
def rule : BoundRule := .scale (.predecessor 0 96302 .coefficient) (.value (.predecessor 1 96303 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96302 .coefficient)
      LeftAuthority96300.bound (LeftAuthority96300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96300.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96303 .coefficient)
      LeftAuthority96291.bound (LeftAuthority96291.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96291.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority96300.bound LeftAuthority96291.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96300.bound, LeftAuthority96291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96300.actual selector witness) * (LeftAuthority96291.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96304

namespace LeftBound96307
def owner : Owner := ⟨.program ⟨214⟩, ⟨6766⟩⟩
def transferEvent : Nat := 96307
def frameStart : Nat := 96241
def rule : BoundRule := .identity (.predecessor 0 96306 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96306 .coefficient)
      LeftAuthority96294.bound (LeftAuthority96294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96294.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96294.derived selector witness)

def rawBound : CoeffClass := LeftAuthority96294.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority96294.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96307

namespace LeftBound96311
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def transferEvent : Nat := 96311
def frameStart : Nat := 96241
def rule : BoundRule := .product (.predecessor 0 96309 .coefficient) (.predecessor 1 96310 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96309 .coefficient)
      LeftBound96307.bound (LeftBound96307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96310 .coefficient)
      LeftBound96304.bound (LeftBound96304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96304.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96304.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96307.bound LeftBound96304.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96307.bound, LeftBound96304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96307.actual selector witness) * (LeftBound96304.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96311

namespace LeftBound96316
def owner : Owner := ⟨.program ⟨214⟩, ⟨12657⟩⟩
def transferEvent : Nat := 96316
def frameStart : Nat := 96241
def rule : BoundRule := .sum [.predecessor 0 96314 .coefficient, .predecessor 1 96315 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96314 .coefficient)
      LeftBound96311.bound (LeftBound96311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96311.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96315 .coefficient)
      LeftBound96288.bound (LeftBound96288.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96288.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96288.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96311.bound, LeftBound96288.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96311.bound, LeftBound96288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96311.actual selector witness, LeftBound96288.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96316

namespace LeftBound96320
def owner : Owner := ⟨.program ⟨214⟩, ⟨25440⟩⟩
def transferEvent : Nat := 96320
def frameStart : Nat := 96241
def rule : BoundRule := .product (.predecessor 0 96318 .coefficient) (.predecessor 1 96319 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96318 .coefficient)
      LeftBound96316.bound (LeftBound96316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96319 .coefficient)
      LeftAuthority96273.bound (LeftAuthority96273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96273.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96273.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96316.bound LeftAuthority96273.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96316.bound, LeftAuthority96273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96316.actual selector witness) * (LeftAuthority96273.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96320

namespace LeftBound96331
def owner : Owner := ⟨.program ⟨214⟩, ⟨16541⟩⟩
def transferEvent : Nat := 96331
def frameStart : Nat := 96241
def rule : BoundRule := .product (.predecessor 0 96329 .coefficient) (.predecessor 1 96330 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96329 .coefficient)
      LeftAuthority96284.bound (LeftAuthority96284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96284.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96330 .coefficient)
      LeftAuthority96327.bound (LeftAuthority96327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96327.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority96284.bound LeftAuthority96327.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96284.bound, LeftAuthority96327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority96284.actual selector witness) * (LeftAuthority96327.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96331

namespace LeftBound96339
def owner : Owner := ⟨.program ⟨214⟩, ⟨16542⟩⟩
def transferEvent : Nat := 96339
def frameStart : Nat := 96241
def rule : BoundRule := .sum [.predecessor 0 96337 .coefficient, .predecessor 1 96338 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96337 .coefficient)
      LeftAuthority96335.bound (LeftAuthority96335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96338 .coefficient)
      LeftBound96331.bound (LeftBound96331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96331.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority96335.bound, LeftBound96331.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96335.bound, LeftBound96331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority96335.actual selector witness, LeftBound96331.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96339

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
