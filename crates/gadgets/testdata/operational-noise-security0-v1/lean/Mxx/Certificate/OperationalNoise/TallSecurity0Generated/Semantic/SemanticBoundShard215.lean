import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard214

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound33018
def owner : Owner := ⟨.program ⟨214⟩, ⟨21919⟩⟩
def transferEvent : Nat := 33018
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 33017) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33017)
      LeftBound33017.bound (LeftBound33017.actual selector witness) := by
  exact .transfer (LeftBound33017.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound33017.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound33017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound33017.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33018

namespace LeftBound33113
def owner : Owner := ⟨.program ⟨214⟩, ⟨16394⟩⟩
def transferEvent : Nat := 33113
def frameStart : Nat := 33074
def rule : BoundRule := .identity (.predecessor 0 33112 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33112 .coefficient)
      LeftAuthority33110.bound (LeftAuthority33110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33110.derived selector witness)

def rawBound : CoeffClass := LeftAuthority33110.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority33110.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound33113

namespace LeftBound33130
def owner : Owner := ⟨.program ⟨214⟩, ⟨16433⟩⟩
def transferEvent : Nat := 33130
def frameStart : Nat := 33074
def rule : BoundRule := .sum [.predecessor 0 33128 .coefficient, .predecessor 1 33129 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33128 .coefficient)
      LeftBound33113.bound (LeftBound33113.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound33113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33129 .coefficient)
      LeftAuthority33126.bound (LeftAuthority33126.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority33126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33113.bound, LeftAuthority33126.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33113.bound, LeftAuthority33126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33113.actual selector witness, LeftAuthority33126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33130

namespace LeftBound33133
def owner : Owner := ⟨.program ⟨214⟩, ⟨16434⟩⟩
def transferEvent : Nat := 33133
def frameStart : Nat := 33074
def rule : BoundRule := .identity (.predecessor 0 33132 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33132 .coefficient)
      LeftBound33130.bound (LeftBound33130.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound33130.derived selector witness)

def rawBound : CoeffClass := LeftBound33130.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound33130.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound33133

namespace LeftBound33139
def owner : Owner := ⟨.program ⟨214⟩, ⟨16435⟩⟩
def transferEvent : Nat := 33139
def frameStart : Nat := 33074
def rule : BoundRule := .product (.predecessor 0 33137 .coefficient) (.predecessor 1 33138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33137 .coefficient)
      LeftAuthority33135.bound (LeftAuthority33135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33135.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33138 .coefficient)
      LeftBound33133.bound (LeftBound33133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33133.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority33135.bound LeftBound33133.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33135.bound, LeftBound33133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority33135.actual selector witness) * (LeftBound33133.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33139

namespace LeftBound33147
def owner : Owner := ⟨.program ⟨214⟩, ⟨16436⟩⟩
def transferEvent : Nat := 33147
def frameStart : Nat := 33074
def rule : BoundRule := .sum [.predecessor 0 33145 .coefficient, .predecessor 1 33146 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33145 .coefficient)
      LeftAuthority33143.bound (LeftAuthority33143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33143.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33146 .coefficient)
      LeftBound33139.bound (LeftBound33139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority33143.bound, LeftBound33139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33143.bound, LeftBound33139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority33143.actual selector witness, LeftBound33139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33147

namespace LeftBound33151
def owner : Owner := ⟨.program ⟨214⟩, ⟨28767⟩⟩
def transferEvent : Nat := 33151
def frameStart : Nat := 33074
def rule : BoundRule := .product (.predecessor 0 33149 .coefficient) (.predecessor 1 33150 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33149 .coefficient)
      LeftBound33147.bound (LeftBound33147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33150 .coefficient)
      LeftAuthority33124.bound (LeftAuthority33124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33124.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33147.bound LeftAuthority33124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33147.bound, LeftAuthority33124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33147.actual selector witness) * (LeftAuthority33124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33151

namespace LeftBound33162
def owner : Owner := ⟨.program ⟨214⟩, ⟨18887⟩⟩
def transferEvent : Nat := 33162
def frameStart : Nat := 33074
def rule : BoundRule := .product (.predecessor 0 33160 .coefficient) (.predecessor 1 33161 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33160 .coefficient)
      LeftAuthority33135.bound (LeftAuthority33135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33135.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33161 .coefficient)
      LeftAuthority33158.bound (LeftAuthority33158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33158.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33158.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority33135.bound LeftAuthority33158.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33135.bound, LeftAuthority33158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority33135.actual selector witness) * (LeftAuthority33158.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33162

namespace LeftBound33170
def owner : Owner := ⟨.program ⟨214⟩, ⟨18892⟩⟩
def transferEvent : Nat := 33170
def frameStart : Nat := 33074
def rule : BoundRule := .sum [.predecessor 0 33168 .coefficient, .predecessor 1 33169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33168 .coefficient)
      LeftAuthority33166.bound (LeftAuthority33166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33169 .coefficient)
      LeftBound33162.bound (LeftBound33162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33162.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority33166.bound, LeftBound33162.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33166.bound, LeftBound33162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority33166.actual selector witness, LeftBound33162.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33170

namespace LeftBound33174
def owner : Owner := ⟨.program ⟨214⟩, ⟨28772⟩⟩
def transferEvent : Nat := 33174
def frameStart : Nat := 33074
def rule : BoundRule := .sum [.predecessor 0 33172 .coefficient, .predecessor 1 33173 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33172 .coefficient)
      LeftBound33170.bound (LeftBound33170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33173 .coefficient)
      LeftBound33151.bound (LeftBound33151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33170.bound, LeftBound33151.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33170.bound, LeftBound33151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33170.actual selector witness, LeftBound33151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33174

namespace LeftBound33187
def owner : Owner := ⟨.program ⟨214⟩, ⟨28769⟩⟩
def transferEvent : Nat := 33187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 33185 .coefficient, .predecessor 1 33186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33185 .coefficient)
      LeftBound33016.bound (LeftBound33016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33186 .coefficient)
      LeftBound32999.bound (LeftBound32999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact33006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33016.bound, LeftBound32999.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33016.bound, LeftBound32999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33016.actual selector witness, LeftBound32999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33187

namespace LeftBound33190
def owner : Owner := ⟨.program ⟨214⟩, ⟨28769⟩⟩
def transferEvent : Nat := 33190
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 33184 .summary, .result 33006 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33184 .summary)
      LeftBound33018.bound (LeftBound33018.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21919⟩⟩) (rawTerms := some (Proof.Events129.exact33184RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33006 .summary)
      LeftBound33001.bound (LeftBound33001.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28768⟩⟩) (rawTerms := some (Proof.Events128.exact33006RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33001.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33018.bound, LeftBound33001.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33018.bound, LeftBound33001.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33018.actual selector witness, LeftBound33001.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33190

namespace LeftBound33194
def owner : Owner := ⟨.program ⟨214⟩, ⟨28770⟩⟩
def transferEvent : Nat := 33194
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33192 .coefficient) (.predecessor 1 33193 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33192 .coefficient)
      LeftBound33187.bound (LeftBound33187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33193 .coefficient)
      LeftBound5638.bound (LeftBound5638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33187.bound LeftBound5638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33187.bound, LeftBound5638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33187.actual selector witness) * (LeftBound5638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33194

namespace LeftBound33195
def owner : Owner := ⟨.program ⟨214⟩, ⟨28770⟩⟩
def transferEvent : Nat := 33195
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩ [⟨.result 5635 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5635 .coefficient)
      LeftAuthority5634.bound (LeftAuthority5634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6673⟩⟩) (rawTerms := some (Proof.Events022.exact5635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5634.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5634.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33195

namespace LeftBound33196
def owner : Owner := ⟨.program ⟨214⟩, ⟨28770⟩⟩
def transferEvent : Nat := 33196
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 33191 .summary) (.transfer 33195) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33191 .summary)
      LeftBound33190.bound (LeftBound33190.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28769⟩⟩) (rawTerms := some (Proof.Events129.exact33191RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33195)
      LeftBound33195.bound (LeftBound33195.actual selector witness) := by
  exact .transfer (LeftBound33195.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33190.bound LeftBound33195.bound
def bound : CoeffClass := .finite ⟨4742652258740286904787271680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33190.bound, LeftBound33195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33190.actual selector witness) * (LeftBound33195.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33196

namespace LeftBound33211
def owner : Owner := ⟨.program ⟨214⟩, ⟨28551⟩⟩
def transferEvent : Nat := 33211
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33209 .coefficient) (.predecessor 1 33210 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33209 .coefficient)
      LeftBound25068.bound (LeftBound25068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33210 .coefficient)
      LeftAuthority33207.bound (LeftAuthority33207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33207.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33207.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25068.bound LeftAuthority33207.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25068.bound, LeftAuthority33207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25068.actual selector witness) * (LeftAuthority33207.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33211

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
