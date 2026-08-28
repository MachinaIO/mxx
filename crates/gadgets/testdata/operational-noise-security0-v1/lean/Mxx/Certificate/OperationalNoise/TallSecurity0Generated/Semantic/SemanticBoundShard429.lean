import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard395
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard428

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64133
def owner : Owner := ⟨.program ⟨214⟩, ⟨27007⟩⟩
def transferEvent : Nat := 64133
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64131 .coefficient, .predecessor 1 64132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64131 .coefficient)
      LeftBound63962.bound (LeftBound63962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64132 .coefficient)
      LeftBound63945.bound (LeftBound63945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63945.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63962.bound, LeftBound63945.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63962.bound, LeftBound63945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63962.actual selector witness, LeftBound63945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64133

namespace LeftBound64136
def owner : Owner := ⟨.program ⟨214⟩, ⟨27007⟩⟩
def transferEvent : Nat := 64136
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64130 .summary, .result 63952 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64130 .summary)
      LeftBound63964.bound (LeftBound63964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20759⟩⟩) (rawTerms := some (Proof.Events250.exact64130RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63952 .summary)
      LeftBound63947.bound (LeftBound63947.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27006⟩⟩) (rawTerms := some (Proof.Events249.exact63952RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63964.bound, LeftBound63947.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63964.bound, LeftBound63947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63964.actual selector witness, LeftBound63947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64136

namespace LeftBound64140
def owner : Owner := ⟨.program ⟨214⟩, ⟨27008⟩⟩
def transferEvent : Nat := 64140
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64138 .coefficient) (.predecessor 1 64139 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64138 .coefficient)
      LeftBound64133.bound (LeftBound64133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64139 .coefficient)
      LeftBound5798.bound (LeftBound5798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64133.bound LeftBound5798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64133.bound, LeftBound5798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64133.actual selector witness) * (LeftBound5798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64140

namespace LeftBound64141
def owner : Owner := ⟨.program ⟨214⟩, ⟨27008⟩⟩
def transferEvent : Nat := 64141
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩ [⟨.result 5795 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5795 .coefficient)
      LeftAuthority5794.bound (LeftAuthority5794.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6655⟩⟩) (rawTerms := some (Proof.Events022.exact5795RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5794.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5794.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5794.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64141

namespace LeftBound64142
def owner : Owner := ⟨.program ⟨214⟩, ⟨27008⟩⟩
def transferEvent : Nat := 64142
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64137 .summary) (.transfer 64141) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64137 .summary)
      LeftBound64136.bound (LeftBound64136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27007⟩⟩) (rawTerms := some (Proof.Events250.exact64137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64141)
      LeftBound64141.bound (LeftBound64141.actual selector witness) := by
  exact .transfer (LeftBound64141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64136.bound LeftBound64141.bound
def bound : CoeffClass := .finite ⟨4741418448262916841427435520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64136.bound, LeftBound64141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64136.actual selector witness) * (LeftBound64141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64142

namespace LeftBound64157
def owner : Owner := ⟨.program ⟨214⟩, ⟨26789⟩⟩
def transferEvent : Nat := 64157
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64155 .coefficient) (.predecessor 1 64156 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64155 .coefficient)
      LeftBound58174.bound (LeftBound58174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64156 .coefficient)
      LeftAuthority64153.bound (LeftAuthority64153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64153.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64153.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58174.bound LeftAuthority64153.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58174.bound, LeftAuthority64153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58174.actual selector witness) * (LeftAuthority64153.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64157

namespace LeftBound64158
def owner : Owner := ⟨.program ⟨214⟩, ⟨26789⟩⟩
def transferEvent : Nat := 64158
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26787⟩⟩]⟩ [⟨.result 64154 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64154 .coefficient)
      LeftAuthority64153.bound (LeftAuthority64153.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26787⟩⟩) (rawTerms := some (Proof.Events250.exact64154RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64153.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64153.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64153.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64153.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64158

namespace LeftBound64159
def owner : Owner := ⟨.program ⟨214⟩, ⟨26789⟩⟩
def transferEvent : Nat := 64159
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 58178 .summary) (.transfer 64158) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58178 .summary)
      LeftBound58177.bound (LeftBound58177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25072⟩⟩) (rawTerms := some (Proof.Events227.exact58178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64158)
      LeftBound64158.bound (LeftBound64158.actual selector witness) := by
  exact .transfer (LeftBound64158.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58177.bound LeftBound64158.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58177.bound, LeftBound64158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58177.actual selector witness) * (LeftBound64158.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64159

namespace LeftBound64170
def owner : Owner := ⟨.program ⟨214⟩, ⟨20614⟩⟩
def transferEvent : Nat := 64170
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 64168 .coefficient) (.value (.predecessor 1 64169 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64168 .coefficient)
      LeftAuthority64166.bound (LeftAuthority64166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64169 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority64166.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64166.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64166.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound64170

namespace LeftBound64174
def owner : Owner := ⟨.program ⟨214⟩, ⟨20615⟩⟩
def transferEvent : Nat := 64174
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64172 .coefficient) (.predecessor 1 64173 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64172 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64173 .coefficient)
      LeftBound64170.bound (LeftBound64170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64170.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound64170.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound64170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound64170.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64174

namespace LeftBound64175
def owner : Owner := ⟨.program ⟨214⟩, ⟨20615⟩⟩
def transferEvent : Nat := 64175
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20612⟩⟩]⟩ [⟨.result 64167 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64167 .coefficient)
      LeftAuthority64166.bound (LeftAuthority64166.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20612⟩⟩) (rawTerms := some (Proof.Events250.exact64167RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64166.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64166.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64166.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64175

namespace LeftBound64176
def owner : Owner := ⟨.program ⟨214⟩, ⟨20615⟩⟩
def transferEvent : Nat := 64176
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 64175) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64175)
      LeftBound64175.bound (LeftBound64175.actual selector witness) := by
  exact .transfer (LeftBound64175.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound64175.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound64175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound64175.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64176

namespace LeftBound64271
def owner : Owner := ⟨.program ⟨214⟩, ⟨15119⟩⟩
def transferEvent : Nat := 64271
def frameStart : Nat := 64232
def rule : BoundRule := .identity (.predecessor 0 64270 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64270 .coefficient)
      LeftAuthority64268.bound (LeftAuthority64268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64268.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64268.derived selector witness)

def rawBound : CoeffClass := LeftAuthority64268.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64268.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority64268.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64271

namespace LeftBound64288
def owner : Owner := ⟨.program ⟨214⟩, ⟨15158⟩⟩
def transferEvent : Nat := 64288
def frameStart : Nat := 64232
def rule : BoundRule := .sum [.predecessor 0 64286 .coefficient, .predecessor 1 64287 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64286 .coefficient)
      LeftBound64271.bound (LeftBound64271.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64271.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64287 .coefficient)
      LeftAuthority64284.bound (LeftAuthority64284.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority64284.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64271.bound, LeftAuthority64284.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64271.bound, LeftAuthority64284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64271.actual selector witness, LeftAuthority64284.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64288

namespace LeftBound64291
def owner : Owner := ⟨.program ⟨214⟩, ⟨15159⟩⟩
def transferEvent : Nat := 64291
def frameStart : Nat := 64232
def rule : BoundRule := .identity (.predecessor 0 64290 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64290 .coefficient)
      LeftBound64288.bound (LeftBound64288.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64288.derived selector witness)

def rawBound : CoeffClass := LeftBound64288.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound64288.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64291

namespace LeftBound64297
def owner : Owner := ⟨.program ⟨214⟩, ⟨15160⟩⟩
def transferEvent : Nat := 64297
def frameStart : Nat := 64232
def rule : BoundRule := .product (.predecessor 0 64295 .coefficient) (.predecessor 1 64296 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64295 .coefficient)
      LeftAuthority64293.bound (LeftAuthority64293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64296 .coefficient)
      LeftBound64291.bound (LeftBound64291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64291.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority64293.bound LeftBound64291.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64293.bound, LeftBound64291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority64293.actual selector witness) * (LeftBound64291.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64297

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
