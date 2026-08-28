import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard146
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard150
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard153
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard154
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard157
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard168
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard172
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard175
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard201

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound30121
def owner : Owner := ⟨.program ⟨214⟩, ⟨27909⟩⟩
def transferEvent : Nat := 30121
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30117 .summary, .result 26713 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30117 .summary)
      LeftBound30116.bound (LeftBound30116.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27692⟩⟩) (rawTerms := some (Proof.Events117.exact30117RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26713 .summary)
      LeftBound26712.bound (LeftBound26712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27908⟩⟩) (rawTerms := some (Proof.Events104.exact26713RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30116.bound, LeftBound26712.bound]
def bound : CoeffClass := .finite ⟨10335729737273439256576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30116.bound, LeftBound26712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30116.actual selector witness, LeftBound26712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30121

namespace LeftBound30125
def owner : Owner := ⟨.program ⟨214⟩, ⟨28126⟩⟩
def transferEvent : Nat := 30125
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30123 .coefficient, .predecessor 1 30124 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30123 .coefficient)
      LeftBound30120.bound (LeftBound30120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30120.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30124 .coefficient)
      LeftBound26227.bound (LeftBound26227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26227.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30120.bound, LeftBound26227.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30120.bound, LeftBound26227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30120.actual selector witness, LeftBound26227.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30125

namespace LeftBound30126
def owner : Owner := ⟨.program ⟨214⟩, ⟨28126⟩⟩
def transferEvent : Nat := 30126
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30122 .summary, .result 26231 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30122 .summary)
      LeftBound30121.bound (LeftBound30121.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27909⟩⟩) (rawTerms := some (Proof.Events117.exact30122RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26231 .summary)
      LeftBound26230.bound (LeftBound26230.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28125⟩⟩) (rawTerms := some (Proof.Events102.exact26231RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26230.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30121.bound, LeftBound26230.bound]
def bound : CoeffClass := .finite ⟨11627843036103066759168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30121.bound, LeftBound26230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30121.actual selector witness, LeftBound26230.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30126

namespace LeftBound30130
def owner : Owner := ⟨.program ⟨214⟩, ⟨28343⟩⟩
def transferEvent : Nat := 30130
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30128 .coefficient, .predecessor 1 30129 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30128 .coefficient)
      LeftBound30125.bound (LeftBound30125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30129 .coefficient)
      LeftBound25745.bound (LeftBound25745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25745.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30125.bound, LeftBound25745.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30125.bound, LeftBound25745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30125.actual selector witness, LeftBound25745.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30130

namespace LeftBound30131
def owner : Owner := ⟨.program ⟨214⟩, ⟨28343⟩⟩
def transferEvent : Nat := 30131
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30127 .summary, .result 25749 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30127 .summary)
      LeftBound30126.bound (LeftBound30126.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28126⟩⟩) (rawTerms := some (Proof.Events117.exact30127RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25749 .summary)
      LeftBound25748.bound (LeftBound25748.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28342⟩⟩) (rawTerms := some (Proof.Events100.exact25749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25748.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30126.bound, LeftBound25748.bound]
def bound : CoeffClass := .finite ⟨12920023572267756019712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30126.bound, LeftBound25748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30126.actual selector witness, LeftBound25748.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30131

namespace LeftBound30135
def owner : Owner := ⟨.program ⟨214⟩, ⟨28560⟩⟩
def transferEvent : Nat := 30135
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30133 .coefficient, .predecessor 1 30134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30133 .coefficient)
      LeftBound30130.bound (LeftBound30130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30134 .coefficient)
      LeftBound25263.bound (LeftBound25263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30130.bound, LeftBound25263.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30130.bound, LeftBound25263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30130.actual selector witness, LeftBound25263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30135

namespace LeftBound30136
def owner : Owner := ⟨.program ⟨214⟩, ⟨28560⟩⟩
def transferEvent : Nat := 30136
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30132 .summary, .result 25267 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30132 .summary)
      LeftBound30131.bound (LeftBound30131.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28343⟩⟩) (rawTerms := some (Proof.Events117.exact30132RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25267 .summary)
      LeftBound25266.bound (LeftBound25266.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28559⟩⟩) (rawTerms := some (Proof.Events098.exact25267RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30131.bound, LeftBound25266.bound]
def bound : CoeffClass := .finite ⟨14212226520877465866240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30131.bound, LeftBound25266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30131.actual selector witness, LeftBound25266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30136

namespace LeftBound30140
def owner : Owner := ⟨.program ⟨214⟩, ⟨28777⟩⟩
def transferEvent : Nat := 30140
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30138 .coefficient, .predecessor 1 30139 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30138 .coefficient)
      LeftBound30135.bound (LeftBound30135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30139 .coefficient)
      LeftBound24781.bound (LeftBound24781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24781.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30135.bound, LeftBound24781.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30135.bound, LeftBound24781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30135.actual selector witness, LeftBound24781.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30140

namespace LeftBound30141
def owner : Owner := ⟨.program ⟨214⟩, ⟨28777⟩⟩
def transferEvent : Nat := 30141
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30137 .summary, .result 24785 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30137 .summary)
      LeftBound30136.bound (LeftBound30136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28560⟩⟩) (rawTerms := some (Proof.Events117.exact30137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24785 .summary)
      LeftBound24784.bound (LeftBound24784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28776⟩⟩) (rawTerms := some (Proof.Events096.exact24785RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24784.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30136.bound, LeftBound24784.bound]
def bound : CoeffClass := .finite ⟨15504496706822237470720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30136.bound, LeftBound24784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30136.actual selector witness, LeftBound24784.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30141

namespace LeftBound30145
def owner : Owner := ⟨.program ⟨214⟩, ⟨28994⟩⟩
def transferEvent : Nat := 30145
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30143 .coefficient, .predecessor 1 30144 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30143 .coefficient)
      LeftBound30140.bound (LeftBound30140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30144 .coefficient)
      LeftBound24299.bound (LeftBound24299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30140.bound, LeftBound24299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30140.bound, LeftBound24299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30140.actual selector witness, LeftBound24299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30145

namespace LeftBound30146
def owner : Owner := ⟨.program ⟨214⟩, ⟨28994⟩⟩
def transferEvent : Nat := 30146
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30142 .summary, .result 24303 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30142 .summary)
      LeftBound30141.bound (LeftBound30141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28777⟩⟩) (rawTerms := some (Proof.Events117.exact30142RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24303 .summary)
      LeftBound24302.bound (LeftBound24302.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28993⟩⟩) (rawTerms := some (Proof.Events094.exact24303RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24302.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30141.bound, LeftBound24302.bound]
def bound : CoeffClass := .finite ⟨16796811717657050247168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30141.bound, LeftBound24302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30141.actual selector witness, LeftBound24302.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30146

namespace LeftBound30150
def owner : Owner := ⟨.program ⟨214⟩, ⟨29211⟩⟩
def transferEvent : Nat := 30150
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30148 .coefficient, .predecessor 1 30149 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30148 .coefficient)
      LeftBound30145.bound (LeftBound30145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30149 .coefficient)
      LeftBound23817.bound (LeftBound23817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23817.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30145.bound, LeftBound23817.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30145.bound, LeftBound23817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30145.actual selector witness, LeftBound23817.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30150

namespace LeftBound30151
def owner : Owner := ⟨.program ⟨214⟩, ⟨29211⟩⟩
def transferEvent : Nat := 30151
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30147 .summary, .result 23821 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30147 .summary)
      LeftBound30146.bound (LeftBound30146.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28994⟩⟩) (rawTerms := some (Proof.Events117.exact30147RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30146.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23821 .summary)
      LeftBound23820.bound (LeftBound23820.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29210⟩⟩) (rawTerms := some (Proof.Events093.exact23821RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23820.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30146.bound, LeftBound23820.bound]
def bound : CoeffClass := .finite ⟨18089149140936883609600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30146.bound, LeftBound23820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30146.actual selector witness, LeftBound23820.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30151

namespace LeftBound30155
def owner : Owner := ⟨.program ⟨214⟩, ⟨29428⟩⟩
def transferEvent : Nat := 30155
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30153 .coefficient, .predecessor 1 30154 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30153 .coefficient)
      LeftBound30150.bound (LeftBound30150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30150.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30154 .coefficient)
      LeftBound23335.bound (LeftBound23335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30150.bound, LeftBound23335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30150.bound, LeftBound23335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30150.actual selector witness, LeftBound23335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30155

namespace LeftBound30156
def owner : Owner := ⟨.program ⟨214⟩, ⟨29428⟩⟩
def transferEvent : Nat := 30156
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30152 .summary, .result 23339 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30152 .summary)
      LeftBound30151.bound (LeftBound30151.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29211⟩⟩) (rawTerms := some (Proof.Events117.exact30152RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23339 .summary)
      LeftBound23338.bound (LeftBound23338.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29427⟩⟩) (rawTerms := some (Proof.Events091.exact23339RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30151.bound, LeftBound23338.bound]
def bound : CoeffClass := .finite ⟨19381531389106758144000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30151.bound, LeftBound23338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30151.actual selector witness, LeftBound23338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30156

namespace LeftBound30160
def owner : Owner := ⟨.program ⟨214⟩, ⟨29645⟩⟩
def transferEvent : Nat := 30160
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30158 .coefficient, .predecessor 1 30159 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30158 .coefficient)
      LeftBound30155.bound (LeftBound30155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30159 .coefficient)
      LeftBound22853.bound (LeftBound22853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30155.bound, LeftBound22853.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30155.bound, LeftBound22853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30155.actual selector witness, LeftBound22853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30160

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
