import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard612
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard613
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard614
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard615
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard616
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard617
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard618
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard619
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard620
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard621
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard637

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound94083
def owner : Owner := ⟨.program ⟨214⟩, ⟨28732⟩⟩
def transferEvent : Nat := 94083
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94079 .summary, .result 91665 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94079 .summary)
      LeftBound94078.bound (LeftBound94078.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28515⟩⟩) (rawTerms := some (Proof.Events367.exact94079RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91665 .summary)
      LeftBound91660.bound (LeftBound91660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28731⟩⟩) (rawTerms := some (Proof.Events358.exact91665RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94078.bound, LeftBound91660.bound]
def bound : CoeffClass := .finite ⟨56901750985984920673341931572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94078.bound, LeftBound91660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94078.actual selector witness, LeftBound91660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94083

namespace LeftBound94087
def owner : Owner := ⟨.program ⟨214⟩, ⟨28949⟩⟩
def transferEvent : Nat := 94087
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94085 .coefficient, .predecessor 1 94086 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94085 .coefficient)
      LeftBound94082.bound (LeftBound94082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94082.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94086 .coefficient)
      LeftBound91446.bound (LeftBound91446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94082.bound, LeftBound91446.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94082.bound, LeftBound91446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94082.actual selector witness, LeftBound91446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94087

namespace LeftBound94088
def owner : Owner := ⟨.program ⟨214⟩, ⟨28949⟩⟩
def transferEvent : Nat := 94088
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94084 .summary, .result 91453 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94084 .summary)
      LeftBound94083.bound (LeftBound94083.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28732⟩⟩) (rawTerms := some (Proof.Events367.exact94084RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91453 .summary)
      LeftBound91448.bound (LeftBound91448.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28948⟩⟩) (rawTerms := some (Proof.Events357.exact91453RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91448.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94083.bound, LeftBound91448.bound]
def bound : CoeffClass := .finite ⟨61644567752788856919910514740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94083.bound, LeftBound91448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94083.actual selector witness, LeftBound91448.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94088

namespace LeftBound94092
def owner : Owner := ⟨.program ⟨214⟩, ⟨29166⟩⟩
def transferEvent : Nat := 94092
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94090 .coefficient, .predecessor 1 94091 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94090 .coefficient)
      LeftBound94087.bound (LeftBound94087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94091 .coefficient)
      LeftBound91234.bound (LeftBound91234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91234.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91234.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94087.bound, LeftBound91234.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94087.bound, LeftBound91234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94087.actual selector witness, LeftBound91234.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94092

namespace LeftBound94093
def owner : Owner := ⟨.program ⟨214⟩, ⟨29166⟩⟩
def transferEvent : Nat := 94093
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94089 .summary, .result 91241 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94089 .summary)
      LeftBound94088.bound (LeftBound94088.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28949⟩⟩) (rawTerms := some (Proof.Events367.exact94089RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94088.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91241 .summary)
      LeftBound91236.bound (LeftBound91236.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29165⟩⟩) (rawTerms := some (Proof.Events356.exact91241RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91236.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94088.bound, LeftBound91236.bound]
def bound : CoeffClass := .finite ⟨66387466773624617837369753652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94088.bound, LeftBound91236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94088.actual selector witness, LeftBound91236.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94093

namespace LeftBound94097
def owner : Owner := ⟨.program ⟨214⟩, ⟨29383⟩⟩
def transferEvent : Nat := 94097
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94095 .coefficient, .predecessor 1 94096 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94095 .coefficient)
      LeftBound94092.bound (LeftBound94092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94092.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94096 .coefficient)
      LeftBound91022.bound (LeftBound91022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91022.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94092.bound, LeftBound91022.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94092.bound, LeftBound91022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94092.actual selector witness, LeftBound91022.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94097

namespace LeftBound94098
def owner : Owner := ⟨.program ⟨214⟩, ⟨29383⟩⟩
def transferEvent : Nat := 94098
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94094 .summary, .result 91029 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94094 .summary)
      LeftBound94093.bound (LeftBound94093.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29166⟩⟩) (rawTerms := some (Proof.Events367.exact94094RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94093.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91029 .summary)
      LeftBound91024.bound (LeftBound91024.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29382⟩⟩) (rawTerms := some (Proof.Events355.exact91029RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91024.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94093.bound, LeftBound91024.bound]
def bound : CoeffClass := .finite ⟨71130530302524028096610304052, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94093.bound, LeftBound91024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94093.actual selector witness, LeftBound91024.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94098

namespace LeftBound94102
def owner : Owner := ⟨.program ⟨214⟩, ⟨29600⟩⟩
def transferEvent : Nat := 94102
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94100 .coefficient, .predecessor 1 94101 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94100 .coefficient)
      LeftBound94097.bound (LeftBound94097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94097.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94101 .coefficient)
      LeftBound90810.bound (LeftBound90810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94097.bound, LeftBound90810.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94097.bound, LeftBound90810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94097.actual selector witness, LeftBound90810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94102

namespace LeftBound94103
def owner : Owner := ⟨.program ⟨214⟩, ⟨29600⟩⟩
def transferEvent : Nat := 94103
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94099 .summary, .result 90817 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94099 .summary)
      LeftBound94098.bound (LeftBound94098.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29383⟩⟩) (rawTerms := some (Proof.Events367.exact94099RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90817 .summary)
      LeftBound90812.bound (LeftBound90812.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29599⟩⟩) (rawTerms := some (Proof.Events354.exact90817RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94098.bound, LeftBound90812.bound]
def bound : CoeffClass := .finite ⟨75873840593518912368522821684, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94098.bound, LeftBound90812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94098.actual selector witness, LeftBound90812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94103

namespace LeftBound94107
def owner : Owner := ⟨.program ⟨214⟩, ⟨29817⟩⟩
def transferEvent : Nat := 94107
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94105 .coefficient, .predecessor 1 94106 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94105 .coefficient)
      LeftBound94102.bound (LeftBound94102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94106 .coefficient)
      LeftBound90598.bound (LeftBound90598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94102.bound, LeftBound90598.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94102.bound, LeftBound90598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94102.actual selector witness, LeftBound90598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94107

namespace LeftBound94108
def owner : Owner := ⟨.program ⟨214⟩, ⟨29817⟩⟩
def transferEvent : Nat := 94108
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94104 .summary, .result 90605 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94104 .summary)
      LeftBound94103.bound (LeftBound94103.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29600⟩⟩) (rawTerms := some (Proof.Events367.exact94104RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90605 .summary)
      LeftBound90600.bound (LeftBound90600.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29816⟩⟩) (rawTerms := some (Proof.Events353.exact90605RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90600.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94103.bound, LeftBound90600.bound]
def bound : CoeffClass := .finite ⟨80617397646609270653107306548, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94103.bound, LeftBound90600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94103.actual selector witness, LeftBound90600.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94108

namespace LeftBound94112
def owner : Owner := ⟨.program ⟨214⟩, ⟨30114⟩⟩
def transferEvent : Nat := 94112
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94110 .coefficient, .predecessor 1 94111 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94110 .coefficient)
      LeftBound94107.bound (LeftBound94107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94111 .coefficient)
      LeftBound90386.bound (LeftBound90386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94107.bound, LeftBound90386.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94107.bound, LeftBound90386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94107.actual selector witness, LeftBound90386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94112

namespace LeftBound94113
def owner : Owner := ⟨.program ⟨214⟩, ⟨30114⟩⟩
def transferEvent : Nat := 94113
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94109 .summary, .result 90393 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94109 .summary)
      LeftBound94108.bound (LeftBound94108.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29817⟩⟩) (rawTerms := some (Proof.Events367.exact94109RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90393 .summary)
      LeftBound90388.bound (LeftBound90388.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30113⟩⟩) (rawTerms := some (Proof.Events353.exact90393RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90388.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94108.bound, LeftBound90388.bound]
def bound : CoeffClass := .finite ⟨85361036953731453608582447156, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94108.bound, LeftBound90388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94108.actual selector witness, LeftBound90388.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94113

namespace LeftBound94117
def owner : Owner := ⟨.program ⟨214⟩, ⟨30125⟩⟩
def transferEvent : Nat := 94117
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94115 .coefficient, .predecessor 1 94116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94115 .coefficient)
      LeftBound94112.bound (LeftBound94112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94116 .coefficient)
      LeftBound90174.bound (LeftBound90174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94112.bound, LeftBound90174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94112.bound, LeftBound90174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94112.actual selector witness, LeftBound90174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94117

namespace LeftBound94118
def owner : Owner := ⟨.program ⟨214⟩, ⟨30125⟩⟩
def transferEvent : Nat := 94118
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94114 .summary, .result 90181 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94114 .summary)
      LeftBound94113.bound (LeftBound94113.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30114⟩⟩) (rawTerms := some (Proof.Events367.exact94114RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90181 .summary)
      LeftBound90176.bound (LeftBound90176.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30123⟩⟩) (rawTerms := some (Proof.Events352.exact90181RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94113.bound, LeftBound90176.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789388161076, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94113.bound, LeftBound90176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94113.actual selector witness, LeftBound90176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94118

namespace LeftBound94124
def owner : Owner := ⟨.program ⟨214⟩, ⟨7094⟩⟩
def transferEvent : Nat := 94124
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94122 .coefficient) (.predecessor 1 94123 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94122 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94123 .coefficient)
      LeftAuthority6163.bound (LeftAuthority6163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6163.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6163.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftAuthority6163.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority6163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftAuthority6163.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94124

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
