import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard022
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard308
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard309
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard310
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard311
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard312
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard313
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard333

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50258
def owner : Owner := ⟨.program ⟨214⟩, ⟨29409⟩⟩
def transferEvent : Nat := 50258
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50256 .coefficient, .predecessor 1 50257 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50256 .coefficient)
      LeftBound50253.bound (LeftBound50253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50253.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50257 .coefficient)
      LeftBound47183.bound (LeftBound47183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47183.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47183.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50253.bound, LeftBound47183.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50253.bound, LeftBound47183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50253.actual selector witness, LeftBound47183.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50258

namespace LeftBound50259
def owner : Owner := ⟨.program ⟨214⟩, ⟨29409⟩⟩
def transferEvent : Nat := 50259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50255 .summary, .result 47190 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50255 .summary)
      LeftBound50254.bound (LeftBound50254.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29192⟩⟩) (rawTerms := some (Proof.Events196.exact50255RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47190 .summary)
      LeftBound47185.bound (LeftBound47185.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29408⟩⟩) (rawTerms := some (Proof.Events184.exact47190RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47185.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50254.bound, LeftBound47185.bound]
def bound : CoeffClass := .finite ⟨71130530302524028096610304052, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50254.bound, LeftBound47185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50254.actual selector witness, LeftBound47185.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50259

namespace LeftBound50263
def owner : Owner := ⟨.program ⟨214⟩, ⟨29626⟩⟩
def transferEvent : Nat := 50263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50261 .coefficient, .predecessor 1 50262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50261 .coefficient)
      LeftBound50258.bound (LeftBound50258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50262 .coefficient)
      LeftBound46971.bound (LeftBound46971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50258.bound, LeftBound46971.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50258.bound, LeftBound46971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50258.actual selector witness, LeftBound46971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50263

namespace LeftBound50264
def owner : Owner := ⟨.program ⟨214⟩, ⟨29626⟩⟩
def transferEvent : Nat := 50264
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50260 .summary, .result 46978 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50260 .summary)
      LeftBound50259.bound (LeftBound50259.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29409⟩⟩) (rawTerms := some (Proof.Events196.exact50260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46978 .summary)
      LeftBound46973.bound (LeftBound46973.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29625⟩⟩) (rawTerms := some (Proof.Events183.exact46978RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50259.bound, LeftBound46973.bound]
def bound : CoeffClass := .finite ⟨75873840593518912368522821684, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50259.bound, LeftBound46973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50259.actual selector witness, LeftBound46973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50264

namespace LeftBound50268
def owner : Owner := ⟨.program ⟨214⟩, ⟨29843⟩⟩
def transferEvent : Nat := 50268
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50266 .coefficient, .predecessor 1 50267 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50266 .coefficient)
      LeftBound50263.bound (LeftBound50263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50267 .coefficient)
      LeftBound46759.bound (LeftBound46759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46759.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50263.bound, LeftBound46759.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50263.bound, LeftBound46759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50263.actual selector witness, LeftBound46759.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50268

namespace LeftBound50269
def owner : Owner := ⟨.program ⟨214⟩, ⟨29843⟩⟩
def transferEvent : Nat := 50269
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50265 .summary, .result 46766 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50265 .summary)
      LeftBound50264.bound (LeftBound50264.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29626⟩⟩) (rawTerms := some (Proof.Events196.exact50265RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46766 .summary)
      LeftBound46761.bound (LeftBound46761.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29842⟩⟩) (rawTerms := some (Proof.Events182.exact46766RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46761.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50264.bound, LeftBound46761.bound]
def bound : CoeffClass := .finite ⟨80617397646609270653107306548, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50264.bound, LeftBound46761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50264.actual selector witness, LeftBound46761.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50269

namespace LeftBound50273
def owner : Owner := ⟨.program ⟨214⟩, ⟨30159⟩⟩
def transferEvent : Nat := 50273
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50271 .coefficient, .predecessor 1 50272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50271 .coefficient)
      LeftBound50268.bound (LeftBound50268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50272 .coefficient)
      LeftBound46547.bound (LeftBound46547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50268.bound, LeftBound46547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50268.bound, LeftBound46547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50268.actual selector witness, LeftBound46547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50273

namespace LeftBound50274
def owner : Owner := ⟨.program ⟨214⟩, ⟨30159⟩⟩
def transferEvent : Nat := 50274
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50270 .summary, .result 46554 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50270 .summary)
      LeftBound50269.bound (LeftBound50269.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29843⟩⟩) (rawTerms := some (Proof.Events196.exact50270RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46554 .summary)
      LeftBound46549.bound (LeftBound46549.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30158⟩⟩) (rawTerms := some (Proof.Events181.exact46554RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50269.bound, LeftBound46549.bound]
def bound : CoeffClass := .finite ⟨85361036953731453608582447156, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50269.bound, LeftBound46549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50269.actual selector witness, LeftBound46549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50274

namespace LeftBound50278
def owner : Owner := ⟨.program ⟨214⟩, ⟨30170⟩⟩
def transferEvent : Nat := 50278
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50276 .coefficient, .predecessor 1 50277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50276 .coefficient)
      LeftBound50273.bound (LeftBound50273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50277 .coefficient)
      LeftBound46335.bound (LeftBound46335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50273.bound, LeftBound46335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50273.bound, LeftBound46335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50273.actual selector witness, LeftBound46335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50278

namespace LeftBound50279
def owner : Owner := ⟨.program ⟨214⟩, ⟨30170⟩⟩
def transferEvent : Nat := 50279
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50275 .summary, .result 46342 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50275 .summary)
      LeftBound50274.bound (LeftBound50274.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30159⟩⟩) (rawTerms := some (Proof.Events196.exact50275RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46342 .summary)
      LeftBound46337.bound (LeftBound46337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30168⟩⟩) (rawTerms := some (Proof.Events181.exact46342RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46337.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50274.bound, LeftBound46337.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789388161076, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50274.bound, LeftBound46337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50274.actual selector witness, LeftBound46337.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50279

namespace LeftBound50285
def owner : Owner := ⟨.program ⟨214⟩, ⟨7091⟩⟩
def transferEvent : Nat := 50285
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50283 .coefficient) (.predecessor 1 50284 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50283 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50284 .coefficient)
      LeftAuthority6043.bound (LeftAuthority6043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6043.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6043.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftAuthority6043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority6043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftAuthority6043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50285

namespace LeftBound50290
def owner : Owner := ⟨.program ⟨214⟩, ⟨7723⟩⟩
def transferEvent : Nat := 50290
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50288 .coefficient, .predecessor 1 50289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50288 .coefficient)
      LeftBound50285.bound (LeftBound50285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50289 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50285.bound, LeftBound36043.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50285.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50285.actual selector witness, LeftBound36043.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50290

namespace LeftBound50294
def owner : Owner := ⟨.program ⟨214⟩, ⟨7724⟩⟩
def transferEvent : Nat := 50294
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50292 .coefficient, .predecessor 1 50293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50292 .coefficient)
      LeftBound50290.bound (LeftBound50290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50293 .coefficient)
      LeftAuthority50281.bound (LeftAuthority50281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50281.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50290.bound, LeftAuthority50281.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50290.bound, LeftAuthority50281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50290.actual selector witness, LeftAuthority50281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50294

namespace LeftBound50295
def owner : Owner := ⟨.program ⟨214⟩, ⟨7724⟩⟩
def transferEvent : Nat := 50295
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨70⟩⟩]⟩ [⟨.result 50282 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50282 .coefficient)
      LeftAuthority50281.bound (LeftAuthority50281.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨70⟩⟩) (rawTerms := some (Proof.Events196.exact50282RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50281.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50281.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority50281.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50281.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50295

namespace LeftBound50300
def owner : Owner := ⟨.program ⟨214⟩, ⟨7901⟩⟩
def transferEvent : Nat := 50300
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50298 .coefficient) (.predecessor 1 50299 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50298 .coefficient)
      LeftBound50294.bound (LeftBound50294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50299 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50294.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50294.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50294.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50300

namespace LeftBound50301
def owner : Owner := ⟨.program ⟨214⟩, ⟨7901⟩⟩
def transferEvent : Nat := 50301
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩ [⟨.result 5957 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5957 .coefficient)
      LeftAuthority5956.bound (LeftAuthority5956.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7885⟩⟩) (rawTerms := some (Proof.Events023.exact5957RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5956.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5956.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5956.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50301

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
