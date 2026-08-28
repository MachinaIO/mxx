import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard558
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard616

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90961
def owner : Owner := ⟨.program ⟨214⟩, ⟨16709⟩⟩
def transferEvent : Nat := 90961
def frameStart : Nat := 90902
def rule : BoundRule := .identity (.predecessor 0 90960 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90960 .coefficient)
      LeftBound90958.bound (LeftBound90958.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound90958.derived selector witness)

def rawBound : CoeffClass := LeftBound90958.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound90958.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90961

namespace LeftBound90967
def owner : Owner := ⟨.program ⟨214⟩, ⟨16710⟩⟩
def transferEvent : Nat := 90967
def frameStart : Nat := 90902
def rule : BoundRule := .product (.predecessor 0 90965 .coefficient) (.predecessor 1 90966 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90965 .coefficient)
      LeftAuthority90963.bound (LeftAuthority90963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90966 .coefficient)
      LeftBound90961.bound (LeftBound90961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90961.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority90963.bound LeftBound90961.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90963.bound, LeftBound90961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority90963.actual selector witness) * (LeftBound90961.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90967

namespace LeftBound90975
def owner : Owner := ⟨.program ⟨214⟩, ⟨16711⟩⟩
def transferEvent : Nat := 90975
def frameStart : Nat := 90902
def rule : BoundRule := .sum [.predecessor 0 90973 .coefficient, .predecessor 1 90974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90973 .coefficient)
      LeftAuthority90971.bound (LeftAuthority90971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90971.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90974 .coefficient)
      LeftBound90967.bound (LeftBound90967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90971.bound, LeftBound90967.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90971.bound, LeftBound90967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority90971.actual selector witness, LeftBound90967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90975

namespace LeftBound90979
def owner : Owner := ⟨.program ⟨214⟩, ⟨29379⟩⟩
def transferEvent : Nat := 90979
def frameStart : Nat := 90902
def rule : BoundRule := .product (.predecessor 0 90977 .coefficient) (.predecessor 1 90978 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90977 .coefficient)
      LeftBound90975.bound (LeftBound90975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90978 .coefficient)
      LeftAuthority90952.bound (LeftAuthority90952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90952.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90975.bound LeftAuthority90952.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90975.bound, LeftAuthority90952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90975.actual selector witness) * (LeftAuthority90952.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90979

namespace LeftBound90990
def owner : Owner := ⟨.program ⟨214⟩, ⟨17720⟩⟩
def transferEvent : Nat := 90990
def frameStart : Nat := 90902
def rule : BoundRule := .product (.predecessor 0 90988 .coefficient) (.predecessor 1 90989 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90988 .coefficient)
      LeftAuthority90963.bound (LeftAuthority90963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90989 .coefficient)
      LeftAuthority90986.bound (LeftAuthority90986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority90963.bound LeftAuthority90986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90963.bound, LeftAuthority90986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority90963.actual selector witness) * (LeftAuthority90986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90990

namespace LeftBound90998
def owner : Owner := ⟨.program ⟨214⟩, ⟨17721⟩⟩
def transferEvent : Nat := 90998
def frameStart : Nat := 90902
def rule : BoundRule := .sum [.predecessor 0 90996 .coefficient, .predecessor 1 90997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90996 .coefficient)
      LeftAuthority90994.bound (LeftAuthority90994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90994.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90997 .coefficient)
      LeftBound90990.bound (LeftBound90990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90990.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90994.bound, LeftBound90990.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90994.bound, LeftBound90990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority90994.actual selector witness, LeftBound90990.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90998

namespace LeftBound91002
def owner : Owner := ⟨.program ⟨214⟩, ⟨29384⟩⟩
def transferEvent : Nat := 91002
def frameStart : Nat := 90902
def rule : BoundRule := .sum [.predecessor 0 91000 .coefficient, .predecessor 1 91001 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91000 .coefficient)
      LeftBound90998.bound (LeftBound90998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91001 .coefficient)
      LeftBound90979.bound (LeftBound90979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90998.bound, LeftBound90979.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90998.bound, LeftBound90979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90998.actual selector witness, LeftBound90979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91002

namespace LeftBound91015
def owner : Owner := ⟨.program ⟨214⟩, ⟨29381⟩⟩
def transferEvent : Nat := 91015
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91013 .coefficient, .predecessor 1 91014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91013 .coefficient)
      LeftBound90844.bound (LeftBound90844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90844.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91014 .coefficient)
      LeftBound90827.bound (LeftBound90827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90844.bound, LeftBound90827.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90844.bound, LeftBound90827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90844.actual selector witness, LeftBound90827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91015

namespace LeftBound91018
def owner : Owner := ⟨.program ⟨214⟩, ⟨29381⟩⟩
def transferEvent : Nat := 91018
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 91012 .summary, .result 90834 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91012 .summary)
      LeftBound90846.bound (LeftBound90846.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22339⟩⟩) (rawTerms := some (Proof.Events355.exact91012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90834 .summary)
      LeftBound90829.bound (LeftBound90829.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29380⟩⟩) (rawTerms := some (Proof.Events354.exact90834RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90829.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90846.bound, LeftBound90829.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90846.bound, LeftBound90829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90846.actual selector witness, LeftBound90829.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91018

namespace LeftBound91022
def owner : Owner := ⟨.program ⟨214⟩, ⟨29382⟩⟩
def transferEvent : Nat := 91022
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91020 .coefficient) (.predecessor 1 91021 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91020 .coefficient)
      LeftBound91015.bound (LeftBound91015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91021 .coefficient)
      LeftBound5578.bound (LeftBound5578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5578.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91015.bound LeftBound5578.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91015.bound, LeftBound5578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91015.actual selector witness) * (LeftBound5578.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91022

namespace LeftBound91023
def owner : Owner := ⟨.program ⟨214⟩, ⟨29382⟩⟩
def transferEvent : Nat := 91023
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩ [⟨.result 5575 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5575 .coefficient)
      LeftAuthority5574.bound (LeftAuthority5574.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6665⟩⟩) (rawTerms := some (Proof.Events021.exact5575RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5574.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5574.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5574.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91023

namespace LeftBound91024
def owner : Owner := ⟨.program ⟨214⟩, ⟨29382⟩⟩
def transferEvent : Nat := 91024
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 91019 .summary) (.transfer 91023) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91019 .summary)
      LeftBound91018.bound (LeftBound91018.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29381⟩⟩) (rawTerms := some (Proof.Events355.exact91019RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91023)
      LeftBound91023.bound (LeftBound91023.actual selector witness) := by
  exact .transfer (LeftBound91023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91018.bound LeftBound91023.bound
def bound : CoeffClass := .finite ⟨4743063528899410259240550400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91018.bound, LeftBound91023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91018.actual selector witness) * (LeftBound91023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91024

namespace LeftBound91039
def owner : Owner := ⟨.program ⟨214⟩, ⟨29163⟩⟩
def transferEvent : Nat := 91039
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91037 .coefficient) (.predecessor 1 91038 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91037 .coefficient)
      LeftBound82112.bound (LeftBound82112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91038 .coefficient)
      LeftAuthority91035.bound (LeftAuthority91035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91035.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82112.bound LeftAuthority91035.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82112.bound, LeftAuthority91035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82112.actual selector witness) * (LeftAuthority91035.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91039

namespace LeftBound91040
def owner : Owner := ⟨.program ⟨214⟩, ⟨29163⟩⟩
def transferEvent : Nat := 91040
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29161⟩⟩]⟩ [⟨.result 91036 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91036 .coefficient)
      LeftAuthority91035.bound (LeftAuthority91035.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29161⟩⟩) (rawTerms := some (Proof.Events355.exact91036RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91035.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91035.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91035.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91040

namespace LeftBound91041
def owner : Owner := ⟨.program ⟨214⟩, ⟨29163⟩⟩
def transferEvent : Nat := 91041
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82116 .summary) (.transfer 91040) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82116 .summary)
      LeftBound82115.bound (LeftBound82115.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25452⟩⟩) (rawTerms := some (Proof.Events320.exact82116RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82115.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91040)
      LeftBound91040.bound (LeftBound91040.actual selector witness) := by
  exact .transfer (LeftBound91040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82115.bound LeftBound91040.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82115.bound, LeftBound91040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82115.actual selector witness) * (LeftBound91040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91041

namespace LeftBound91052
def owner : Owner := ⟨.program ⟨214⟩, ⟨22194⟩⟩
def transferEvent : Nat := 91052
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 91050 .coefficient) (.value (.predecessor 1 91051 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91050 .coefficient)
      LeftAuthority91048.bound (LeftAuthority91048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91051 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority91048.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91048.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91048.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91052

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
