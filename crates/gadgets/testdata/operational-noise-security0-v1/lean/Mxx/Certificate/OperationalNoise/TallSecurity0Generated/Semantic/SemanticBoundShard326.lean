import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard290
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard325

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound49260
def owner : Owner := ⟨.program ⟨214⟩, ⟨27235⟩⟩
def transferEvent : Nat := 49260
def frameStart : Nat := 49183
def rule : BoundRule := .product (.predecessor 0 49258 .coefficient) (.predecessor 1 49259 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49258 .coefficient)
      LeftBound49256.bound (LeftBound49256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49259 .coefficient)
      LeftAuthority49233.bound (LeftAuthority49233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49233.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49256.bound LeftAuthority49233.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49256.bound, LeftAuthority49233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49256.actual selector witness) * (LeftAuthority49233.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49260

namespace LeftBound49271
def owner : Owner := ⟨.program ⟨214⟩, ⟨17836⟩⟩
def transferEvent : Nat := 49271
def frameStart : Nat := 49183
def rule : BoundRule := .product (.predecessor 0 49269 .coefficient) (.predecessor 1 49270 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49269 .coefficient)
      LeftAuthority49244.bound (LeftAuthority49244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49244.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49270 .coefficient)
      LeftAuthority49267.bound (LeftAuthority49267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49267.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49267.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority49244.bound LeftAuthority49267.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49244.bound, LeftAuthority49267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority49244.actual selector witness) * (LeftAuthority49267.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49271

namespace LeftBound49279
def owner : Owner := ⟨.program ⟨214⟩, ⟨17837⟩⟩
def transferEvent : Nat := 49279
def frameStart : Nat := 49183
def rule : BoundRule := .sum [.predecessor 0 49277 .coefficient, .predecessor 1 49278 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49277 .coefficient)
      LeftAuthority49275.bound (LeftAuthority49275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49275.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49275.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49278 .coefficient)
      LeftBound49271.bound (LeftBound49271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49271.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49275.bound, LeftBound49271.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49275.bound, LeftBound49271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49275.actual selector witness, LeftBound49271.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49279

namespace LeftBound49283
def owner : Owner := ⟨.program ⟨214⟩, ⟨27240⟩⟩
def transferEvent : Nat := 49283
def frameStart : Nat := 49183
def rule : BoundRule := .sum [.predecessor 0 49281 .coefficient, .predecessor 1 49282 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49281 .coefficient)
      LeftBound49279.bound (LeftBound49279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49282 .coefficient)
      LeftBound49260.bound (LeftBound49260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49260.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49279.bound, LeftBound49260.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49279.bound, LeftBound49260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49279.actual selector witness, LeftBound49260.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49283

namespace LeftBound49296
def owner : Owner := ⟨.program ⟨214⟩, ⟨27237⟩⟩
def transferEvent : Nat := 49296
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 49294 .coefficient, .predecessor 1 49295 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49294 .coefficient)
      LeftBound49125.bound (LeftBound49125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49295 .coefficient)
      LeftBound49108.bound (LeftBound49108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49108.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49125.bound, LeftBound49108.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49125.bound, LeftBound49108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49125.actual selector witness, LeftBound49108.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49296

namespace LeftBound49299
def owner : Owner := ⟨.program ⟨214⟩, ⟨27237⟩⟩
def transferEvent : Nat := 49299
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 49293 .summary, .result 49115 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49293 .summary)
      LeftBound49127.bound (LeftBound49127.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20907⟩⟩) (rawTerms := some (Proof.Events192.exact49293RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49115 .summary)
      LeftBound49110.bound (LeftBound49110.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27236⟩⟩) (rawTerms := some (Proof.Events191.exact49115RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49127.bound, LeftBound49110.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49127.bound, LeftBound49110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49127.actual selector witness, LeftBound49110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49299

namespace LeftBound49303
def owner : Owner := ⟨.program ⟨214⟩, ⟨27238⟩⟩
def transferEvent : Nat := 49303
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49301 .coefficient) (.predecessor 1 49302 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49301 .coefficient)
      LeftBound49296.bound (LeftBound49296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49296.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49302 .coefficient)
      LeftBound5778.bound (LeftBound5778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49296.bound LeftBound5778.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49296.bound, LeftBound5778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49296.actual selector witness) * (LeftBound5778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49303

namespace LeftBound49304
def owner : Owner := ⟨.program ⟨214⟩, ⟨27238⟩⟩
def transferEvent : Nat := 49304
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩ [⟨.result 5775 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5775 .coefficient)
      LeftAuthority5774.bound (LeftAuthority5774.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6649⟩⟩) (rawTerms := some (Proof.Events022.exact5775RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5774.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5774.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5774.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49304

namespace LeftBound49305
def owner : Owner := ⟨.program ⟨214⟩, ⟨27238⟩⟩
def transferEvent : Nat := 49305
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 49300 .summary) (.transfer 49304) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49300 .summary)
      LeftBound49299.bound (LeftBound49299.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27237⟩⟩) (rawTerms := some (Proof.Events192.exact49300RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49304)
      LeftBound49304.bound (LeftBound49304.actual selector witness) := by
  exact .transfer (LeftBound49304.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49299.bound LeftBound49304.bound
def bound : CoeffClass := .finite ⟨4741582956326566183208747008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49299.bound, LeftBound49304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49299.actual selector witness) * (LeftBound49304.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49305

namespace LeftBound49320
def owner : Owner := ⟨.program ⟨214⟩, ⟨27019⟩⟩
def transferEvent : Nat := 49320
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49318 .coefficient) (.predecessor 1 49319 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49318 .coefficient)
      LeftBound43067.bound (LeftBound43067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49319 .coefficient)
      LeftAuthority49316.bound (LeftAuthority49316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49316.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49316.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43067.bound LeftAuthority49316.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43067.bound, LeftAuthority49316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43067.actual selector witness) * (LeftAuthority49316.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49320

namespace LeftBound49321
def owner : Owner := ⟨.program ⟨214⟩, ⟨27019⟩⟩
def transferEvent : Nat := 49321
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27017⟩⟩]⟩ [⟨.result 49317 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49317 .coefficient)
      LeftAuthority49316.bound (LeftAuthority49316.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27017⟩⟩) (rawTerms := some (Proof.Events192.exact49317RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49316.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49316.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority49316.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49316.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49321

namespace LeftBound49322
def owner : Owner := ⟨.program ⟨214⟩, ⟨27019⟩⟩
def transferEvent : Nat := 49322
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43071 .summary) (.transfer 49321) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43071 .summary)
      LeftBound43070.bound (LeftBound43070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25308⟩⟩) (rawTerms := some (Proof.Events168.exact43071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49321)
      LeftBound49321.bound (LeftBound49321.actual selector witness) := by
  exact .transfer (LeftBound49321.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43070.bound LeftBound49321.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43070.bound, LeftBound49321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43070.actual selector witness) * (LeftBound49321.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49322

namespace LeftBound49333
def owner : Owner := ⟨.program ⟨214⟩, ⟨20762⟩⟩
def transferEvent : Nat := 49333
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 49331 .coefficient) (.value (.predecessor 1 49332 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49331 .coefficient)
      LeftAuthority49329.bound (LeftAuthority49329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49329.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49329.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49332 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority49329.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49329.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49329.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound49333

namespace LeftBound49337
def owner : Owner := ⟨.program ⟨214⟩, ⟨20763⟩⟩
def transferEvent : Nat := 49337
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49335 .coefficient) (.predecessor 1 49336 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49335 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49336 .coefficient)
      LeftBound49333.bound (LeftBound49333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49333.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound49333.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound49333.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound49333.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49337

namespace LeftBound49338
def owner : Owner := ⟨.program ⟨214⟩, ⟨20763⟩⟩
def transferEvent : Nat := 49338
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20760⟩⟩]⟩ [⟨.result 49330 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49330 .coefficient)
      LeftAuthority49329.bound (LeftAuthority49329.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20760⟩⟩) (rawTerms := some (Proof.Events192.exact49330RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49329.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49329.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority49329.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49329.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49338

namespace LeftBound49339
def owner : Owner := ⟨.program ⟨214⟩, ⟨20763⟩⟩
def transferEvent : Nat := 49339
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 49338) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49338)
      LeftBound49338.bound (LeftBound49338.actual selector witness) := by
  exact .transfer (LeftBound49338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound49338.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound49338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound49338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49339

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
