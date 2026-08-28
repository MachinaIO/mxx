import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard001
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard002
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard005
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard012
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard016

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound5267
def owner : Owner := ⟨.program ⟨214⟩, ⟨18799⟩⟩
def transferEvent : Nat := 5267
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5265 .coefficient, .predecessor 1 5266 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5265 .coefficient)
      LeftBound5263.bound (LeftBound5263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5266 .coefficient)
      LeftBound5070.bound (LeftBound5070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5263.bound, LeftBound5070.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5263.bound, LeftBound5070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5263.actual selector witness, LeftBound5070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5267

namespace LeftBound5271
def owner : Owner := ⟨.program ⟨214⟩, ⟨18800⟩⟩
def transferEvent : Nat := 5271
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5269 .coefficient, .predecessor 1 5270 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5269 .coefficient)
      LeftBound5267.bound (LeftBound5267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5267.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5270 .coefficient)
      LeftBound5062.bound (LeftBound5062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5267.bound, LeftBound5062.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5267.bound, LeftBound5062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5267.actual selector witness, LeftBound5062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5271

namespace LeftBound5275
def owner : Owner := ⟨.program ⟨214⟩, ⟨18802⟩⟩
def transferEvent : Nat := 5275
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5273 .coefficient, .predecessor 1 5274 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5273 .coefficient)
      LeftBound5271.bound (LeftBound5271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5271.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5274 .coefficient)
      LeftBound5054.bound (LeftBound5054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5271.bound, LeftBound5054.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5271.bound, LeftBound5054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5271.actual selector witness, LeftBound5054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5275

namespace LeftBound5279
def owner : Owner := ⟨.program ⟨214⟩, ⟨18803⟩⟩
def transferEvent : Nat := 5279
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5277 .coefficient) (.predecessor 1 5278 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5277 .coefficient)
      LeftBound5275.bound (LeftBound5275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5275.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5275.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5278 .coefficient)
      LeftAuthority4562.bound (LeftAuthority4562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound5275.bound LeftAuthority4562.bound
def bound : CoeffClass := .finite ⟨2777451680365593313469174690627642684510383604607272166840540148335299501594379777984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5275.bound, LeftAuthority4562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound5275.actual selector witness) * (LeftAuthority4562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5279

namespace LeftBound5302
def owner : Owner := ⟨.program ⟨214⟩, ⟨18804⟩⟩
def transferEvent : Nat := 5302
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5300 .coefficient, .predecessor 1 5301 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5300 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5301 .coefficient)
      LeftBound5279.bound (LeftBound5279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5279.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound5279.bound]
def bound : CoeffClass := .finite ⟨2777451680365593313469174690627642684510383604607272166840540148335299501594379777986, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound5279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound726.actual selector witness, LeftBound5279.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5302

namespace LeftBound5306
def owner : Owner := ⟨.program ⟨214⟩, ⟨18844⟩⟩
def transferEvent : Nat := 5306
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5304 .coefficient, .predecessor 1 5305 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5304 .coefficient)
      LeftBound5302.bound (LeftBound5302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5305 .coefficient)
      LeftBound4541.bound (LeftBound4541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5302.bound, LeftBound4541.bound]
def bound : CoeffClass := .finite ⟨6899444407929433029479313359221176367220926876881727069350441936216336588624622089698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5302.bound, LeftBound4541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5302.actual selector witness, LeftBound4541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5306

namespace LeftBound5310
def owner : Owner := ⟨.program ⟨214⟩, ⟨18845⟩⟩
def transferEvent : Nat := 5310
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5308 .coefficient, .predecessor 1 5309 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5308 .coefficient)
      LeftBound5306.bound (LeftBound5306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5309 .coefficient)
      LeftBound3799.bound (LeftBound3799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5306.bound, LeftBound3799.bound]
def bound : CoeffClass := .finite ⟨9327185996870120055146645113335820723240028443441670214362659866411009061559009440898, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5306.bound, LeftBound3799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5306.actual selector witness, LeftBound3799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5310

namespace LeftBound5314
def owner : Owner := ⟨.program ⟨214⟩, ⟨18860⟩⟩
def transferEvent : Nat := 5314
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5312 .coefficient, .predecessor 1 5313 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5312 .coefficient)
      LeftBound5310.bound (LeftBound5310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5313 .coefficient)
      LeftBound3051.bound (LeftBound3051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3051.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5310.bound, LeftBound3051.bound]
def bound : CoeffClass := .finite ⟨16040835534369575115870582776928934511632862732110380935057809879859171327254859926082, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5310.bound, LeftBound3051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5310.actual selector witness, LeftBound3051.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5314

namespace LeftBound5318
def owner : Owner := ⟨.program ⟨214⟩, ⟨18875⟩⟩
def transferEvent : Nat := 5318
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5316 .coefficient, .predecessor 1 5317 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5316 .coefficient)
      LeftBound5314.bound (LeftBound5314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5317 .coefficient)
      LeftBound2303.bound (LeftBound2303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2303.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2303.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5314.bound, LeftBound2303.bound]
def bound : CoeffClass := .finite ⟨24371812142836559120194427721506006521424617037556826230487791464028405842739151482530, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5314.bound, LeftBound2303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5314.actual selector witness, LeftBound2303.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5318

namespace LeftBound5322
def owner : Owner := ⟨.program ⟨214⟩, ⟨18890⟩⟩
def transferEvent : Nat := 5322
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5320 .coefficient, .predecessor 1 5321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5320 .coefficient)
      LeftBound5318.bound (LeftBound5318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5321 .coefficient)
      LeftBound1555.bound (LeftBound1555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5318.bound, LeftBound1555.bound]
def bound : CoeffClass := .finite ⟨26793426257238540784008943931846642262100253801848130548376554246360265852163604490114, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5318.bound, LeftBound1555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5318.actual selector witness, LeftBound1555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5322

namespace LeftBound5326
def owner : Owner := ⟨.program ⟨214⟩, ⟨18905⟩⟩
def transferEvent : Nat := 5326
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5324 .coefficient, .predecessor 1 5325 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5324 .coefficient)
      LeftBound5322.bound (LeftBound5322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5325 .coefficient)
      LeftBound807.bound (LeftBound807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound807.bound, RecordedBoundRefines] <;> decide)
      (LeftBound807.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5322.bound, LeftBound807.bound]
def bound : CoeffClass := .finite ⟨31369995936811926932431848593283855135596231783588826172212324271763083601361406445090, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5322.bound, LeftBound807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5322.actual selector witness, LeftBound807.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5326

namespace LeftBound5330
def owner : Owner := ⟨.program ⟨214⟩, ⟨18907⟩⟩
def transferEvent : Nat := 5330
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5328 .coefficient) (.predecessor 1 5329 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5328 .coefficient)
      LeftBound5326.bound (LeftBound5326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5329 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound5326.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5326.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound5326.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5330

namespace LeftBound5467
def owner : Owner := ⟨.program ⟨214⟩, ⟨6594⟩⟩
def transferEvent : Nat := 5467
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5465 .coefficient) (.predecessor 1 5466 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5465 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5466 .coefficient)
      LeftAuthority33.bound (LeftAuthority33.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact34RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1.bound LeftAuthority33.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1.bound, LeftAuthority33.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority1.actual selector witness) * (LeftAuthority33.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5467

namespace LeftBound5475
def owner : Owner := ⟨.program ⟨214⟩, ⟨6646⟩⟩
def transferEvent : Nat := 5475
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 5473 .coefficient) (.value (.predecessor 1 5474 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5473 .coefficient)
      LeftAuthority5471.bound (LeftAuthority5471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5471.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5474 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority5471.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5471.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5471.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound5475

namespace LeftBound5486
def owner : Owner := ⟨.program ⟨214⟩, ⟨7820⟩⟩
def transferEvent : Nat := 5486
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 5484 .coefficient) (.value (.predecessor 1 5485 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5484 .coefficient)
      LeftAuthority5482.bound (LeftAuthority5482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5485 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority5482.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5482.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5482.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound5486

namespace LeftBound5490
def owner : Owner := ⟨.program ⟨214⟩, ⟨6597⟩⟩
def transferEvent : Nat := 5490
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5488 .coefficient) (.predecessor 1 5489 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5488 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5489 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1.bound LeftAuthority35.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority1.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5490

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
