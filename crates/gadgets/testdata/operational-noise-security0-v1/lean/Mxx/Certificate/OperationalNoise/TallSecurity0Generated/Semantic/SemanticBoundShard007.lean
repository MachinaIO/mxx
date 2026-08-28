import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard005
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard006

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound2275
def owner : Owner := ⟨.program ⟨214⟩, ⟨18866⟩⟩
def transferEvent : Nat := 2275
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2273 .coefficient, .predecessor 1 2274 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2273 .coefficient)
      LeftBound2271.bound (LeftBound2271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2271.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2274 .coefficient)
      LeftBound2126.bound (LeftBound2126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2271.bound, LeftBound2126.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2271.bound, LeftBound2126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound2271.actual selector witness, LeftBound2126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2275

namespace LeftBound2279
def owner : Owner := ⟨.program ⟨214⟩, ⟨18867⟩⟩
def transferEvent : Nat := 2279
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2277 .coefficient, .predecessor 1 2278 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2277 .coefficient)
      LeftBound2275.bound (LeftBound2275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2275.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2275.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2278 .coefficient)
      LeftBound2118.bound (LeftBound2118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2275.bound, LeftBound2118.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2275.bound, LeftBound2118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound2275.actual selector witness, LeftBound2118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2279

namespace LeftBound2283
def owner : Owner := ⟨.program ⟨214⟩, ⟨18868⟩⟩
def transferEvent : Nat := 2283
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2281 .coefficient, .predecessor 1 2282 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2281 .coefficient)
      LeftBound2279.bound (LeftBound2279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2282 .coefficient)
      LeftBound2110.bound (LeftBound2110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2279.bound, LeftBound2110.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2279.bound, LeftBound2110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound2279.actual selector witness, LeftBound2110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2283

namespace LeftBound2287
def owner : Owner := ⟨.program ⟨214⟩, ⟨18869⟩⟩
def transferEvent : Nat := 2287
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2285 .coefficient, .predecessor 1 2286 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2285 .coefficient)
      LeftBound2283.bound (LeftBound2283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2286 .coefficient)
      LeftBound2102.bound (LeftBound2102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2283.bound, LeftBound2102.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2283.bound, LeftBound2102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound2283.actual selector witness, LeftBound2102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2287

namespace LeftBound2291
def owner : Owner := ⟨.program ⟨214⟩, ⟨18870⟩⟩
def transferEvent : Nat := 2291
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2289 .coefficient, .predecessor 1 2290 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2289 .coefficient)
      LeftBound2287.bound (LeftBound2287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2290 .coefficient)
      LeftBound2094.bound (LeftBound2094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2287.bound, LeftBound2094.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2287.bound, LeftBound2094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound2287.actual selector witness, LeftBound2094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2291

namespace LeftBound2295
def owner : Owner := ⟨.program ⟨214⟩, ⟨18871⟩⟩
def transferEvent : Nat := 2295
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2293 .coefficient, .predecessor 1 2294 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2293 .coefficient)
      LeftBound2291.bound (LeftBound2291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2291.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2294 .coefficient)
      LeftBound2086.bound (LeftBound2086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2291.bound, LeftBound2086.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2291.bound, LeftBound2086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound2291.actual selector witness, LeftBound2086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2295

namespace LeftBound2299
def owner : Owner := ⟨.program ⟨214⟩, ⟨18873⟩⟩
def transferEvent : Nat := 2299
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2297 .coefficient, .predecessor 1 2298 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2297 .coefficient)
      LeftBound2295.bound (LeftBound2295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2298 .coefficient)
      LeftBound2078.bound (LeftBound2078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2295.bound, LeftBound2078.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2295.bound, LeftBound2078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound2295.actual selector witness, LeftBound2078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2299

namespace LeftBound2303
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def transferEvent : Nat := 2303
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2301 .coefficient) (.predecessor 1 2302 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2301 .coefficient)
      LeftBound2299.bound (LeftBound2299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events008.exact2300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2302 .coefficient)
      LeftAuthority1576.bound (LeftAuthority1576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1576.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound2299.bound LeftAuthority1576.bound
def bound : CoeffClass := .finite ⟨8330976608466984004323844944577072009791754305446445295429981584169234515484291556448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2299.bound, LeftAuthority1576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound2299.actual selector witness) * (LeftAuthority1576.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2303

namespace LeftBound2826
def owner : Owner := ⟨.program ⟨214⟩, ⟨18500⟩⟩
def transferEvent : Nat := 2826
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2824 .coefficient) (.predecessor 1 2825 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2824 .coefficient)
      LeftAuthority2822.bound (LeftAuthority2822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2825 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority2822.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2822.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority2822.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2826

namespace LeftBound2834
def owner : Owner := ⟨.program ⟨214⟩, ⟨18129⟩⟩
def transferEvent : Nat := 2834
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2832 .coefficient) (.predecessor 1 2833 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2832 .coefficient)
      LeftAuthority2830.bound (LeftAuthority2830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2830.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2833 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority2830.bound LeftAuthority542.bound
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2830.bound, LeftAuthority542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority2830.actual selector witness) * (LeftAuthority542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2834

namespace LeftBound2842
def owner : Owner := ⟨.program ⟨214⟩, ⟨16932⟩⟩
def transferEvent : Nat := 2842
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2840 .coefficient) (.predecessor 1 2841 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2840 .coefficient)
      LeftAuthority2838.bound (LeftAuthority2838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2841 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority2838.bound LeftAuthority552.bound
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2838.bound, LeftAuthority552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority2838.actual selector witness) * (LeftAuthority552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2842

namespace LeftBound2850
def owner : Owner := ⟨.program ⟨214⟩, ⟨17499⟩⟩
def transferEvent : Nat := 2850
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2848 .coefficient) (.predecessor 1 2849 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2848 .coefficient)
      LeftAuthority2846.bound (LeftAuthority2846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2849 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority2846.bound LeftAuthority562.bound
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2846.bound, LeftAuthority562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority2846.actual selector witness) * (LeftAuthority562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2850

namespace LeftBound2858
def owner : Owner := ⟨.program ⟨214⟩, ⟨17723⟩⟩
def transferEvent : Nat := 2858
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2856 .coefficient) (.predecessor 1 2857 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2856 .coefficient)
      LeftAuthority2854.bound (LeftAuthority2854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2857 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority2854.bound LeftAuthority572.bound
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2854.bound, LeftAuthority572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority2854.actual selector witness) * (LeftAuthority572.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2858

namespace LeftBound2866
def owner : Owner := ⟨.program ⟨214⟩, ⟨17954⟩⟩
def transferEvent : Nat := 2866
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2864 .coefficient) (.predecessor 1 2865 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2864 .coefficient)
      LeftAuthority2862.bound (LeftAuthority2862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2865 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority2862.bound LeftAuthority582.bound
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2862.bound, LeftAuthority582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority2862.actual selector witness) * (LeftAuthority582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2866

namespace LeftBound2874
def owner : Owner := ⟨.program ⟨214⟩, ⟨17555⟩⟩
def transferEvent : Nat := 2874
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2872 .coefficient) (.predecessor 1 2873 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2872 .coefficient)
      LeftAuthority2870.bound (LeftAuthority2870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2873 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority2870.bound LeftAuthority592.bound
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2870.bound, LeftAuthority592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority2870.actual selector witness) * (LeftAuthority592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2874

namespace LeftBound2882
def owner : Owner := ⟨.program ⟨214⟩, ⟨18849⟩⟩
def transferEvent : Nat := 2882
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 2880 .coefficient) (.predecessor 1 2881 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 2880 .coefficient)
      LeftAuthority2878.bound (LeftAuthority2878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2878.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 2881 .coefficient)
      LeftAuthority602.bound (LeftAuthority602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority602.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority2878.bound LeftAuthority602.bound
def bound : CoeffClass := .finite ⟨228236850212900051643120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2878.bound, LeftAuthority602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority2878.actual selector witness) * (LeftAuthority602.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound2882

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
