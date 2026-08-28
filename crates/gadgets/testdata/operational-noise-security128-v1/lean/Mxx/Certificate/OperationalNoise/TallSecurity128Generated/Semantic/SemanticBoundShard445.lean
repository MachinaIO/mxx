import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard444

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound71160
def owner : Owner := ⟨.program ⟨257⟩, ⟨51295⟩⟩
def transferEvent : Nat := 71160
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71158 .coefficient, .predecessor 1 71159 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71158 .coefficient)
      LeftBound71156.bound (LeftBound71156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71159 .coefficient)
      LeftAuthority71052.bound (LeftAuthority71052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71052.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71156.bound, LeftAuthority71052.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71156.bound, LeftAuthority71052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71156.actual selector witness, LeftAuthority71052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71160

namespace LeftBound71164
def owner : Owner := ⟨.program ⟨257⟩, ⟨54275⟩⟩
def transferEvent : Nat := 71164
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71162 .coefficient, .predecessor 1 71163 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71162 .coefficient)
      LeftBound71160.bound (LeftBound71160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71163 .coefficient)
      LeftAuthority71029.bound (LeftAuthority71029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71029.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71160.bound, LeftAuthority71029.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71160.bound, LeftAuthority71029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71160.actual selector witness, LeftAuthority71029.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71164

namespace LeftBound71168
def owner : Owner := ⟨.program ⟨257⟩, ⟨57255⟩⟩
def transferEvent : Nat := 71168
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71166 .coefficient, .predecessor 1 71167 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71166 .coefficient)
      LeftBound71164.bound (LeftBound71164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71167 .coefficient)
      LeftAuthority71006.bound (LeftAuthority71006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71164.bound, LeftAuthority71006.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71164.bound, LeftAuthority71006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71164.actual selector witness, LeftAuthority71006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71168

namespace LeftBound71172
def owner : Owner := ⟨.program ⟨257⟩, ⟨60235⟩⟩
def transferEvent : Nat := 71172
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71170 .coefficient, .predecessor 1 71171 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71170 .coefficient)
      LeftBound71168.bound (LeftBound71168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71168.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71171 .coefficient)
      LeftAuthority70983.bound (LeftAuthority70983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact70984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70983.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71168.bound, LeftAuthority70983.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71168.bound, LeftAuthority70983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71168.actual selector witness, LeftAuthority70983.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71172

namespace LeftBound71176
def owner : Owner := ⟨.program ⟨257⟩, ⟨63215⟩⟩
def transferEvent : Nat := 71176
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71174 .coefficient, .predecessor 1 71175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71174 .coefficient)
      LeftBound71172.bound (LeftBound71172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71175 .coefficient)
      LeftAuthority70960.bound (LeftAuthority70960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact70961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70960.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71172.bound, LeftAuthority70960.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71172.bound, LeftAuthority70960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71172.actual selector witness, LeftAuthority70960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71176

namespace LeftBound71180
def owner : Owner := ⟨.program ⟨257⟩, ⟨67092⟩⟩
def transferEvent : Nat := 71180
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71178 .coefficient, .predecessor 1 71179 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71178 .coefficient)
      LeftBound71176.bound (LeftBound71176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71179 .coefficient)
      LeftAuthority70937.bound (LeftAuthority70937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact70938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70937.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71176.bound, LeftAuthority70937.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71176.bound, LeftAuthority70937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71176.actual selector witness, LeftAuthority70937.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71180

namespace LeftBound71184
def owner : Owner := ⟨.program ⟨257⟩, ⟨67093⟩⟩
def transferEvent : Nat := 71184
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71182 .coefficient, .predecessor 1 71183 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71182 .coefficient)
      LeftBound71180.bound (LeftBound71180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71183 .coefficient)
      LeftAuthority70914.bound (LeftAuthority70914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact70915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71180.bound, LeftAuthority70914.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71180.bound, LeftAuthority70914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71180.actual selector witness, LeftAuthority70914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71184

namespace LeftBound71188
def owner : Owner := ⟨.program ⟨257⟩, ⟨67094⟩⟩
def transferEvent : Nat := 71188
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71186 .coefficient, .predecessor 1 71187 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71186 .coefficient)
      LeftBound71184.bound (LeftBound71184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71187 .coefficient)
      LeftAuthority70891.bound (LeftAuthority70891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70891.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71184.bound, LeftAuthority70891.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71184.bound, LeftAuthority70891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71184.actual selector witness, LeftAuthority70891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71188

namespace LeftBound71192
def owner : Owner := ⟨.program ⟨257⟩, ⟨67095⟩⟩
def transferEvent : Nat := 71192
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71190 .coefficient, .predecessor 1 71191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71190 .coefficient)
      LeftBound71188.bound (LeftBound71188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71191 .coefficient)
      LeftAuthority70868.bound (LeftAuthority70868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70868.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71188.bound, LeftAuthority70868.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71188.bound, LeftAuthority70868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71188.actual selector witness, LeftAuthority70868.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71192

namespace LeftBound71196
def owner : Owner := ⟨.program ⟨257⟩, ⟨67096⟩⟩
def transferEvent : Nat := 71196
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71194 .coefficient, .predecessor 1 71195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71194 .coefficient)
      LeftBound71192.bound (LeftBound71192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71195 .coefficient)
      LeftAuthority70845.bound (LeftAuthority70845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71192.bound, LeftAuthority70845.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71192.bound, LeftAuthority70845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71192.actual selector witness, LeftAuthority70845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71196

namespace LeftBound71200
def owner : Owner := ⟨.program ⟨257⟩, ⟨67097⟩⟩
def transferEvent : Nat := 71200
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71198 .coefficient, .predecessor 1 71199 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71198 .coefficient)
      LeftBound71196.bound (LeftBound71196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71199 .coefficient)
      LeftAuthority70822.bound (LeftAuthority70822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70822.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71196.bound, LeftAuthority70822.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71196.bound, LeftAuthority70822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71196.actual selector witness, LeftAuthority70822.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71200

namespace LeftBound71204
def owner : Owner := ⟨.program ⟨257⟩, ⟨67098⟩⟩
def transferEvent : Nat := 71204
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71202 .coefficient, .predecessor 1 71203 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71202 .coefficient)
      LeftBound71200.bound (LeftBound71200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71203 .coefficient)
      LeftAuthority70799.bound (LeftAuthority70799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71200.bound, LeftAuthority70799.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71200.bound, LeftAuthority70799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71200.actual selector witness, LeftAuthority70799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71204

namespace LeftBound71208
def owner : Owner := ⟨.program ⟨257⟩, ⟨67099⟩⟩
def transferEvent : Nat := 71208
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71206 .coefficient, .predecessor 1 71207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71206 .coefficient)
      LeftBound71204.bound (LeftBound71204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71207 .coefficient)
      LeftAuthority70776.bound (LeftAuthority70776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70776.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71204.bound, LeftAuthority70776.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71204.bound, LeftAuthority70776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71204.actual selector witness, LeftAuthority70776.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71208

namespace LeftBound71212
def owner : Owner := ⟨.program ⟨257⟩, ⟨67100⟩⟩
def transferEvent : Nat := 71212
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71210 .coefficient, .predecessor 1 71211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71210 .coefficient)
      LeftBound71208.bound (LeftBound71208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71211 .coefficient)
      LeftAuthority70753.bound (LeftAuthority70753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70753.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70753.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71208.bound, LeftAuthority70753.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71208.bound, LeftAuthority70753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71208.actual selector witness, LeftAuthority70753.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71212

namespace LeftBound71215
def owner : Owner := ⟨.program ⟨257⟩, ⟨67101⟩⟩
def transferEvent : Nat := 71215
def frameStart : Nat := 70711
def rule : BoundRule := .identity (.predecessor 0 71214 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71214 .coefficient)
      LeftBound71212.bound (LeftBound71212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71212.derived selector witness)

def rawBound : CoeffClass := LeftBound71212.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound71212.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71215

namespace LeftBound71232
def owner : Owner := ⟨.program ⟨257⟩, ⟨69115⟩⟩
def transferEvent : Nat := 71232
def frameStart : Nat := 70711
def rule : BoundRule := .sum [.predecessor 0 71230 .coefficient, .predecessor 1 71231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 71230 .coefficient)
      LeftBound71215.bound (LeftBound71215.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 71231 .coefficient)
      LeftAuthority71228.bound (LeftAuthority71228.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority71228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71215.bound, LeftAuthority71228.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71215.bound, LeftAuthority71228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound71215.actual selector witness, LeftAuthority71228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71232

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
