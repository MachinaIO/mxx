import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard037

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound11955
def owner : Owner := ⟨.program ⟨257⟩, ⟨18825⟩⟩
def transferEvent : Nat := 11955
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11953 .coefficient, .predecessor 1 11954 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11953 .coefficient)
      LeftBound11951.bound (LeftBound11951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11954 .coefficient)
      LeftBound11938.bound (LeftBound11938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11938.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11951.bound, LeftBound11938.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11951.bound, LeftBound11938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11951.actual selector witness, LeftBound11938.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11955

namespace LeftBound11959
def owner : Owner := ⟨.program ⟨257⟩, ⟨22045⟩⟩
def transferEvent : Nat := 11959
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11957 .coefficient, .predecessor 1 11958 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11957 .coefficient)
      LeftBound11955.bound (LeftBound11955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11955.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11958 .coefficient)
      LeftBound11930.bound (LeftBound11930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11955.bound, LeftBound11930.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11955.bound, LeftBound11930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11955.actual selector witness, LeftBound11930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11959

namespace LeftBound11963
def owner : Owner := ⟨.program ⟨257⟩, ⟨32065⟩⟩
def transferEvent : Nat := 11963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11961 .coefficient, .predecessor 1 11962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11961 .coefficient)
      LeftBound11959.bound (LeftBound11959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11962 .coefficient)
      LeftBound11922.bound (LeftBound11922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11959.bound, LeftBound11922.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11959.bound, LeftBound11922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11959.actual selector witness, LeftBound11922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11963

namespace LeftBound11967
def owner : Owner := ⟨.program ⟨257⟩, ⟨51129⟩⟩
def transferEvent : Nat := 11967
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11965 .coefficient, .predecessor 1 11966 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11965 .coefficient)
      LeftBound11963.bound (LeftBound11963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11966 .coefficient)
      LeftBound11914.bound (LeftBound11914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11963.bound, LeftBound11914.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11963.bound, LeftBound11914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11963.actual selector witness, LeftBound11914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11967

namespace LeftBound11971
def owner : Owner := ⟨.program ⟨257⟩, ⟨54109⟩⟩
def transferEvent : Nat := 11971
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11969 .coefficient, .predecessor 1 11970 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11969 .coefficient)
      LeftBound11967.bound (LeftBound11967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11970 .coefficient)
      LeftBound11906.bound (LeftBound11906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11967.bound, LeftBound11906.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11967.bound, LeftBound11906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11967.actual selector witness, LeftBound11906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11971

namespace LeftBound11975
def owner : Owner := ⟨.program ⟨257⟩, ⟨57089⟩⟩
def transferEvent : Nat := 11975
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11973 .coefficient, .predecessor 1 11974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11973 .coefficient)
      LeftBound11971.bound (LeftBound11971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11974 .coefficient)
      LeftBound11898.bound (LeftBound11898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11971.bound, LeftBound11898.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11971.bound, LeftBound11898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11971.actual selector witness, LeftBound11898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11975

namespace LeftBound11979
def owner : Owner := ⟨.program ⟨257⟩, ⟨60069⟩⟩
def transferEvent : Nat := 11979
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11977 .coefficient, .predecessor 1 11978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11977 .coefficient)
      LeftBound11975.bound (LeftBound11975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11978 .coefficient)
      LeftBound11890.bound (LeftBound11890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11975.bound, LeftBound11890.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11975.bound, LeftBound11890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11975.actual selector witness, LeftBound11890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11979

namespace LeftBound11983
def owner : Owner := ⟨.program ⟨257⟩, ⟨63049⟩⟩
def transferEvent : Nat := 11983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11981 .coefficient, .predecessor 1 11982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11981 .coefficient)
      LeftBound11979.bound (LeftBound11979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11982 .coefficient)
      LeftBound11882.bound (LeftBound11882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11979.bound, LeftBound11882.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11979.bound, LeftBound11882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11979.actual selector witness, LeftBound11882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11983

namespace LeftBound11987
def owner : Owner := ⟨.program ⟨257⟩, ⟨66450⟩⟩
def transferEvent : Nat := 11987
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11985 .coefficient, .predecessor 1 11986 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11985 .coefficient)
      LeftBound11983.bound (LeftBound11983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11986 .coefficient)
      LeftBound11874.bound (LeftBound11874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11983.bound, LeftBound11874.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11983.bound, LeftBound11874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11983.actual selector witness, LeftBound11874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11987

namespace LeftBound11991
def owner : Owner := ⟨.program ⟨257⟩, ⟨66451⟩⟩
def transferEvent : Nat := 11991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11989 .coefficient, .predecessor 1 11990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11989 .coefficient)
      LeftBound11987.bound (LeftBound11987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11990 .coefficient)
      LeftBound11866.bound (LeftBound11866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11987.bound, LeftBound11866.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11987.bound, LeftBound11866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11987.actual selector witness, LeftBound11866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11991

namespace LeftBound11995
def owner : Owner := ⟨.program ⟨257⟩, ⟨66452⟩⟩
def transferEvent : Nat := 11995
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11993 .coefficient, .predecessor 1 11994 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11993 .coefficient)
      LeftBound11991.bound (LeftBound11991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11994 .coefficient)
      LeftBound11858.bound (LeftBound11858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11858.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11991.bound, LeftBound11858.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11991.bound, LeftBound11858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11991.actual selector witness, LeftBound11858.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11995

namespace LeftBound11999
def owner : Owner := ⟨.program ⟨257⟩, ⟨66453⟩⟩
def transferEvent : Nat := 11999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11997 .coefficient, .predecessor 1 11998 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11997 .coefficient)
      LeftBound11995.bound (LeftBound11995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11998 .coefficient)
      LeftBound11850.bound (LeftBound11850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11850.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11995.bound, LeftBound11850.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11995.bound, LeftBound11850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11995.actual selector witness, LeftBound11850.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11999

namespace LeftBound12003
def owner : Owner := ⟨.program ⟨257⟩, ⟨66454⟩⟩
def transferEvent : Nat := 12003
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12001 .coefficient, .predecessor 1 12002 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12001 .coefficient)
      LeftBound11999.bound (LeftBound11999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12002 .coefficient)
      LeftBound11842.bound (LeftBound11842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11999.bound, LeftBound11842.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11999.bound, LeftBound11842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11999.actual selector witness, LeftBound11842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12003

namespace LeftBound12007
def owner : Owner := ⟨.program ⟨257⟩, ⟨66455⟩⟩
def transferEvent : Nat := 12007
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12005 .coefficient, .predecessor 1 12006 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12005 .coefficient)
      LeftBound12003.bound (LeftBound12003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12006 .coefficient)
      LeftBound11834.bound (LeftBound11834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12003.bound, LeftBound11834.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12003.bound, LeftBound11834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12003.actual selector witness, LeftBound11834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12007

namespace LeftBound12011
def owner : Owner := ⟨.program ⟨257⟩, ⟨66456⟩⟩
def transferEvent : Nat := 12011
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12009 .coefficient, .predecessor 1 12010 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12009 .coefficient)
      LeftBound12007.bound (LeftBound12007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12010 .coefficient)
      LeftBound11826.bound (LeftBound11826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12007.bound, LeftBound11826.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12007.bound, LeftBound11826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12007.actual selector witness, LeftBound11826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12011

namespace LeftBound12015
def owner : Owner := ⟨.program ⟨257⟩, ⟨66457⟩⟩
def transferEvent : Nat := 12015
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12013 .coefficient, .predecessor 1 12014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12013 .coefficient)
      LeftBound12011.bound (LeftBound12011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12014 .coefficient)
      LeftBound11818.bound (LeftBound11818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11818.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12011.bound, LeftBound11818.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12011.bound, LeftBound11818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12011.actual selector witness, LeftBound11818.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12015

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
