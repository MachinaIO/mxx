import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard001
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard046
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard047

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound14947
def owner : Owner := ⟨.program ⟨257⟩, ⟨53957⟩⟩
def transferEvent : Nat := 14947
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14945 .coefficient, .predecessor 1 14946 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14945 .coefficient)
      LeftBound14943.bound (LeftBound14943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14946 .coefficient)
      LeftBound14882.bound (LeftBound14882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14943.bound, LeftBound14882.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14943.bound, LeftBound14882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14943.actual selector witness, LeftBound14882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14947

namespace LeftBound14951
def owner : Owner := ⟨.program ⟨257⟩, ⟨56937⟩⟩
def transferEvent : Nat := 14951
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14949 .coefficient, .predecessor 1 14950 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14949 .coefficient)
      LeftBound14947.bound (LeftBound14947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14950 .coefficient)
      LeftBound14874.bound (LeftBound14874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14947.bound, LeftBound14874.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14947.bound, LeftBound14874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14947.actual selector witness, LeftBound14874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14951

namespace LeftBound14955
def owner : Owner := ⟨.program ⟨257⟩, ⟨59917⟩⟩
def transferEvent : Nat := 14955
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14953 .coefficient, .predecessor 1 14954 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14953 .coefficient)
      LeftBound14951.bound (LeftBound14951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14954 .coefficient)
      LeftBound14866.bound (LeftBound14866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14951.bound, LeftBound14866.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14951.bound, LeftBound14866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14951.actual selector witness, LeftBound14866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14955

namespace LeftBound14959
def owner : Owner := ⟨.program ⟨257⟩, ⟨62897⟩⟩
def transferEvent : Nat := 14959
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14957 .coefficient, .predecessor 1 14958 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14957 .coefficient)
      LeftBound14955.bound (LeftBound14955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14955.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14958 .coefficient)
      LeftBound14858.bound (LeftBound14858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14858.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14955.bound, LeftBound14858.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14955.bound, LeftBound14858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14955.actual selector witness, LeftBound14858.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14959

namespace LeftBound14963
def owner : Owner := ⟨.program ⟨257⟩, ⟨65890⟩⟩
def transferEvent : Nat := 14963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14961 .coefficient, .predecessor 1 14962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14961 .coefficient)
      LeftBound14959.bound (LeftBound14959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14962 .coefficient)
      LeftBound14850.bound (LeftBound14850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14850.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14959.bound, LeftBound14850.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14959.bound, LeftBound14850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14959.actual selector witness, LeftBound14850.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14963

namespace LeftBound14967
def owner : Owner := ⟨.program ⟨257⟩, ⟨65891⟩⟩
def transferEvent : Nat := 14967
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14965 .coefficient, .predecessor 1 14966 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14965 .coefficient)
      LeftBound14963.bound (LeftBound14963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14966 .coefficient)
      LeftBound14842.bound (LeftBound14842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14963.bound, LeftBound14842.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14963.bound, LeftBound14842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14963.actual selector witness, LeftBound14842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14967

namespace LeftBound14971
def owner : Owner := ⟨.program ⟨257⟩, ⟨65892⟩⟩
def transferEvent : Nat := 14971
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14969 .coefficient, .predecessor 1 14970 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14969 .coefficient)
      LeftBound14967.bound (LeftBound14967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14970 .coefficient)
      LeftBound14834.bound (LeftBound14834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14967.bound, LeftBound14834.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14967.bound, LeftBound14834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14967.actual selector witness, LeftBound14834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14971

namespace LeftBound14975
def owner : Owner := ⟨.program ⟨257⟩, ⟨65893⟩⟩
def transferEvent : Nat := 14975
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14973 .coefficient, .predecessor 1 14974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14973 .coefficient)
      LeftBound14971.bound (LeftBound14971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14974 .coefficient)
      LeftBound14826.bound (LeftBound14826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14971.bound, LeftBound14826.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14971.bound, LeftBound14826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14971.actual selector witness, LeftBound14826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14975

namespace LeftBound14979
def owner : Owner := ⟨.program ⟨257⟩, ⟨65894⟩⟩
def transferEvent : Nat := 14979
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14977 .coefficient, .predecessor 1 14978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14977 .coefficient)
      LeftBound14975.bound (LeftBound14975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14978 .coefficient)
      LeftBound14818.bound (LeftBound14818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14818.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14975.bound, LeftBound14818.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14975.bound, LeftBound14818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14975.actual selector witness, LeftBound14818.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14979

namespace LeftBound14983
def owner : Owner := ⟨.program ⟨257⟩, ⟨65895⟩⟩
def transferEvent : Nat := 14983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14981 .coefficient, .predecessor 1 14982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14981 .coefficient)
      LeftBound14979.bound (LeftBound14979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14982 .coefficient)
      LeftBound14810.bound (LeftBound14810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14979.bound, LeftBound14810.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14979.bound, LeftBound14810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14979.actual selector witness, LeftBound14810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14983

namespace LeftBound14987
def owner : Owner := ⟨.program ⟨257⟩, ⟨65896⟩⟩
def transferEvent : Nat := 14987
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14985 .coefficient, .predecessor 1 14986 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14985 .coefficient)
      LeftBound14983.bound (LeftBound14983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14986 .coefficient)
      LeftBound14802.bound (LeftBound14802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14983.bound, LeftBound14802.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14983.bound, LeftBound14802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14983.actual selector witness, LeftBound14802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14987

namespace LeftBound14991
def owner : Owner := ⟨.program ⟨257⟩, ⟨65897⟩⟩
def transferEvent : Nat := 14991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14989 .coefficient, .predecessor 1 14990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14989 .coefficient)
      LeftBound14987.bound (LeftBound14987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14990 .coefficient)
      LeftBound14794.bound (LeftBound14794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14794.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14794.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14987.bound, LeftBound14794.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14987.bound, LeftBound14794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14987.actual selector witness, LeftBound14794.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14991

namespace LeftBound14995
def owner : Owner := ⟨.program ⟨257⟩, ⟨65898⟩⟩
def transferEvent : Nat := 14995
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14993 .coefficient, .predecessor 1 14994 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14993 .coefficient)
      LeftBound14991.bound (LeftBound14991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14994 .coefficient)
      LeftBound14786.bound (LeftBound14786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14786.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14991.bound, LeftBound14786.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14991.bound, LeftBound14786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14991.actual selector witness, LeftBound14786.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14995

namespace LeftBound14999
def owner : Owner := ⟨.program ⟨257⟩, ⟨67274⟩⟩
def transferEvent : Nat := 14999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14997 .coefficient, .predecessor 1 14998 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14997 .coefficient)
      LeftBound14995.bound (LeftBound14995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14998 .coefficient)
      LeftBound14778.bound (LeftBound14778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14778.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14995.bound, LeftBound14778.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14995.bound, LeftBound14778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14995.actual selector witness, LeftBound14778.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14999

namespace LeftBound15003
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def transferEvent : Nat := 15003
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 15001 .coefficient) (.predecessor 1 15002 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15001 .coefficient)
      LeftBound14999.bound (LeftBound14999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15002 .coefficient)
      LeftAuthority14286.bound (LeftAuthority14286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14286.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound14999.bound LeftAuthority14286.bound
def bound : CoeffClass := .finite ⟨317900092882589794259989485593113355416443120616549196878438212169531598890192664223658290155030721415179207832889017592392935849101459277691070679052143696720501865021934304884565896761239882203136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14999.bound, LeftAuthority14286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound14999.actual selector witness) * (LeftAuthority14286.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15003

namespace LeftBound15026
def owner : Owner := ⟨.program ⟨257⟩, ⟨67276⟩⟩
def transferEvent : Nat := 15026
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15024 .coefficient, .predecessor 1 15025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15024 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15025 .coefficient)
      LeftBound15003.bound (LeftBound15003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound15003.bound]
def bound : CoeffClass := .finite ⟨317900092882589794259989485593113355416443120616549196878438212169531598890192664223658290155030721415179207832889017592392935849101459277691070679052143696720501865021934304884565896761239882203138, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound15003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound726.actual selector witness, LeftBound15003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15026

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
