import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard024
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard025

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound8227
def owner : Owner := ⟨.program ⟨257⟩, ⟨51243⟩⟩
def transferEvent : Nat := 8227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8225 .coefficient, .predecessor 1 8226 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8225 .coefficient)
      LeftBound8223.bound (LeftBound8223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8226 .coefficient)
      LeftBound8174.bound (LeftBound8174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8223.bound, LeftBound8174.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8223.bound, LeftBound8174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8223.actual selector witness, LeftBound8174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8227

namespace LeftBound8231
def owner : Owner := ⟨.program ⟨257⟩, ⟨54223⟩⟩
def transferEvent : Nat := 8231
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8229 .coefficient, .predecessor 1 8230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8229 .coefficient)
      LeftBound8227.bound (LeftBound8227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8230 .coefficient)
      LeftBound8166.bound (LeftBound8166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8227.bound, LeftBound8166.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8227.bound, LeftBound8166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8227.actual selector witness, LeftBound8166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8231

namespace LeftBound8235
def owner : Owner := ⟨.program ⟨257⟩, ⟨57203⟩⟩
def transferEvent : Nat := 8235
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8233 .coefficient, .predecessor 1 8234 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8233 .coefficient)
      LeftBound8231.bound (LeftBound8231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8234 .coefficient)
      LeftBound8158.bound (LeftBound8158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8231.bound, LeftBound8158.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8231.bound, LeftBound8158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8231.actual selector witness, LeftBound8158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8235

namespace LeftBound8239
def owner : Owner := ⟨.program ⟨257⟩, ⟨60183⟩⟩
def transferEvent : Nat := 8239
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8237 .coefficient, .predecessor 1 8238 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8237 .coefficient)
      LeftBound8235.bound (LeftBound8235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8238 .coefficient)
      LeftBound8150.bound (LeftBound8150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8235.bound, LeftBound8150.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8235.bound, LeftBound8150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8235.actual selector witness, LeftBound8150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8239

namespace LeftBound8243
def owner : Owner := ⟨.program ⟨257⟩, ⟨63163⟩⟩
def transferEvent : Nat := 8243
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8241 .coefficient, .predecessor 1 8242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8241 .coefficient)
      LeftBound8239.bound (LeftBound8239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8242 .coefficient)
      LeftBound8142.bound (LeftBound8142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8239.bound, LeftBound8142.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8239.bound, LeftBound8142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8239.actual selector witness, LeftBound8142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8243

namespace LeftBound8247
def owner : Owner := ⟨.program ⟨257⟩, ⟨66870⟩⟩
def transferEvent : Nat := 8247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8245 .coefficient, .predecessor 1 8246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8245 .coefficient)
      LeftBound8243.bound (LeftBound8243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8246 .coefficient)
      LeftBound8134.bound (LeftBound8134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8243.bound, LeftBound8134.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8243.bound, LeftBound8134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8243.actual selector witness, LeftBound8134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8247

namespace LeftBound8251
def owner : Owner := ⟨.program ⟨257⟩, ⟨66871⟩⟩
def transferEvent : Nat := 8251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8249 .coefficient, .predecessor 1 8250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8249 .coefficient)
      LeftBound8247.bound (LeftBound8247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8250 .coefficient)
      LeftBound8126.bound (LeftBound8126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8247.bound, LeftBound8126.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8247.bound, LeftBound8126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8247.actual selector witness, LeftBound8126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8251

namespace LeftBound8255
def owner : Owner := ⟨.program ⟨257⟩, ⟨66872⟩⟩
def transferEvent : Nat := 8255
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8253 .coefficient, .predecessor 1 8254 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8253 .coefficient)
      LeftBound8251.bound (LeftBound8251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8254 .coefficient)
      LeftBound8118.bound (LeftBound8118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8251.bound, LeftBound8118.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8251.bound, LeftBound8118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8251.actual selector witness, LeftBound8118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8255

namespace LeftBound8259
def owner : Owner := ⟨.program ⟨257⟩, ⟨66873⟩⟩
def transferEvent : Nat := 8259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8257 .coefficient, .predecessor 1 8258 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8257 .coefficient)
      LeftBound8255.bound (LeftBound8255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8258 .coefficient)
      LeftBound8110.bound (LeftBound8110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8255.bound, LeftBound8110.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8255.bound, LeftBound8110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8255.actual selector witness, LeftBound8110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8259

namespace LeftBound8263
def owner : Owner := ⟨.program ⟨257⟩, ⟨66874⟩⟩
def transferEvent : Nat := 8263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8261 .coefficient, .predecessor 1 8262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8261 .coefficient)
      LeftBound8259.bound (LeftBound8259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8262 .coefficient)
      LeftBound8102.bound (LeftBound8102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8259.bound, LeftBound8102.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8259.bound, LeftBound8102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8259.actual selector witness, LeftBound8102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8263

namespace LeftBound8267
def owner : Owner := ⟨.program ⟨257⟩, ⟨66875⟩⟩
def transferEvent : Nat := 8267
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8265 .coefficient, .predecessor 1 8266 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8265 .coefficient)
      LeftBound8263.bound (LeftBound8263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8266 .coefficient)
      LeftBound8094.bound (LeftBound8094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8263.bound, LeftBound8094.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8263.bound, LeftBound8094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8263.actual selector witness, LeftBound8094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8267

namespace LeftBound8271
def owner : Owner := ⟨.program ⟨257⟩, ⟨66876⟩⟩
def transferEvent : Nat := 8271
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8269 .coefficient, .predecessor 1 8270 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8269 .coefficient)
      LeftBound8267.bound (LeftBound8267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8267.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8270 .coefficient)
      LeftBound8086.bound (LeftBound8086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8267.bound, LeftBound8086.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8267.bound, LeftBound8086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8267.actual selector witness, LeftBound8086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8271

namespace LeftBound8275
def owner : Owner := ⟨.program ⟨257⟩, ⟨66877⟩⟩
def transferEvent : Nat := 8275
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8273 .coefficient, .predecessor 1 8274 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8273 .coefficient)
      LeftBound8271.bound (LeftBound8271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8271.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8274 .coefficient)
      LeftBound8078.bound (LeftBound8078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8271.bound, LeftBound8078.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8271.bound, LeftBound8078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8271.actual selector witness, LeftBound8078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8275

namespace LeftBound8279
def owner : Owner := ⟨.program ⟨257⟩, ⟨66878⟩⟩
def transferEvent : Nat := 8279
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8277 .coefficient, .predecessor 1 8278 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8277 .coefficient)
      LeftBound8275.bound (LeftBound8275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8275.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8275.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8278 .coefficient)
      LeftBound8070.bound (LeftBound8070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8275.bound, LeftBound8070.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8275.bound, LeftBound8070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8275.actual selector witness, LeftBound8070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8279

namespace LeftBound8283
def owner : Owner := ⟨.program ⟨257⟩, ⟨67541⟩⟩
def transferEvent : Nat := 8283
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8281 .coefficient, .predecessor 1 8282 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8281 .coefficient)
      LeftBound8279.bound (LeftBound8279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8282 .coefficient)
      LeftBound8062.bound (LeftBound8062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8279.bound, LeftBound8062.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8279.bound, LeftBound8062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8279.actual selector witness, LeftBound8062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8283

namespace LeftBound8287
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def transferEvent : Nat := 8287
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8285 .coefficient) (.predecessor 1 8286 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8285 .coefficient)
      LeftBound8283.bound (LeftBound8283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8286 .coefficient)
      LeftAuthority7560.bound (LeftAuthority7560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7560.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7560.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound8283.bound LeftAuthority7560.bound
def bound : CoeffClass := .finite ⟨55627767500075853938083822181989862319971904251141117012615783461179150185478398050615437136164660776079340398378119543753777682305210719493116584432252873685578064948364640554851808692785016012800, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8283.bound, LeftAuthority7560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound8283.actual selector witness) * (LeftAuthority7560.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8287

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
