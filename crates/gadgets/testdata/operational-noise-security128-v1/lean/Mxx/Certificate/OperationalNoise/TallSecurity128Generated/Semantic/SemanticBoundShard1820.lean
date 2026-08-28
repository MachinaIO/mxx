import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1798
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1819

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound268998
def owner : Owner := ⟨.program ⟨257⟩, ⟨30509⟩⟩
def transferEvent : Nat := 268998
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 268996 .coefficient) (.predecessor 1 268997 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268996 .coefficient)
      LeftBound268992.bound (LeftBound268992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1050.exact268995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268997 .coefficient)
      LeftAuthority268930.bound (LeftAuthority268930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1050.exact268931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268930.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268930.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound268992.bound LeftAuthority268930.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268992.bound, LeftAuthority268930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound268992.actual selector witness) * (LeftAuthority268930.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound268998

namespace LeftBound268999
def owner : Owner := ⟨.program ⟨257⟩, ⟨30509⟩⟩
def transferEvent : Nat := 268999
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨30508⟩⟩]⟩ [⟨.result 268931 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 268931 .coefficient)
      LeftAuthority268930.bound (LeftAuthority268930.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨30508⟩⟩) (rawTerms := some (Proof.Events1050.exact268931RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268930.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268930.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority268930.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority268930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority268930.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound268999

namespace LeftBound269000
def owner : Owner := ⟨.program ⟨257⟩, ⟨30509⟩⟩
def transferEvent : Nat := 269000
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 268995 .summary) (.transfer 268999) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 268995 .summary)
      LeftBound268994.bound (LeftBound268994.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28581⟩⟩) (rawTerms := some (Proof.Events1050.exact268995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound268994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 268999)
      LeftBound268999.bound (LeftBound268999.actual selector witness) := by
  exact .transfer (LeftBound268999.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound268994.bound LeftBound268999.bound
def bound : CoeffClass := .finite ⟨2997925237700553605120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268994.bound, LeftBound268999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound268994.actual selector witness) * (LeftBound268999.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269000

namespace LeftBound269011
def owner : Owner := ⟨.program ⟨257⟩, ⟨29448⟩⟩
def transferEvent : Nat := 269011
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 269009 .coefficient) (.value (.predecessor 1 269010 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269009 .coefficient)
      LeftAuthority269007.bound (LeftAuthority269007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1050.exact269008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269007.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269010 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority269007.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269007.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority269007.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound269011

namespace LeftBound269015
def owner : Owner := ⟨.program ⟨257⟩, ⟨29449⟩⟩
def transferEvent : Nat := 269015
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 269013 .coefficient) (.predecessor 1 269014 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269013 .coefficient)
      LeftBound266117.bound (LeftBound266117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269014 .coefficient)
      LeftBound269011.bound (LeftBound269011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1050.exact269012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269011.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound266117.bound LeftBound269011.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound266117.bound, LeftBound269011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound266117.actual selector witness) * (LeftBound269011.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269015

namespace LeftBound269016
def owner : Owner := ⟨.program ⟨257⟩, ⟨29449⟩⟩
def transferEvent : Nat := 269016
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩ [⟨.result 269008 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 269008 .coefficient)
      LeftAuthority269007.bound (LeftAuthority269007.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨29446⟩⟩) (rawTerms := some (Proof.Events1050.exact269008RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269007.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269007.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority269007.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority269007.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound269016

namespace LeftBound269017
def owner : Owner := ⟨.program ⟨257⟩, ⟨29449⟩⟩
def transferEvent : Nat := 269017
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 266120 .summary) (.transfer 269016) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 266120 .summary)
      LeftBound266118.bound (LeftBound266118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5449⟩⟩) (rawTerms := some (Proof.Events1039.exact266120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound266118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 269016)
      LeftBound269016.bound (LeftBound269016.actual selector witness) := by
  exact .transfer (LeftBound269016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound266118.bound LeftBound269016.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound266118.bound, LeftBound269016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound266118.actual selector witness) * (LeftBound269016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269017

namespace LeftBound269096
def owner : Owner := ⟨.program ⟨257⟩, ⟨28575⟩⟩
def transferEvent : Nat := 269096
def frameStart : Nat := 269067
def rule : BoundRule := .product (.predecessor 0 269094 .coefficient) (.predecessor 1 269095 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269094 .coefficient)
      LeftAuthority269092.bound (LeftAuthority269092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269095 .coefficient)
      LeftAuthority269089.bound (LeftAuthority269089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269089.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority269092.bound LeftAuthority269089.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269092.bound, LeftAuthority269089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority269092.actual selector witness) * (LeftAuthority269089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269096

namespace LeftBound269100
def owner : Owner := ⟨.program ⟨257⟩, ⟨28576⟩⟩
def transferEvent : Nat := 269100
def frameStart : Nat := 269067
def rule : BoundRule := .identity (.predecessor 0 269099 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269099 .coefficient)
      LeftBound269096.bound (LeftBound269096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269096.derived selector witness)

def rawBound : CoeffClass := LeftBound269096.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound269096.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound269100

namespace LeftBound269117
def owner : Owner := ⟨.program ⟨257⟩, ⟨30334⟩⟩
def transferEvent : Nat := 269117
def frameStart : Nat := 269067
def rule : BoundRule := .sum [.predecessor 0 269115 .coefficient, .predecessor 1 269116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269115 .coefficient)
      LeftBound269100.bound (LeftBound269100.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound269100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269116 .coefficient)
      LeftAuthority269113.bound (LeftAuthority269113.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority269113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound269100.bound, LeftAuthority269113.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269100.bound, LeftAuthority269113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound269100.actual selector witness, LeftAuthority269113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269117

namespace LeftBound269120
def owner : Owner := ⟨.program ⟨257⟩, ⟨30335⟩⟩
def transferEvent : Nat := 269120
def frameStart : Nat := 269067
def rule : BoundRule := .identity (.predecessor 0 269119 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269119 .coefficient)
      LeftBound269117.bound (LeftBound269117.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound269117.derived selector witness)

def rawBound : CoeffClass := LeftBound269117.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound269117.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound269120

namespace LeftBound269126
def owner : Owner := ⟨.program ⟨257⟩, ⟨30336⟩⟩
def transferEvent : Nat := 269126
def frameStart : Nat := 269067
def rule : BoundRule := .product (.predecessor 0 269124 .coefficient) (.predecessor 1 269125 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269124 .coefficient)
      LeftAuthority269122.bound (LeftAuthority269122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269122.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269125 .coefficient)
      LeftBound269120.bound (LeftBound269120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority269122.bound LeftBound269120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269122.bound, LeftBound269120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority269122.actual selector witness) * (LeftBound269120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269126

namespace LeftBound269142
def owner : Owner := ⟨.program ⟨257⟩, ⟨9548⟩⟩
def transferEvent : Nat := 269142
def frameStart : Nat := 269067
def rule : BoundRule := .scale (.predecessor 0 269140 .coefficient) (.value (.predecessor 1 269141 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269140 .coefficient)
      LeftAuthority269138.bound (LeftAuthority269138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269141 .coefficient)
      LeftAuthority269129.bound (LeftAuthority269129.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority269129.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority269138.bound LeftAuthority269129.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269138.bound, LeftAuthority269129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority269138.actual selector witness) * (LeftAuthority269129.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound269142

namespace LeftBound269145
def owner : Owner := ⟨.program ⟨257⟩, ⟨7296⟩⟩
def transferEvent : Nat := 269145
def frameStart : Nat := 269067
def rule : BoundRule := .identity (.predecessor 0 269144 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269144 .coefficient)
      LeftAuthority269132.bound (LeftAuthority269132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269132.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269132.derived selector witness)

def rawBound : CoeffClass := LeftAuthority269132.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority269132.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound269145

namespace LeftBound269149
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def transferEvent : Nat := 269149
def frameStart : Nat := 269067
def rule : BoundRule := .product (.predecessor 0 269147 .coefficient) (.predecessor 1 269148 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269147 .coefficient)
      LeftBound269145.bound (LeftBound269145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269148 .coefficient)
      LeftBound269142.bound (LeftBound269142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269142.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound269145.bound LeftBound269142.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269145.bound, LeftBound269142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound269145.actual selector witness) * (LeftBound269142.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269149

namespace LeftBound269154
def owner : Owner := ⟨.program ⟨257⟩, ⟨30337⟩⟩
def transferEvent : Nat := 269154
def frameStart : Nat := 269067
def rule : BoundRule := .sum [.predecessor 0 269152 .coefficient, .predecessor 1 269153 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269152 .coefficient)
      LeftBound269149.bound (LeftBound269149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269153 .coefficient)
      LeftBound269126.bound (LeftBound269126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound269149.bound, LeftBound269126.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269149.bound, LeftBound269126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound269149.actual selector witness, LeftBound269126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269154

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
