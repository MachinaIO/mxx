import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard985
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1056
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1057
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1058
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1059
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1060
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1061
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1062
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1064
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1081

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound163231
def owner : Owner := ⟨.program ⟨257⟩, ⟨69933⟩⟩
def transferEvent : Nat := 163231
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163229 .coefficient, .predecessor 1 163230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163229 .coefficient)
      LeftBound163226.bound (LeftBound163226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163230 .coefficient)
      LeftBound160590.bound (LeftBound160590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events627.exact160597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163226.bound, LeftBound160590.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163226.bound, LeftBound160590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163226.actual selector witness, LeftBound160590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163231

namespace LeftBound163232
def owner : Owner := ⟨.program ⟨257⟩, ⟨69933⟩⟩
def transferEvent : Nat := 163232
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163228 .summary, .result 160597 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163228 .summary)
      LeftBound163227.bound (LeftBound163227.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69932⟩⟩) (rawTerms := some (Proof.Events637.exact163228RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 160597 .summary)
      LeftBound160592.bound (LeftBound160592.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36552⟩⟩) (rawTerms := some (Proof.Events627.exact160597RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound160592.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163227.bound, LeftBound160592.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163227.bound, LeftBound160592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163227.actual selector witness, LeftBound160592.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163232

namespace LeftBound163236
def owner : Owner := ⟨.program ⟨257⟩, ⟨69934⟩⟩
def transferEvent : Nat := 163236
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163234 .coefficient, .predecessor 1 163235 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163234 .coefficient)
      LeftBound163231.bound (LeftBound163231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163235 .coefficient)
      LeftBound160378.bound (LeftBound160378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events626.exact160385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163231.bound, LeftBound160378.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163231.bound, LeftBound160378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163231.actual selector witness, LeftBound160378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163236

namespace LeftBound163237
def owner : Owner := ⟨.program ⟨257⟩, ⟨69934⟩⟩
def transferEvent : Nat := 163237
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163233 .summary, .result 160385 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163233 .summary)
      LeftBound163232.bound (LeftBound163232.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69933⟩⟩) (rawTerms := some (Proof.Events637.exact163233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 160385 .summary)
      LeftBound160380.bound (LeftBound160380.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39232⟩⟩) (rawTerms := some (Proof.Events626.exact160385RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound160380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163232.bound, LeftBound160380.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163232.bound, LeftBound160380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163232.actual selector witness, LeftBound160380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163237

namespace LeftBound163241
def owner : Owner := ⟨.program ⟨257⟩, ⟨69935⟩⟩
def transferEvent : Nat := 163241
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163239 .coefficient, .predecessor 1 163240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163239 .coefficient)
      LeftBound163236.bound (LeftBound163236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163240 .coefficient)
      LeftBound160166.bound (LeftBound160166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163236.bound, LeftBound160166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163236.bound, LeftBound160166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163236.actual selector witness, LeftBound160166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163241

namespace LeftBound163242
def owner : Owner := ⟨.program ⟨257⟩, ⟨69935⟩⟩
def transferEvent : Nat := 163242
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163238 .summary, .result 160173 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163238 .summary)
      LeftBound163237.bound (LeftBound163237.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69934⟩⟩) (rawTerms := some (Proof.Events637.exact163238RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 160173 .summary)
      LeftBound160168.bound (LeftBound160168.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41912⟩⟩) (rawTerms := some (Proof.Events625.exact160173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound160168.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163237.bound, LeftBound160168.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163237.bound, LeftBound160168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163237.actual selector witness, LeftBound160168.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163242

namespace LeftBound163246
def owner : Owner := ⟨.program ⟨257⟩, ⟨69936⟩⟩
def transferEvent : Nat := 163246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163244 .coefficient, .predecessor 1 163245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163244 .coefficient)
      LeftBound163241.bound (LeftBound163241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163245 .coefficient)
      LeftBound159954.bound (LeftBound159954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events624.exact159961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163241.bound, LeftBound159954.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163241.bound, LeftBound159954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163241.actual selector witness, LeftBound159954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163246

namespace LeftBound163247
def owner : Owner := ⟨.program ⟨257⟩, ⟨69936⟩⟩
def transferEvent : Nat := 163247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163243 .summary, .result 159961 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163243 .summary)
      LeftBound163242.bound (LeftBound163242.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69935⟩⟩) (rawTerms := some (Proof.Events637.exact163243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 159961 .summary)
      LeftBound159956.bound (LeftBound159956.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44592⟩⟩) (rawTerms := some (Proof.Events624.exact159961RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound159956.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163242.bound, LeftBound159956.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163242.bound, LeftBound159956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163242.actual selector witness, LeftBound159956.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163247

namespace LeftBound163251
def owner : Owner := ⟨.program ⟨257⟩, ⟨69937⟩⟩
def transferEvent : Nat := 163251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163249 .coefficient, .predecessor 1 163250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163249 .coefficient)
      LeftBound163246.bound (LeftBound163246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163250 .coefficient)
      LeftBound159742.bound (LeftBound159742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events624.exact159749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159742.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163246.bound, LeftBound159742.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163246.bound, LeftBound159742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163246.actual selector witness, LeftBound159742.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163251

namespace LeftBound163252
def owner : Owner := ⟨.program ⟨257⟩, ⟨69937⟩⟩
def transferEvent : Nat := 163252
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163248 .summary, .result 159749 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163248 .summary)
      LeftBound163247.bound (LeftBound163247.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69936⟩⟩) (rawTerms := some (Proof.Events637.exact163248RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 159749 .summary)
      LeftBound159744.bound (LeftBound159744.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47272⟩⟩) (rawTerms := some (Proof.Events624.exact159749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound159744.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163247.bound, LeftBound159744.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163247.bound, LeftBound159744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163247.actual selector witness, LeftBound159744.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163252

namespace LeftBound163256
def owner : Owner := ⟨.program ⟨257⟩, ⟨69938⟩⟩
def transferEvent : Nat := 163256
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163254 .coefficient, .predecessor 1 163255 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163254 .coefficient)
      LeftBound163251.bound (LeftBound163251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163255 .coefficient)
      LeftBound159530.bound (LeftBound159530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events623.exact159537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163251.bound, LeftBound159530.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163251.bound, LeftBound159530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163251.actual selector witness, LeftBound159530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163256

namespace LeftBound163257
def owner : Owner := ⟨.program ⟨257⟩, ⟨69938⟩⟩
def transferEvent : Nat := 163257
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163253 .summary, .result 159537 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163253 .summary)
      LeftBound163252.bound (LeftBound163252.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69937⟩⟩) (rawTerms := some (Proof.Events637.exact163253RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 159537 .summary)
      LeftBound159532.bound (LeftBound159532.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49952⟩⟩) (rawTerms := some (Proof.Events623.exact159537RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound159532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163252.bound, LeftBound159532.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163252.bound, LeftBound159532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163252.actual selector witness, LeftBound159532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163257

namespace LeftBound163261
def owner : Owner := ⟨.program ⟨257⟩, ⟨71148⟩⟩
def transferEvent : Nat := 163261
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163259 .coefficient, .predecessor 1 163260 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163259 .coefficient)
      LeftBound163256.bound (LeftBound163256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163260 .coefficient)
      LeftBound159318.bound (LeftBound159318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events622.exact159325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163256.bound, LeftBound159318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163256.bound, LeftBound159318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163256.actual selector witness, LeftBound159318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163261

namespace LeftBound163262
def owner : Owner := ⟨.program ⟨257⟩, ⟨71148⟩⟩
def transferEvent : Nat := 163262
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163258 .summary, .result 159325 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163258 .summary)
      LeftBound163257.bound (LeftBound163257.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69938⟩⟩) (rawTerms := some (Proof.Events637.exact163258RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 159325 .summary)
      LeftBound159320.bound (LeftBound159320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71146⟩⟩) (rawTerms := some (Proof.Events622.exact159325RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound159320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163257.bound, LeftBound159320.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163257.bound, LeftBound159320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163257.actual selector witness, LeftBound159320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163262

namespace LeftBound163268
def owner : Owner := ⟨.program ⟨257⟩, ⟨7410⟩⟩
def transferEvent : Nat := 163268
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 163266 .coefficient) (.predecessor 1 163267 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163266 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163267 .coefficient)
      LeftAuthority16346.bound (LeftAuthority16346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16346.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16346.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftAuthority16346.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority16346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftAuthority16346.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163268

namespace LeftBound163273
def owner : Owner := ⟨.program ⟨257⟩, ⟨9223⟩⟩
def transferEvent : Nat := 163273
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163271 .coefficient, .predecessor 1 163272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163271 .coefficient)
      LeftBound163268.bound (LeftBound163268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163272 .coefficient)
      LeftBound149026.bound (LeftBound149026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163268.bound, LeftBound149026.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163268.bound, LeftBound149026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163268.actual selector witness, LeftBound149026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163273

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
