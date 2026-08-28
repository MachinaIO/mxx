import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1882
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1883
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1884
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1886
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1887
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1888
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1890
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1891

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound280171
def owner : Owner := ⟨.program ⟨257⟩, ⟨17527⟩⟩
def transferEvent : Nat := 280171
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280169 .coefficient, .predecessor 1 280170 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280169 .coefficient)
      LeftBound280164.bound (LeftBound280164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280170 .coefficient)
      LeftBound280134.bound (LeftBound280134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280164.bound, LeftBound280134.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280164.bound, LeftBound280134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280164.actual selector witness, LeftBound280134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280171

namespace LeftBound280172
def owner : Owner := ⟨.program ⟨257⟩, ⟨17527⟩⟩
def transferEvent : Nat := 280172
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280168 .summary, .result 280141 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280168 .summary)
      LeftBound280167.bound (LeftBound280167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9452⟩⟩) (rawTerms := some (Proof.Events1094.exact280168RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280141 .summary)
      LeftBound280136.bound (LeftBound280136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17526⟩⟩) (rawTerms := some (Proof.Events1094.exact280141RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280136.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280167.bound, LeftBound280136.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280167.bound, LeftBound280136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280167.actual selector witness, LeftBound280136.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280172

namespace LeftBound280176
def owner : Owner := ⟨.program ⟨257⟩, ⟨20393⟩⟩
def transferEvent : Nat := 280176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280174 .coefficient, .predecessor 1 280175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280174 .coefficient)
      LeftBound280171.bound (LeftBound280171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280175 .coefficient)
      LeftBound279922.bound (LeftBound279922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1093.exact279929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound279922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound279922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280171.bound, LeftBound279922.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280171.bound, LeftBound279922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280171.actual selector witness, LeftBound279922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280176

namespace LeftBound280177
def owner : Owner := ⟨.program ⟨257⟩, ⟨20393⟩⟩
def transferEvent : Nat := 280177
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280173 .summary, .result 279929 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280173 .summary)
      LeftBound280172.bound (LeftBound280172.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17527⟩⟩) (rawTerms := some (Proof.Events1094.exact280173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 279929 .summary)
      LeftBound279924.bound (LeftBound279924.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20392⟩⟩) (rawTerms := some (Proof.Events1093.exact279929RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound279924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280172.bound, LeftBound279924.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280172.bound, LeftBound279924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280172.actual selector witness, LeftBound279924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280177

namespace LeftBound280181
def owner : Owner := ⟨.program ⟨257⟩, ⟨23613⟩⟩
def transferEvent : Nat := 280181
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280179 .coefficient, .predecessor 1 280180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280179 .coefficient)
      LeftBound280176.bound (LeftBound280176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280180 .coefficient)
      LeftBound279710.bound (LeftBound279710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1092.exact279717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound279710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound279710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280176.bound, LeftBound279710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280176.bound, LeftBound279710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280176.actual selector witness, LeftBound279710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280181

namespace LeftBound280182
def owner : Owner := ⟨.program ⟨257⟩, ⟨23613⟩⟩
def transferEvent : Nat := 280182
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280178 .summary, .result 279717 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280178 .summary)
      LeftBound280177.bound (LeftBound280177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20393⟩⟩) (rawTerms := some (Proof.Events1094.exact280178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 279717 .summary)
      LeftBound279712.bound (LeftBound279712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23612⟩⟩) (rawTerms := some (Proof.Events1092.exact279717RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound279712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280177.bound, LeftBound279712.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280177.bound, LeftBound279712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280177.actual selector witness, LeftBound279712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280182

namespace LeftBound280186
def owner : Owner := ⟨.program ⟨257⟩, ⟨33633⟩⟩
def transferEvent : Nat := 280186
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280184 .coefficient, .predecessor 1 280185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280184 .coefficient)
      LeftBound280181.bound (LeftBound280181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280185 .coefficient)
      LeftBound279498.bound (LeftBound279498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1091.exact279505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound279498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound279498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280181.bound, LeftBound279498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280181.bound, LeftBound279498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280181.actual selector witness, LeftBound279498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280186

namespace LeftBound280187
def owner : Owner := ⟨.program ⟨257⟩, ⟨33633⟩⟩
def transferEvent : Nat := 280187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280183 .summary, .result 279505 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280183 .summary)
      LeftBound280182.bound (LeftBound280182.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23613⟩⟩) (rawTerms := some (Proof.Events1094.exact280183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 279505 .summary)
      LeftBound279500.bound (LeftBound279500.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33632⟩⟩) (rawTerms := some (Proof.Events1091.exact279505RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound279500.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280182.bound, LeftBound279500.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280182.bound, LeftBound279500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280182.actual selector witness, LeftBound279500.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280187

namespace LeftBound280191
def owner : Owner := ⟨.program ⟨257⟩, ⟨52693⟩⟩
def transferEvent : Nat := 280191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280189 .coefficient, .predecessor 1 280190 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280189 .coefficient)
      LeftBound280186.bound (LeftBound280186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280190 .coefficient)
      LeftBound279286.bound (LeftBound279286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1090.exact279293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound279286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound279286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280186.bound, LeftBound279286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280186.bound, LeftBound279286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280186.actual selector witness, LeftBound279286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280191

namespace LeftBound280192
def owner : Owner := ⟨.program ⟨257⟩, ⟨52693⟩⟩
def transferEvent : Nat := 280192
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280188 .summary, .result 279293 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280188 .summary)
      LeftBound280187.bound (LeftBound280187.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33633⟩⟩) (rawTerms := some (Proof.Events1094.exact280188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 279293 .summary)
      LeftBound279288.bound (LeftBound279288.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52692⟩⟩) (rawTerms := some (Proof.Events1090.exact279293RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound279288.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280187.bound, LeftBound279288.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280187.bound, LeftBound279288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280187.actual selector witness, LeftBound279288.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280192

namespace LeftBound280196
def owner : Owner := ⟨.program ⟨257⟩, ⟨55673⟩⟩
def transferEvent : Nat := 280196
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280194 .coefficient, .predecessor 1 280195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280194 .coefficient)
      LeftBound280191.bound (LeftBound280191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280195 .coefficient)
      LeftBound279074.bound (LeftBound279074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1090.exact279081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound279074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound279074.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280191.bound, LeftBound279074.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280191.bound, LeftBound279074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280191.actual selector witness, LeftBound279074.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280196

namespace LeftBound280197
def owner : Owner := ⟨.program ⟨257⟩, ⟨55673⟩⟩
def transferEvent : Nat := 280197
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280193 .summary, .result 279081 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280193 .summary)
      LeftBound280192.bound (LeftBound280192.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52693⟩⟩) (rawTerms := some (Proof.Events1094.exact280193RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 279081 .summary)
      LeftBound279076.bound (LeftBound279076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55672⟩⟩) (rawTerms := some (Proof.Events1090.exact279081RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound279076.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280192.bound, LeftBound279076.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280192.bound, LeftBound279076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280192.actual selector witness, LeftBound279076.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280197

namespace LeftBound280201
def owner : Owner := ⟨.program ⟨257⟩, ⟨58653⟩⟩
def transferEvent : Nat := 280201
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280199 .coefficient, .predecessor 1 280200 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280199 .coefficient)
      LeftBound280196.bound (LeftBound280196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280200 .coefficient)
      LeftBound278862.bound (LeftBound278862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1089.exact278869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound278862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound278862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280196.bound, LeftBound278862.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280196.bound, LeftBound278862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280196.actual selector witness, LeftBound278862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280201

namespace LeftBound280202
def owner : Owner := ⟨.program ⟨257⟩, ⟨58653⟩⟩
def transferEvent : Nat := 280202
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280198 .summary, .result 278869 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280198 .summary)
      LeftBound280197.bound (LeftBound280197.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55673⟩⟩) (rawTerms := some (Proof.Events1094.exact280198RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 278869 .summary)
      LeftBound278864.bound (LeftBound278864.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58652⟩⟩) (rawTerms := some (Proof.Events1089.exact278869RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound278864.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280197.bound, LeftBound278864.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280197.bound, LeftBound278864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280197.actual selector witness, LeftBound278864.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280202

namespace LeftBound280206
def owner : Owner := ⟨.program ⟨257⟩, ⟨61633⟩⟩
def transferEvent : Nat := 280206
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280204 .coefficient, .predecessor 1 280205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280204 .coefficient)
      LeftBound280201.bound (LeftBound280201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280205 .coefficient)
      LeftBound278650.bound (LeftBound278650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1088.exact278657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound278650.bound, RecordedBoundRefines] <;> decide)
      (LeftBound278650.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280201.bound, LeftBound278650.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280201.bound, LeftBound278650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280201.actual selector witness, LeftBound278650.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280206

namespace LeftBound280207
def owner : Owner := ⟨.program ⟨257⟩, ⟨61633⟩⟩
def transferEvent : Nat := 280207
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280203 .summary, .result 278657 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280203 .summary)
      LeftBound280202.bound (LeftBound280202.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58653⟩⟩) (rawTerms := some (Proof.Events1094.exact280203RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 278657 .summary)
      LeftBound278652.bound (LeftBound278652.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61632⟩⟩) (rawTerms := some (Proof.Events1088.exact278657RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound278652.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280202.bound, LeftBound278652.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280202.bound, LeftBound278652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280202.actual selector witness, LeftBound278652.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280207

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
