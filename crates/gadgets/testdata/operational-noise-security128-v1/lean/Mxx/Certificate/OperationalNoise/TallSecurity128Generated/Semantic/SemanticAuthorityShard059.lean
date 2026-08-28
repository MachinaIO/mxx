import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftAuthority12337
def owner : Owner := ⟨.program ⟨257⟩, ⟨24710⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨12, by decide⟩
def producerEvent : Nat := 12337
def resultEvent : Nat := 12338
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 12) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12337

namespace LeftAuthority12340
def owner : Owner := ⟨.program ⟨257⟩, ⟨53390⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨12, by decide⟩
def producerEvent : Nat := 12340
def resultEvent : Nat := 12341
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 12) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12340

namespace LeftAuthority12360
def owner : Owner := ⟨.program ⟨257⟩, ⟨24470⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨10, by decide⟩
def producerEvent : Nat := 12360
def resultEvent : Nat := 12361
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24470⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 10) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12360

namespace LeftAuthority12363
def owner : Owner := ⟨.program ⟨257⟩, ⟨50410⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨10, by decide⟩
def producerEvent : Nat := 12363
def resultEvent : Nat := 12364
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 10) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12363

namespace LeftAuthority12383
def owner : Owner := ⟨.program ⟨257⟩, ⟨24230⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨6, by decide⟩
def producerEvent : Nat := 12383
def resultEvent : Nat := 12384
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 6) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12383

namespace LeftAuthority12386
def owner : Owner := ⟨.program ⟨257⟩, ⟨31350⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨6, by decide⟩
def producerEvent : Nat := 12386
def resultEvent : Nat := 12387
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 6) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12386

namespace LeftAuthority12406
def owner : Owner := ⟨.program ⟨257⟩, ⟨21374⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨4, by decide⟩
def producerEvent : Nat := 12406
def resultEvent : Nat := 12407
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 4) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12406

namespace LeftAuthority12409
def owner : Owner := ⟨.program ⟨257⟩, ⟨21026⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨4, by decide⟩
def producerEvent : Nat := 12409
def resultEvent : Nat := 12410
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 4) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12409

namespace LeftAuthority12429
def owner : Owner := ⟨.program ⟨257⟩, ⟨18154⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨3, by decide⟩
def producerEvent : Nat := 12429
def resultEvent : Nat := 12430
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18154⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 3) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12429

namespace LeftAuthority12432
def owner : Owner := ⟨.program ⟨257⟩, ⟨12606⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨3, by decide⟩
def producerEvent : Nat := 12432
def resultEvent : Nat := 12433
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12606⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 3) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12432

namespace LeftAuthority12452
def owner : Owner := ⟨.program ⟨257⟩, ⟨15354⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨2, by decide⟩
def producerEvent : Nat := 12452
def resultEvent : Nat := 12453
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 2) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12452

namespace LeftAuthority12455
def owner : Owner := ⟨.program ⟨257⟩, ⟨12306⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨2, by decide⟩
def producerEvent : Nat := 12455
def resultEvent : Nat := 12456
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12306⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 2) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12455

namespace LeftAuthority12546
def owner : Owner := ⟨.program ⟨257⟩, ⟨67363⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨18, by decide⟩
def producerEvent : Nat := 12546
def resultEvent : Nat := 12547
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67363⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 18) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12546

namespace LeftAuthority12554
def owner : Owner := ⟨.program ⟨257⟩, ⟨48294⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨60, by decide⟩
def producerEvent : Nat := 12554
def resultEvent : Nat := 12555
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48294⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 60) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12554

namespace LeftAuthority12562
def owner : Owner := ⟨.program ⟨257⟩, ⟨45614⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨58, by decide⟩
def producerEvent : Nat := 12562
def resultEvent : Nat := 12563
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45614⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 58) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12562

namespace LeftAuthority12570
def owner : Owner := ⟨.program ⟨257⟩, ⟨42937⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨52, by decide⟩
def producerEvent : Nat := 12570
def resultEvent : Nat := 12571
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42937⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 52) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority12570

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
