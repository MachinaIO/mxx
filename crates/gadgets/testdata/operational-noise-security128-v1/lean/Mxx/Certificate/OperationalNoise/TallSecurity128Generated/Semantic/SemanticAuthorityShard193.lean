import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftAuthority54610
def owner : Owner := ⟨.program ⟨257⟩, ⟨18652⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨3, by decide⟩
def producerEvent : Nat := 54610
def resultEvent : Nat := 54611
def frameStart : Nat := 54512
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 3) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54610

namespace LeftAuthority54618
def owner : Owner := ⟨.program ⟨257⟩, ⟨7180⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 54618
def resultEvent : Nat := 54619
def frameStart : Nat := 54512
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54618

namespace LeftAuthority54655
def owner : Owner := ⟨.program ⟨257⟩, ⟨19616⟩⟩
def authority : Authority := .relationPreimageSource ⟨59⟩
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
def producerEvent : Nat := 54655
def resultEvent : Nat := 54656
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19616⟩⟩] } }]) (recordedCoefficientBound := .finite 5647228698) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54655

namespace LeftAuthority54757
def owner : Owner := ⟨.program ⟨257⟩, ⟨18652⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨3, by decide⟩
def producerEvent : Nat := 54757
def resultEvent : Nat := 54758
def frameStart : Nat := 54721
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 3) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54757

namespace LeftAuthority54768
def owner : Owner := ⟨.program ⟨257⟩, ⟨19933⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 54768
def resultEvent : Nat := 54769
def frameStart : Nat := 54721
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19933⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54768

namespace LeftAuthority54771
def owner : Owner := ⟨.program ⟨257⟩, ⟨20900⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 54771
def resultEvent : Nat := 54772
def frameStart : Nat := 54721
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20900⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54771

namespace LeftAuthority54773
def owner : Owner := ⟨.program ⟨257⟩, ⟨136⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .exactZero
def producerEvent : Nat := 54773
def resultEvent : Nat := 54774
def frameStart : Nat := 54721
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultCoefficient (by decide) (by rfl) (by rfl)
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54773

namespace LeftAuthority54782
def owner : Owner := ⟨.program ⟨257⟩, ⟨6908⟩⟩
def authority : Authority := .factStore
def bound : CoeffClass := .large
def producerEvent : Nat := 54782
def resultEvent : Nat := 54783
def frameStart : Nat := 54721
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54782

namespace LeftAuthority54790
def owner : Owner := ⟨.program ⟨257⟩, ⟨7180⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 54790
def resultEvent : Nat := 54791
def frameStart : Nat := 54721
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54790

namespace LeftAuthority54805
def owner : Owner := ⟨.program ⟨257⟩, ⟨19018⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨48, by decide⟩
def producerEvent : Nat := 54805
def resultEvent : Nat := 54806
def frameStart : Nat := 54721
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 48) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54805

namespace LeftAuthority54813
def owner : Owner := ⟨.program ⟨257⟩, ⟨7200⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 54813
def resultEvent : Nat := 54814
def frameStart : Nat := 54721
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54813

namespace LeftAuthority54844
def owner : Owner := ⟨.program ⟨257⟩, ⟨17073⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 54844
def resultEvent : Nat := 54845
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17073⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54844

namespace LeftAuthority54847
def owner : Owner := ⟨.program ⟨257⟩, ⟨17985⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 54847
def resultEvent : Nat := 54848
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17985⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54847

namespace LeftAuthority54854
def owner : Owner := ⟨.program ⟨257⟩, ⟨16897⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 54854
def resultEvent : Nat := 54855
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16897⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54854

namespace LeftAuthority54857
def owner : Owner := ⟨.program ⟨257⟩, ⟨17447⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 54857
def resultEvent : Nat := 54858
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17447⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54857

namespace LeftAuthority54934
def owner : Owner := ⟨.program ⟨257⟩, ⟨16369⟩⟩
def authority : Authority := .relationPreimageSource ⟨36⟩
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
def producerEvent : Nat := 54934
def resultEvent : Nat := 54935
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16369⟩⟩] } }]) (recordedCoefficientBound := .finite 5647228698) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority54934

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
