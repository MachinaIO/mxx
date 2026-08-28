import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftAuthority49965
def owner : Owner := ⟨.program ⟨214⟩, ⟨20328⟩⟩
def authority : Authority := .relationPreimageSource ⟨27⟩
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
def producerEvent : Nat := 49965
def resultEvent : Nat := 49966
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20328⟩⟩] } }]) (recordedCoefficientBound := .finite 136065468) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority49965

namespace LeftAuthority50067
def owner : Owner := ⟨.program ⟨214⟩, ⟨14800⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨2, by decide⟩
def producerEvent : Nat := 50067
def resultEvent : Nat := 50068
def frameStart : Nat := 50031
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 2) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50067

namespace LeftAuthority50078
def owner : Owner := ⟨.program ⟨214⟩, ⟨23726⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 50078
def resultEvent : Nat := 50079
def frameStart : Nat := 50031
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50078

namespace LeftAuthority50081
def owner : Owner := ⟨.program ⟨214⟩, ⟨26375⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 50081
def resultEvent : Nat := 50082
def frameStart : Nat := 50031
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26375⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50081

namespace LeftAuthority50083
def owner : Owner := ⟨.program ⟨214⟩, ⟨110⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .exactZero
def producerEvent : Nat := 50083
def resultEvent : Nat := 50084
def frameStart : Nat := 50031
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultCoefficient (by decide) (by rfl) (by rfl)
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50083

namespace LeftAuthority50092
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def authority : Authority := .factStore
def bound : CoeffClass := .large
def producerEvent : Nat := 50092
def resultEvent : Nat := 50093
def frameStart : Nat := 50031
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50092

namespace LeftAuthority50100
def owner : Owner := ⟨.program ⟨214⟩, ⟨6690⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 50100
def resultEvent : Nat := 50101
def frameStart : Nat := 50031
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50100

namespace LeftAuthority50115
def owner : Owner := ⟨.program ⟨214⟩, ⟨14896⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨2, by decide⟩
def producerEvent : Nat := 50115
def resultEvent : Nat := 50116
def frameStart : Nat := 50031
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 2) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50115

namespace LeftAuthority50123
def owner : Owner := ⟨.program ⟨214⟩, ⟨6708⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 50123
def resultEvent : Nat := 50124
def frameStart : Nat := 50031
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50123

namespace LeftAuthority50281
def owner : Owner := ⟨.program ⟨214⟩, ⟨70⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨26, by decide⟩
def producerEvent : Nat := 50281
def resultEvent : Nat := 50282
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨70⟩⟩] } }]) (recordedCoefficientBound := .finite 26) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50281

namespace LeftAuthority50502
def owner : Owner := ⟨.program ⟨214⟩, ⟨69⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨26, by decide⟩
def producerEvent : Nat := 50502
def resultEvent : Nat := 50503
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨69⟩⟩] } }]) (recordedCoefficientBound := .finite 26) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50502

namespace LeftAuthority50641
def owner : Owner := ⟨.program ⟨214⟩, ⟨18620⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 50641
def resultEvent : Nat := 50642
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50641

namespace LeftAuthority50644
def owner : Owner := ⟨.program ⟨214⟩, ⟨18684⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 50644
def resultEvent : Nat := 50645
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50644

namespace LeftAuthority50651
def owner : Owner := ⟨.program ⟨214⟩, ⟨24795⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 50651
def resultEvent : Nat := 50652
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50651

namespace LeftAuthority50654
def owner : Owner := ⟨.program ⟨214⟩, ⟨30139⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 50654
def resultEvent : Nat := 50655
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30139⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50654

namespace LeftAuthority50661
def owner : Owner := ⟨.program ⟨214⟩, ⟨23418⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 50661
def resultEvent : Nat := 50662
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority50661

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
