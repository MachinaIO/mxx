import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge110724
def owner : Owner := ⟨.program ⟨257⟩, ⟨57422⟩⟩
def mergeEvent : Nat := 110724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25022⟩⟩, ⟨.program ⟨257⟩, ⟨56532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57975⟩⟩] } }
def rhsRaw : List Term := Proof.Events432.exact110719RawTerms
def group : MergeGroup := .relation 110721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110721) (rhsResult := 110719)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57419⟩⟩]⟩) (none) 110719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25022⟩⟩, ⟨.program ⟨257⟩, ⟨56532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57975⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25022⟩⟩, ⟨.program ⟨257⟩, ⟨56532⟩⟩], [⟨.program ⟨257⟩, ⟨57975⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110724

namespace LeftMerge110725
def owner : Owner := ⟨.program ⟨257⟩, ⟨57422⟩⟩
def mergeEvent : Nat := 110725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events432.exact110719RawTerms
def group : MergeGroup := .relation 110721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110721) (rhsResult := 110719)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57419⟩⟩]⟩) (none) 110719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110725

namespace LeftMerge110730
def owner : Owner := ⟨.program ⟨257⟩, ⟨58492⟩⟩
def mergeEvent : Nat := 110730
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25022⟩⟩, ⟨.program ⟨257⟩, ⟨56532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57975⟩⟩] } }
def leftRaw : List Term := Proof.Events432.exact110726RawTerms
def rightRaw : List Term := Proof.Events431.exact110540RawTerms
def group : MergeGroup := .operator 110726 110540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110726) (leftOrdinal := 2)
    (rightResult := 110540) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25022⟩⟩, ⟨.program ⟨257⟩, ⟨56532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57975⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25022⟩⟩, ⟨.program ⟨257⟩, ⟨56532⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57975⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25022⟩⟩, ⟨.program ⟨257⟩, ⟨56532⟩⟩], [⟨.program ⟨257⟩, ⟨57975⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110730

namespace LeftMerge110731
def owner : Owner := ⟨.program ⟨257⟩, ⟨58492⟩⟩
def mergeEvent : Nat := 110731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58490⟩⟩] } }
def leftRaw : List Term := Proof.Events432.exact110726RawTerms
def rightRaw : List Term := Proof.Events431.exact110540RawTerms
def group : MergeGroup := .operator 110726 110540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110726) (leftOrdinal := 1)
    (rightResult := 110540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58490⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58490⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58490⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110731

namespace LeftMerge110739
def owner : Owner := ⟨.program ⟨257⟩, ⟨58945⟩⟩
def mergeEvent : Nat := 110739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩] } }
def leftRaw : List Term := Proof.Events432.exact110733RawTerms
def rightRaw : List Term := Proof.Events431.exact110456RawTerms
def group : MergeGroup := .operator 110733 110456
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110733) (leftOrdinal := 0)
    (rightResult := 110456) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58943⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110739

namespace LeftMerge110740
def owner : Owner := ⟨.program ⟨257⟩, ⟨58945⟩⟩
def mergeEvent : Nat := 110740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩] } }
def leftRaw : List Term := Proof.Events432.exact110733RawTerms
def rightRaw : List Term := Proof.Events431.exact110456RawTerms
def group : MergeGroup := .operator 110733 110456
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110733) (leftOrdinal := 1)
    (rightResult := 110456) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58943⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110740

namespace LeftMerge110742
def owner : Owner := ⟨.program ⟨257⟩, ⟨58945⟩⟩
def mergeEvent : Nat := 110742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58130⟩⟩] } }
def rhsRaw : List Term := Proof.Events431.exact110453RawTerms
def group : MergeGroup := .relation 110741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110741) (rhsResult := 110453)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58943⟩⟩) ⟨58130⟩ 110453) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58130⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨58130⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110742

namespace LeftMerge110756
def owner : Owner := ⟨.program ⟨257⟩, ⟨57739⟩⟩
def mergeEvent : Nat := 110756
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57736⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events432.exact110750RawTerms
def group : MergeGroup := .operator 105245 110750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 110750) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57736⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57736⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110756

namespace LeftMerge110877
def owner : Owner := ⟨.program ⟨257⟩, ⟨58332⟩⟩
def mergeEvent : Nat := 110877
def frameStart : Nat := 110811
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events433.exact110873RawTerms
def rightRaw : List Term := Proof.Events433.exact110871RawTerms
def group : MergeGroup := .operator 110873 110871
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110873) (leftOrdinal := 0)
    (rightResult := 110871) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110877

namespace LeftMerge110889
def owner : Owner := ⟨.program ⟨257⟩, ⟨58944⟩⟩
def mergeEvent : Nat := 110889
def frameStart : Nat := 110811
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩] } }
def leftRaw : List Term := Proof.Events433.exact110885RawTerms
def rightRaw : List Term := Proof.Events433.exact110862RawTerms
def group : MergeGroup := .operator 110885 110862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110885) (leftOrdinal := 0)
    (rightResult := 110862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58943⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110889

namespace LeftMerge110890
def owner : Owner := ⟨.program ⟨257⟩, ⟨58944⟩⟩
def mergeEvent : Nat := 110890
def frameStart : Nat := 110811
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩] } }
def leftRaw : List Term := Proof.Events433.exact110885RawTerms
def rightRaw : List Term := Proof.Events433.exact110862RawTerms
def group : MergeGroup := .operator 110885 110862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110885) (leftOrdinal := 1)
    (rightResult := 110862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58943⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110890

namespace LeftMerge110892
def owner : Owner := ⟨.program ⟨257⟩, ⟨58944⟩⟩
def mergeEvent : Nat := 110892
def frameStart : Nat := 110811
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58130⟩⟩] } }
def rhsRaw : List Term := Proof.Events433.exact110859RawTerms
def group : MergeGroup := .relation 110891
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110891) (rhsResult := 110859)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58943⟩⟩) ⟨58130⟩ 110859) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58130⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨58130⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110892

namespace LeftMerge110900
def owner : Owner := ⟨.program ⟨257⟩, ⟨57142⟩⟩
def mergeEvent : Nat := 110900
def frameStart : Nat := 110811
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events433.exact110873RawTerms
def rightRaw : List Term := Proof.Events433.exact110896RawTerms
def group : MergeGroup := .operator 110873 110896
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 110873) (leftOrdinal := 0)
    (rightResult := 110896) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57140⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110900

namespace LeftMerge110917
def owner : Owner := ⟨.program ⟨257⟩, ⟨57739⟩⟩
def mergeEvent : Nat := 110917
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }
def rhsRaw : List Term := Proof.Events433.exact110914RawTerms
def group : MergeGroup := .relation 110916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110916) (rhsResult := 110914)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110915 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57736⟩⟩]⟩) (none) 110914) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110917

namespace LeftMerge110918
def owner : Owner := ⟨.program ⟨257⟩, ⟨57739⟩⟩
def mergeEvent : Nat := 110918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩] } }
def rhsRaw : List Term := Proof.Events433.exact110914RawTerms
def group : MergeGroup := .relation 110916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110916) (rhsResult := 110914)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110915 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57736⟩⟩]⟩) (none) 110914) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58943⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge110918

namespace LeftMerge110919
def owner : Owner := ⟨.program ⟨257⟩, ⟨57739⟩⟩
def mergeEvent : Nat := 110919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58130⟩⟩] } }
def rhsRaw : List Term := Proof.Events433.exact110914RawTerms
def group : MergeGroup := .relation 110916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 110916) (rhsResult := 110914)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 110915 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57736⟩⟩]⟩) (none) 110914) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58130⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨56856⟩⟩], [⟨.program ⟨257⟩, ⟨58130⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge110919

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
