import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge138980
def owner : Owner := ⟨.program ⟨257⟩, ⟨64365⟩⟩
def mergeEvent : Nat := 138980
def frameStart : Nat := 138888
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩] } }
def leftRaw : List Term := Proof.Events542.exact138976RawTerms
def rightRaw : List Term := Proof.Events542.exact138933RawTerms
def group : MergeGroup := .operator 138976 138933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138976) (leftOrdinal := 0)
    (rightResult := 138933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64362⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138980

namespace LeftMerge138981
def owner : Owner := ⟨.program ⟨257⟩, ⟨64365⟩⟩
def mergeEvent : Nat := 138981
def frameStart : Nat := 138888
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩] } }
def leftRaw : List Term := Proof.Events542.exact138976RawTerms
def rightRaw : List Term := Proof.Events542.exact138933RawTerms
def group : MergeGroup := .operator 138976 138933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138976) (leftOrdinal := 1)
    (rightResult := 138933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64362⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138981

namespace LeftMerge138983
def owner : Owner := ⟨.program ⟨257⟩, ⟨64365⟩⟩
def mergeEvent : Nat := 138983
def frameStart : Nat := 138888
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63887⟩⟩] } }
def rhsRaw : List Term := Proof.Events542.exact138930RawTerms
def group : MergeGroup := .relation 138982
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138982) (rhsResult := 138930)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64362⟩⟩) ⟨63887⟩ 138930) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63887⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138983

namespace LeftMerge138991
def owner : Owner := ⟨.program ⟨257⟩, ⟨62754⟩⟩
def mergeEvent : Nat := 138991
def frameStart : Nat := 138888
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events542.exact138944RawTerms
def rightRaw : List Term := Proof.Events542.exact138987RawTerms
def group : MergeGroup := .operator 138944 138987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138944) (leftOrdinal := 0)
    (rightResult := 138987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138991

namespace LeftMerge139008
def owner : Owner := ⟨.program ⟨257⟩, ⟨63302⟩⟩
def mergeEvent : Nat := 139008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events542.exact139005RawTerms
def group : MergeGroup := .relation 139007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139007) (rhsResult := 139005)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139006 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩) (none) 139005) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139008

namespace LeftMerge139009
def owner : Owner := ⟨.program ⟨257⟩, ⟨63302⟩⟩
def mergeEvent : Nat := 139009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩] } }
def rhsRaw : List Term := Proof.Events542.exact139005RawTerms
def group : MergeGroup := .relation 139007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139007) (rhsResult := 139005)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139006 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩) (none) 139005) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139009

namespace LeftMerge139010
def owner : Owner := ⟨.program ⟨257⟩, ⟨63302⟩⟩
def mergeEvent : Nat := 139010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63887⟩⟩] } }
def rhsRaw : List Term := Proof.Events542.exact139005RawTerms
def group : MergeGroup := .relation 139007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139007) (rhsResult := 139005)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139006 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩) (none) 139005) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63887⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139010

namespace LeftMerge139011
def owner : Owner := ⟨.program ⟨257⟩, ⟨63302⟩⟩
def mergeEvent : Nat := 139011
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events542.exact139005RawTerms
def group : MergeGroup := .relation 139007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139007) (rhsResult := 139005)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 139006 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩) (none) 139005) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139011

namespace LeftMerge139016
def owner : Owner := ⟨.program ⟨257⟩, ⟨64364⟩⟩
def mergeEvent : Nat := 139016
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63887⟩⟩] } }
def leftRaw : List Term := Proof.Events543.exact139012RawTerms
def rightRaw : List Term := Proof.Events542.exact138826RawTerms
def group : MergeGroup := .operator 139012 138826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139012) (leftOrdinal := 2)
    (rightResult := 138826) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63887⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63887⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139016

namespace LeftMerge139017
def owner : Owner := ⟨.program ⟨257⟩, ⟨64364⟩⟩
def mergeEvent : Nat := 139017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩] } }
def leftRaw : List Term := Proof.Events543.exact139012RawTerms
def rightRaw : List Term := Proof.Events542.exact138826RawTerms
def group : MergeGroup := .operator 139012 138826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139012) (leftOrdinal := 1)
    (rightResult := 138826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139017

namespace LeftMerge139025
def owner : Owner := ⟨.program ⟨257⟩, ⟨64657⟩⟩
def mergeEvent : Nat := 139025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64655⟩⟩] } }
def leftRaw : List Term := Proof.Events543.exact139019RawTerms
def rightRaw : List Term := Proof.Events541.exact138742RawTerms
def group : MergeGroup := .operator 139019 138742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139019) (leftOrdinal := 0)
    (rightResult := 138742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64655⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139025

namespace LeftMerge139026
def owner : Owner := ⟨.program ⟨257⟩, ⟨64657⟩⟩
def mergeEvent : Nat := 139026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64655⟩⟩] } }
def leftRaw : List Term := Proof.Events543.exact139019RawTerms
def rightRaw : List Term := Proof.Events541.exact138742RawTerms
def group : MergeGroup := .operator 139019 138742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139019) (leftOrdinal := 1)
    (rightResult := 138742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64655⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64655⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139026

namespace LeftMerge139028
def owner : Owner := ⟨.program ⟨257⟩, ⟨64657⟩⟩
def mergeEvent : Nat := 139028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64018⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138739RawTerms
def group : MergeGroup := .relation 139027
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 139027) (rhsResult := 138739)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64655⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64655⟩⟩) ⟨64018⟩ 138739) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64018⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨64018⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge139028

namespace LeftMerge139042
def owner : Owner := ⟨.program ⟨257⟩, ⟨63539⟩⟩
def mergeEvent : Nat := 139042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63536⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events543.exact139036RawTerms
def group : MergeGroup := .operator 134495 139036
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 139036) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139042

namespace LeftMerge139163
def owner : Owner := ⟨.program ⟨257⟩, ⟨64260⟩⟩
def mergeEvent : Nat := 139163
def frameStart : Nat := 139097
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events543.exact139159RawTerms
def rightRaw : List Term := Proof.Events543.exact139157RawTerms
def group : MergeGroup := .operator 139159 139157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139159) (leftOrdinal := 0)
    (rightResult := 139157) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62752⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139163

namespace LeftMerge139175
def owner : Owner := ⟨.program ⟨257⟩, ⟨64656⟩⟩
def mergeEvent : Nat := 139175
def frameStart : Nat := 139097
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64655⟩⟩] } }
def leftRaw : List Term := Proof.Events543.exact139171RawTerms
def rightRaw : List Term := Proof.Events543.exact139148RawTerms
def group : MergeGroup := .operator 139171 139148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 139171) (leftOrdinal := 0)
    (rightResult := 139148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64655⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge139175

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
