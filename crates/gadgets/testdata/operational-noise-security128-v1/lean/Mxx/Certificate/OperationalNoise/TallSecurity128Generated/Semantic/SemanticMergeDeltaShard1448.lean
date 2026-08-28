import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge235830
def owner : Owner := ⟨.program ⟨257⟩, ⟨23837⟩⟩
def mergeEvent : Nat := 235830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23071⟩⟩] } }
def leftRaw : List Term := Proof.Events921.exact235825RawTerms
def rightRaw : List Term := Proof.Events920.exact235647RawTerms
def group : MergeGroup := .operator 235825 235647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 235825) (leftOrdinal := 2)
    (rightResult := 235647) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23071⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23071⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge235830

namespace LeftMerge235838
def owner : Owner := ⟨.program ⟨257⟩, ⟨23838⟩⟩
def mergeEvent : Nat := 235838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩] } }
def leftRaw : List Term := Proof.Events921.exact235832RawTerms
def rightRaw : List Term := Proof.Events061.exact15842RawTerms
def group : MergeGroup := .operator 235832 15842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 235832) (leftOrdinal := 0)
    (rightResult := 15842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7155⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge235838

namespace LeftMerge235839
def owner : Owner := ⟨.program ⟨257⟩, ⟨23838⟩⟩
def mergeEvent : Nat := 235839
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩] } }
def leftRaw : List Term := Proof.Events921.exact235832RawTerms
def rightRaw : List Term := Proof.Events061.exact15842RawTerms
def group : MergeGroup := .operator 235832 15842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 235832) (leftOrdinal := 1)
    (rightResult := 15842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7155⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge235839

namespace LeftMerge235841
def owner : Owner := ⟨.program ⟨257⟩, ⟨23838⟩⟩
def mergeEvent : Nat := 235841
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15835RawTerms
def group : MergeGroup := .relation 235840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 235840) (rhsResult := 15835)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7155⟩⟩) ⟨7043⟩ 15835) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge235841

namespace LeftMerge235855
def owner : Owner := ⟨.program ⟨257⟩, ⟨20616⟩⟩
def mergeEvent : Nat := 235855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230143RawTerms
def rightRaw : List Term := Proof.Events921.exact235849RawTerms
def group : MergeGroup := .operator 230143 235849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230143) (leftOrdinal := 0)
    (rightResult := 235849) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20614⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge235855

namespace LeftMerge235856
def owner : Owner := ⟨.program ⟨257⟩, ⟨20616⟩⟩
def mergeEvent : Nat := 235856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230143RawTerms
def rightRaw : List Term := Proof.Events921.exact235849RawTerms
def group : MergeGroup := .operator 230143 235849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230143) (leftOrdinal := 1)
    (rightResult := 235849) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20614⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge235856

namespace LeftMerge235858
def owner : Owner := ⟨.program ⟨257⟩, ⟨20616⟩⟩
def mergeEvent : Nat := 235858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19851⟩⟩] } }
def rhsRaw : List Term := Proof.Events921.exact235846RawTerms
def group : MergeGroup := .relation 235857
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 235857) (rhsResult := 235846)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20614⟩⟩) ⟨19851⟩ 235846) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19851⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge235858

namespace LeftMerge235872
def owner : Owner := ⟨.program ⟨257⟩, ⟨19435⟩⟩
def mergeEvent : Nat := 235872
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events921.exact235866RawTerms
def group : MergeGroup := .operator 222245 235866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 235866) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19432⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge235872

namespace LeftMerge235993
def owner : Owner := ⟨.program ⟨257⟩, ⟨20064⟩⟩
def mergeEvent : Nat := 235993
def frameStart : Nat := 235927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events921.exact235989RawTerms
def rightRaw : List Term := Proof.Events921.exact235987RawTerms
def group : MergeGroup := .operator 235989 235987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 235989) (leftOrdinal := 0)
    (rightResult := 235987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge235993

namespace LeftMerge236005
def owner : Owner := ⟨.program ⟨257⟩, ⟨20615⟩⟩
def mergeEvent : Nat := 236005
def frameStart : Nat := 235927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩] } }
def leftRaw : List Term := Proof.Events921.exact236001RawTerms
def rightRaw : List Term := Proof.Events921.exact235978RawTerms
def group : MergeGroup := .operator 236001 235978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236001) (leftOrdinal := 0)
    (rightResult := 235978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20614⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236005

namespace LeftMerge236006
def owner : Owner := ⟨.program ⟨257⟩, ⟨20615⟩⟩
def mergeEvent : Nat := 236006
def frameStart : Nat := 235927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩] } }
def leftRaw : List Term := Proof.Events921.exact236001RawTerms
def rightRaw : List Term := Proof.Events921.exact235978RawTerms
def group : MergeGroup := .operator 236001 235978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236001) (leftOrdinal := 1)
    (rightResult := 235978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20614⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge236006

namespace LeftMerge236008
def owner : Owner := ⟨.program ⟨257⟩, ⟨20615⟩⟩
def mergeEvent : Nat := 236008
def frameStart : Nat := 235927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19851⟩⟩] } }
def rhsRaw : List Term := Proof.Events921.exact235975RawTerms
def group : MergeGroup := .relation 236007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 236007) (rhsResult := 235975)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20614⟩⟩) ⟨19851⟩ 235975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19851⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge236008

namespace LeftMerge236016
def owner : Owner := ⟨.program ⟨257⟩, ⟨18845⟩⟩
def mergeEvent : Nat := 236016
def frameStart : Nat := 235927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events921.exact235989RawTerms
def rightRaw : List Term := Proof.Events921.exact236012RawTerms
def group : MergeGroup := .operator 235989 236012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 235989) (leftOrdinal := 0)
    (rightResult := 236012) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18842⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236016

namespace LeftMerge236033
def owner : Owner := ⟨.program ⟨257⟩, ⟨19435⟩⟩
def mergeEvent : Nat := 236033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }
def rhsRaw : List Term := Proof.Events921.exact236030RawTerms
def group : MergeGroup := .relation 236032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 236032) (rhsResult := 236030)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 236031 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩) (none) 236030) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236033

namespace LeftMerge236034
def owner : Owner := ⟨.program ⟨257⟩, ⟨19435⟩⟩
def mergeEvent : Nat := 236034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩] } }
def rhsRaw : List Term := Proof.Events921.exact236030RawTerms
def group : MergeGroup := .relation 236032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 236032) (rhsResult := 236030)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 236031 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩) (none) 236030) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge236034

namespace LeftMerge236035
def owner : Owner := ⟨.program ⟨257⟩, ⟨19435⟩⟩
def mergeEvent : Nat := 236035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19851⟩⟩] } }
def rhsRaw : List Term := Proof.Events921.exact236030RawTerms
def group : MergeGroup := .relation 236032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 236032) (rhsResult := 236030)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 236031 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩) (none) 236030) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19851⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236035

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
