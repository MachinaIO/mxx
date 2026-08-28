import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge101829
def owner : Owner := ⟨.program ⟨214⟩, ⟨24899⟩⟩
def mergeEvent : Nat := 101829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101823RawTerms
def rightRaw : List Term := Proof.Events397.exact101759RawTerms
def group : MergeGroup := .operator 101823 101759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101823) (leftOrdinal := 1)
    (rightResult := 101759) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24898⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101829

namespace LeftMerge101831
def owner : Owner := ⟨.program ⟨214⟩, ⟨24899⟩⟩
def mergeEvent : Nat := 101831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }
def rhsRaw : List Term := Proof.Events397.exact101756RawTerms
def group : MergeGroup := .relation 101830
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101830) (rhsResult := 101756)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24898⟩⟩) ⟨22948⟩ 101756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨22948⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101831

namespace LeftMerge101832
def owner : Owner := ⟨.program ⟨214⟩, ⟨24899⟩⟩
def mergeEvent : Nat := 101832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101823RawTerms
def rightRaw : List Term := Proof.Events397.exact101759RawTerms
def group : MergeGroup := .operator 101823 101759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101823) (leftOrdinal := 0)
    (rightResult := 101759) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24898⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101832

namespace LeftMerge101846
def owner : Owner := ⟨.program ⟨214⟩, ⟨19016⟩⟩
def mergeEvent : Nat := 101846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events397.exact101840RawTerms
def group : MergeGroup := .operator 94462 101840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 101840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19013⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101846

namespace LeftMerge101901
def owner : Owner := ⟨.program ⟨214⟩, ⟨10457⟩⟩
def mergeEvent : Nat := 101901
def frameStart : Nat := 101883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events398.exact101897RawTerms
def rightRaw : List Term := Proof.Events398.exact101894RawTerms
def group : MergeGroup := .operator 101897 101894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101897) (leftOrdinal := 0)
    (rightResult := 101894) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101901

namespace LeftMerge101931
def owner : Owner := ⟨.program ⟨214⟩, ⟨10570⟩⟩
def mergeEvent : Nat := 101931
def frameStart : Nat := 101883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact101927RawTerms
def rightRaw : List Term := Proof.Events398.exact101925RawTerms
def group : MergeGroup := .operator 101927 101925
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101927) (leftOrdinal := 0)
    (rightResult := 101925) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101931

namespace LeftMerge101954
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def mergeEvent : Nat := 101954
def frameStart : Nat := 101883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact101950RawTerms
def rightRaw : List Term := Proof.Events398.exact101947RawTerms
def group : MergeGroup := .operator 101950 101947
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101950) (leftOrdinal := 0)
    (rightResult := 101947) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7831⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101954

namespace LeftMerge101963
def owner : Owner := ⟨.program ⟨214⟩, ⟨24901⟩⟩
def mergeEvent : Nat := 101963
def frameStart : Nat := 101883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact101959RawTerms
def rightRaw : List Term := Proof.Events398.exact101916RawTerms
def group : MergeGroup := .operator 101959 101916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101959) (leftOrdinal := 0)
    (rightResult := 101916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24898⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101963

namespace LeftMerge101964
def owner : Owner := ⟨.program ⟨214⟩, ⟨24901⟩⟩
def mergeEvent : Nat := 101964
def frameStart : Nat := 101883
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact101959RawTerms
def rightRaw : List Term := Proof.Events398.exact101916RawTerms
def group : MergeGroup := .operator 101959 101916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101959) (leftOrdinal := 1)
    (rightResult := 101916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24898⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101964

namespace LeftMerge101966
def owner : Owner := ⟨.program ⟨214⟩, ⟨24901⟩⟩
def mergeEvent : Nat := 101966
def frameStart : Nat := 101883
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }
def rhsRaw : List Term := Proof.Events398.exact101913RawTerms
def group : MergeGroup := .relation 101965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101965) (rhsResult := 101913)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24898⟩⟩) ⟨22948⟩ 101913) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨22948⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101966

namespace LeftMerge101974
def owner : Owner := ⟨.program ⟨214⟩, ⟨14784⟩⟩
def mergeEvent : Nat := 101974
def frameStart : Nat := 101883
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact101927RawTerms
def rightRaw : List Term := Proof.Events398.exact101970RawTerms
def group : MergeGroup := .operator 101927 101970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101927) (leftOrdinal := 0)
    (rightResult := 101970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101974

namespace LeftMerge101991
def owner : Owner := ⟨.program ⟨214⟩, ⟨19016⟩⟩
def mergeEvent : Nat := 101991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }
def rhsRaw : List Term := Proof.Events398.exact101988RawTerms
def group : MergeGroup := .relation 101990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101990) (rhsResult := 101988)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101989 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩) (none) 101988) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101991

namespace LeftMerge101992
def owner : Owner := ⟨.program ⟨214⟩, ⟨19016⟩⟩
def mergeEvent : Nat := 101992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }
def rhsRaw : List Term := Proof.Events398.exact101988RawTerms
def group : MergeGroup := .relation 101990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101990) (rhsResult := 101988)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101989 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩) (none) 101988) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101992

namespace LeftMerge101993
def owner : Owner := ⟨.program ⟨214⟩, ⟨19016⟩⟩
def mergeEvent : Nat := 101993
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }
def rhsRaw : List Term := Proof.Events398.exact101988RawTerms
def group : MergeGroup := .relation 101990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101990) (rhsResult := 101988)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101989 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩) (none) 101988) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨22948⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101993

namespace LeftMerge101994
def owner : Owner := ⟨.program ⟨214⟩, ⟨19016⟩⟩
def mergeEvent : Nat := 101994
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events398.exact101988RawTerms
def group : MergeGroup := .relation 101990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101990) (rhsResult := 101988)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101989 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩) (none) 101988) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101994

namespace LeftMerge101999
def owner : Owner := ⟨.program ⟨214⟩, ⟨24900⟩⟩
def mergeEvent : Nat := 101999
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact101995RawTerms
def rightRaw : List Term := Proof.Events397.exact101833RawTerms
def group : MergeGroup := .operator 101995 101833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101995) (leftOrdinal := 2)
    (rightResult := 101833) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨22948⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101999

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
