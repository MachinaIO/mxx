import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge62901
def owner : Owner := ⟨.program ⟨257⟩, ⟨39963⟩⟩
def mergeEvent : Nat := 62901
def frameStart : Nat := 62871
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events245.exact62897RawTerms
def rightRaw : List Term := Proof.Events245.exact62894RawTerms
def group : MergeGroup := .operator 62897 62894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62897) (leftOrdinal := 0)
    (rightResult := 62894) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14286⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62901

namespace LeftMerge62931
def owner : Owner := ⟨.program ⟨257⟩, ⟨41416⟩⟩
def mergeEvent : Nat := 62931
def frameStart : Nat := 62871
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62927RawTerms
def rightRaw : List Term := Proof.Events245.exact62925RawTerms
def group : MergeGroup := .operator 62927 62925
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62927) (leftOrdinal := 0)
    (rightResult := 62925) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62931

namespace LeftMerge62954
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 62954
def frameStart : Nat := 62871
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62950RawTerms
def rightRaw : List Term := Proof.Events245.exact62947RawTerms
def group : MergeGroup := .operator 62950 62947
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62950) (leftOrdinal := 0)
    (rightResult := 62947) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62954

namespace LeftMerge62963
def owner : Owner := ⟨.program ⟨257⟩, ⟨41699⟩⟩
def mergeEvent : Nat := 62963
def frameStart : Nat := 62871
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62959RawTerms
def rightRaw : List Term := Proof.Events245.exact62916RawTerms
def group : MergeGroup := .operator 62959 62916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62959) (leftOrdinal := 0)
    (rightResult := 62916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41696⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62963

namespace LeftMerge62964
def owner : Owner := ⟨.program ⟨257⟩, ⟨41699⟩⟩
def mergeEvent : Nat := 62964
def frameStart : Nat := 62871
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62959RawTerms
def rightRaw : List Term := Proof.Events245.exact62916RawTerms
def group : MergeGroup := .operator 62959 62916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62959) (leftOrdinal := 1)
    (rightResult := 62916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62964

namespace LeftMerge62966
def owner : Owner := ⟨.program ⟨257⟩, ⟨41699⟩⟩
def mergeEvent : Nat := 62966
def frameStart : Nat := 62871
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41151⟩⟩] } }
def rhsRaw : List Term := Proof.Events245.exact62913RawTerms
def group : MergeGroup := .relation 62965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62965) (rhsResult := 62913)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41696⟩⟩) ⟨41151⟩ 62913) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41151⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], [⟨.program ⟨257⟩, ⟨41151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62966

namespace LeftMerge62974
def owner : Owner := ⟨.program ⟨257⟩, ⟨40166⟩⟩
def mergeEvent : Nat := 62974
def frameStart : Nat := 62871
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40164⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events245.exact62927RawTerms
def rightRaw : List Term := Proof.Events245.exact62970RawTerms
def group : MergeGroup := .operator 62927 62970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62927) (leftOrdinal := 0)
    (rightResult := 62970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40164⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62974

namespace LeftMerge62991
def owner : Owner := ⟨.program ⟨257⟩, ⟨40622⟩⟩
def mergeEvent : Nat := 62991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events246.exact62988RawTerms
def group : MergeGroup := .relation 62990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62990) (rhsResult := 62988)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62989 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40619⟩⟩]⟩) (none) 62988) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62991

namespace LeftMerge62992
def owner : Owner := ⟨.program ⟨257⟩, ⟨40622⟩⟩
def mergeEvent : Nat := 62992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩] } }
def rhsRaw : List Term := Proof.Events246.exact62988RawTerms
def group : MergeGroup := .relation 62990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62990) (rhsResult := 62988)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62989 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40619⟩⟩]⟩) (none) 62988) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62992

namespace LeftMerge62993
def owner : Owner := ⟨.program ⟨257⟩, ⟨40622⟩⟩
def mergeEvent : Nat := 62993
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41151⟩⟩] } }
def rhsRaw : List Term := Proof.Events246.exact62988RawTerms
def group : MergeGroup := .relation 62990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62990) (rhsResult := 62988)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62989 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40619⟩⟩]⟩) (none) 62988) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41151⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], [⟨.program ⟨257⟩, ⟨41151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62993

namespace LeftMerge62994
def owner : Owner := ⟨.program ⟨257⟩, ⟨40622⟩⟩
def mergeEvent : Nat := 62994
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40164⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events246.exact62988RawTerms
def group : MergeGroup := .relation 62990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62990) (rhsResult := 62988)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62989 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40619⟩⟩]⟩) (none) 62988) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40164⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62994

namespace LeftMerge62999
def owner : Owner := ⟨.program ⟨257⟩, ⟨41698⟩⟩
def mergeEvent : Nat := 62999
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41151⟩⟩] } }
def leftRaw : List Term := Proof.Events246.exact62995RawTerms
def rightRaw : List Term := Proof.Events245.exact62809RawTerms
def group : MergeGroup := .operator 62995 62809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62995) (leftOrdinal := 2)
    (rightResult := 62809) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41151⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], [⟨.program ⟨257⟩, ⟨41151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62999

namespace LeftMerge63000
def owner : Owner := ⟨.program ⟨257⟩, ⟨41698⟩⟩
def mergeEvent : Nat := 63000
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩] } }
def leftRaw : List Term := Proof.Events246.exact62995RawTerms
def rightRaw : List Term := Proof.Events245.exact62809RawTerms
def group : MergeGroup := .operator 62995 62809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62995) (leftOrdinal := 1)
    (rightResult := 62809) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63000

namespace LeftMerge63008
def owner : Owner := ⟨.program ⟨257⟩, ⟨42166⟩⟩
def mergeEvent : Nat := 63008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42164⟩⟩] } }
def leftRaw : List Term := Proof.Events246.exact63002RawTerms
def rightRaw : List Term := Proof.Events245.exact62725RawTerms
def group : MergeGroup := .operator 63002 62725
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63002) (leftOrdinal := 0)
    (rightResult := 62725) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42164⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42164⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63008

namespace LeftMerge63009
def owner : Owner := ⟨.program ⟨257⟩, ⟨42166⟩⟩
def mergeEvent : Nat := 63009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40164⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42164⟩⟩] } }
def leftRaw : List Term := Proof.Events246.exact63002RawTerms
def rightRaw : List Term := Proof.Events245.exact62725RawTerms
def group : MergeGroup := .operator 63002 62725
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63002) (leftOrdinal := 1)
    (rightResult := 62725) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40164⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42164⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63009

namespace LeftMerge63011
def owner : Owner := ⟨.program ⟨257⟩, ⟨42166⟩⟩
def mergeEvent : Nat := 63011
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40164⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41324⟩⟩] } }
def rhsRaw : List Term := Proof.Events245.exact62722RawTerms
def group : MergeGroup := .relation 63010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63010) (rhsResult := 62722)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42164⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42164⟩⟩) ⟨41324⟩ 62722) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41324⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40164⟩⟩], [⟨.program ⟨257⟩, ⟨41324⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63011

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
