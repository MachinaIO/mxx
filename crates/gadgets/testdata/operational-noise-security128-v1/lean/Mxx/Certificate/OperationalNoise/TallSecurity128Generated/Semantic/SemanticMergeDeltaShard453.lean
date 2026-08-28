import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge76110
def owner : Owner := ⟨.program ⟨257⟩, ⟨49452⟩⟩
def mergeEvent : Nat := 76110
def frameStart : Nat := 76050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76106RawTerms
def rightRaw : List Term := Proof.Events297.exact76104RawTerms
def group : MergeGroup := .operator 76106 76104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76106) (leftOrdinal := 0)
    (rightResult := 76104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76110

namespace LeftMerge76133
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 76133
def frameStart : Nat := 76050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76129RawTerms
def rightRaw : List Term := Proof.Events297.exact76126RawTerms
def group : MergeGroup := .operator 76129 76126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76129) (leftOrdinal := 0)
    (rightResult := 76126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76133

namespace LeftMerge76142
def owner : Owner := ⟨.program ⟨257⟩, ⟨49728⟩⟩
def mergeEvent : Nat := 76142
def frameStart : Nat := 76050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76138RawTerms
def rightRaw : List Term := Proof.Events297.exact76095RawTerms
def group : MergeGroup := .operator 76138 76095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76138) (leftOrdinal := 0)
    (rightResult := 76095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49725⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76142

namespace LeftMerge76143
def owner : Owner := ⟨.program ⟨257⟩, ⟨49728⟩⟩
def mergeEvent : Nat := 76143
def frameStart : Nat := 76050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76138RawTerms
def rightRaw : List Term := Proof.Events297.exact76095RawTerms
def group : MergeGroup := .operator 76138 76095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76138) (leftOrdinal := 1)
    (rightResult := 76095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49725⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge76143

namespace LeftMerge76145
def owner : Owner := ⟨.program ⟨257⟩, ⟨49728⟩⟩
def mergeEvent : Nat := 76145
def frameStart : Nat := 76050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49185⟩⟩] } }
def rhsRaw : List Term := Proof.Events297.exact76092RawTerms
def group : MergeGroup := .relation 76144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 76144) (rhsResult := 76092)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49725⟩⟩) ⟨49185⟩ 76092) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨49185⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge76145

namespace LeftMerge76153
def owner : Owner := ⟨.program ⟨257⟩, ⟨48198⟩⟩
def mergeEvent : Nat := 76153
def frameStart : Nat := 76050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76106RawTerms
def rightRaw : List Term := Proof.Events297.exact76149RawTerms
def group : MergeGroup := .operator 76106 76149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76106) (leftOrdinal := 0)
    (rightResult := 76149) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48196⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76153

namespace LeftMerge76170
def owner : Owner := ⟨.program ⟨257⟩, ⟨48652⟩⟩
def mergeEvent : Nat := 76170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events297.exact76167RawTerms
def group : MergeGroup := .relation 76169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 76169) (rhsResult := 76167)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 76168 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩) (none) 76167) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76170

namespace LeftMerge76171
def owner : Owner := ⟨.program ⟨257⟩, ⟨48652⟩⟩
def mergeEvent : Nat := 76171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩] } }
def rhsRaw : List Term := Proof.Events297.exact76167RawTerms
def group : MergeGroup := .relation 76169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 76169) (rhsResult := 76167)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 76168 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩) (none) 76167) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge76171

namespace LeftMerge76172
def owner : Owner := ⟨.program ⟨257⟩, ⟨48652⟩⟩
def mergeEvent : Nat := 76172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49185⟩⟩] } }
def rhsRaw : List Term := Proof.Events297.exact76167RawTerms
def group : MergeGroup := .relation 76169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 76169) (rhsResult := 76167)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 76168 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩) (none) 76167) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨49185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76172

namespace LeftMerge76173
def owner : Owner := ⟨.program ⟨257⟩, ⟨48652⟩⟩
def mergeEvent : Nat := 76173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events297.exact76167RawTerms
def group : MergeGroup := .relation 76169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 76169) (rhsResult := 76167)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 76168 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩) (none) 76167) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge76173

namespace LeftMerge76178
def owner : Owner := ⟨.program ⟨257⟩, ⟨49727⟩⟩
def mergeEvent : Nat := 76178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49185⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76174RawTerms
def rightRaw : List Term := Proof.Events296.exact75977RawTerms
def group : MergeGroup := .operator 76174 75977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76174) (leftOrdinal := 2)
    (rightResult := 75977) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49185⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49185⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨49185⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge76178

namespace LeftMerge76179
def owner : Owner := ⟨.program ⟨257⟩, ⟨49727⟩⟩
def mergeEvent : Nat := 76179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76174RawTerms
def rightRaw : List Term := Proof.Events296.exact75977RawTerms
def group : MergeGroup := .operator 76174 75977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76174) (leftOrdinal := 1)
    (rightResult := 75977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76179

namespace LeftMerge76187
def owner : Owner := ⟨.program ⟨257⟩, ⟨50181⟩⟩
def mergeEvent : Nat := 76187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76181RawTerms
def rightRaw : List Term := Proof.Events296.exact75888RawTerms
def group : MergeGroup := .operator 76181 75888
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76181) (leftOrdinal := 0)
    (rightResult := 75888) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50179⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76187

namespace LeftMerge76188
def owner : Owner := ⟨.program ⟨257⟩, ⟨50181⟩⟩
def mergeEvent : Nat := 76188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩] } }
def leftRaw : List Term := Proof.Events297.exact76181RawTerms
def rightRaw : List Term := Proof.Events296.exact75888RawTerms
def group : MergeGroup := .operator 76181 75888
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76181) (leftOrdinal := 1)
    (rightResult := 75888) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50179⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge76188

namespace LeftMerge76190
def owner : Owner := ⟨.program ⟨257⟩, ⟨50181⟩⟩
def mergeEvent : Nat := 76190
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49355⟩⟩] } }
def rhsRaw : List Term := Proof.Events296.exact75885RawTerms
def group : MergeGroup := .relation 76189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 76189) (rhsResult := 75885)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50179⟩⟩) ⟨49355⟩ 75885) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49355⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge76190

namespace LeftMerge76204
def owner : Owner := ⟨.program ⟨257⟩, ⟨49019⟩⟩
def mergeEvent : Nat := 76204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49016⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events297.exact76198RawTerms
def group : MergeGroup := .operator 75995 76198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 76198) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49016⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49016⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76204

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
