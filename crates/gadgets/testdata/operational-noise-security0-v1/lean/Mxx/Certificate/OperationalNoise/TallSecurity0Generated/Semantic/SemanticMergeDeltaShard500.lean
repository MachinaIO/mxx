import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge81128
def owner : Owner := ⟨.program ⟨214⟩, ⟨16754⟩⟩
def mergeEvent : Nat := 81128
def frameStart : Nat := 81027
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact81083RawTerms
def rightRaw : List Term := Proof.Events316.exact81124RawTerms
def group : MergeGroup := .operator 81083 81124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81083) (leftOrdinal := 0)
    (rightResult := 81124) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81128

namespace LeftMerge81145
def owner : Owner := ⟨.program ⟨214⟩, ⟨20107⟩⟩
def mergeEvent : Nat := 81145
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact81142RawTerms
def group : MergeGroup := .relation 81144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81144) (rhsResult := 81142)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81143 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩) (none) 81142) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81145

namespace LeftMerge81146
def owner : Owner := ⟨.program ⟨214⟩, ⟨20107⟩⟩
def mergeEvent : Nat := 81146
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact81142RawTerms
def group : MergeGroup := .relation 81144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81144) (rhsResult := 81142)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81143 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩) (none) 81142) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81146

namespace LeftMerge81147
def owner : Owner := ⟨.program ⟨214⟩, ⟨20107⟩⟩
def mergeEvent : Nat := 81147
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23332⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact81142RawTerms
def group : MergeGroup := .relation 81144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81144) (rhsResult := 81142)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81143 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩) (none) 81142) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23332⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81147

namespace LeftMerge81148
def owner : Owner := ⟨.program ⟨214⟩, ⟨20107⟩⟩
def mergeEvent : Nat := 81148
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact81142RawTerms
def group : MergeGroup := .relation 81144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81144) (rhsResult := 81142)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81143 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩) (none) 81142) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81148

namespace LeftMerge81153
def owner : Owner := ⟨.program ⟨214⟩, ⟨25606⟩⟩
def mergeEvent : Nat := 81153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23332⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact81149RawTerms
def rightRaw : List Term := Proof.Events316.exact80965RawTerms
def group : MergeGroup := .operator 81149 80965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81149) (leftOrdinal := 2)
    (rightResult := 80965) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23332⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23332⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81153

namespace LeftMerge81154
def owner : Owner := ⟨.program ⟨214⟩, ⟨25606⟩⟩
def mergeEvent : Nat := 81154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact81149RawTerms
def rightRaw : List Term := Proof.Events316.exact80965RawTerms
def group : MergeGroup := .operator 81149 80965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81149) (leftOrdinal := 1)
    (rightResult := 80965) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81154

namespace LeftMerge81162
def owner : Owner := ⟨.program ⟨214⟩, ⟨29604⟩⟩
def mergeEvent : Nat := 81162
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩] } }
def leftRaw : List Term := Proof.Events317.exact81156RawTerms
def rightRaw : List Term := Proof.Events315.exact80881RawTerms
def group : MergeGroup := .operator 81156 80881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81156) (leftOrdinal := 0)
    (rightResult := 80881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29602⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81162

namespace LeftMerge81163
def owner : Owner := ⟨.program ⟨214⟩, ⟨29604⟩⟩
def mergeEvent : Nat := 81163
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩] } }
def leftRaw : List Term := Proof.Events317.exact81156RawTerms
def rightRaw : List Term := Proof.Events315.exact80881RawTerms
def group : MergeGroup := .operator 81156 80881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81156) (leftOrdinal := 1)
    (rightResult := 80881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29602⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81163

namespace LeftMerge81165
def owner : Owner := ⟨.program ⟨214⟩, ⟨29604⟩⟩
def mergeEvent : Nat := 81165
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24666⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80878RawTerms
def group : MergeGroup := .relation 81164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81164) (rhsResult := 80878)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29602⟩⟩) ⟨24666⟩ 80878) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24666⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24666⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81165

namespace LeftMerge81179
def owner : Owner := ⟨.program ⟨214⟩, ⟨22555⟩⟩
def mergeEvent : Nat := 81179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22552⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events317.exact81173RawTerms
def group : MergeGroup := .operator 80012 81173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 81173) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22552⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22552⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81179

namespace LeftMerge81300
def owner : Owner := ⟨.program ⟨214⟩, ⟨16829⟩⟩
def mergeEvent : Nat := 81300
def frameStart : Nat := 81234
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events317.exact81296RawTerms
def rightRaw : List Term := Proof.Events317.exact81294RawTerms
def group : MergeGroup := .operator 81296 81294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81296) (leftOrdinal := 0)
    (rightResult := 81294) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81300

namespace LeftMerge81312
def owner : Owner := ⟨.program ⟨214⟩, ⟨29603⟩⟩
def mergeEvent : Nat := 81312
def frameStart : Nat := 81234
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩] } }
def leftRaw : List Term := Proof.Events317.exact81308RawTerms
def rightRaw : List Term := Proof.Events317.exact81285RawTerms
def group : MergeGroup := .operator 81308 81285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81308) (leftOrdinal := 0)
    (rightResult := 81285) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29602⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81312

namespace LeftMerge81313
def owner : Owner := ⟨.program ⟨214⟩, ⟨29603⟩⟩
def mergeEvent : Nat := 81313
def frameStart : Nat := 81234
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩] } }
def leftRaw : List Term := Proof.Events317.exact81308RawTerms
def rightRaw : List Term := Proof.Events317.exact81285RawTerms
def group : MergeGroup := .operator 81308 81285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81308) (leftOrdinal := 1)
    (rightResult := 81285) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29602⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81313

namespace LeftMerge81315
def owner : Owner := ⟨.program ⟨214⟩, ⟨29603⟩⟩
def mergeEvent : Nat := 81315
def frameStart : Nat := 81234
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24666⟩⟩] } }
def rhsRaw : List Term := Proof.Events317.exact81282RawTerms
def group : MergeGroup := .relation 81314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81314) (rhsResult := 81282)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29602⟩⟩) ⟨24666⟩ 81282) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24666⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24666⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81315

namespace LeftMerge81323
def owner : Owner := ⟨.program ⟨214⟩, ⟨16799⟩⟩
def mergeEvent : Nat := 81323
def frameStart : Nat := 81234
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events317.exact81296RawTerms
def rightRaw : List Term := Proof.Events317.exact81319RawTerms
def group : MergeGroup := .operator 81296 81319
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81296) (leftOrdinal := 0)
    (rightResult := 81319) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81323

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
