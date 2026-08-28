import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge41896
def owner : Owner := ⟨.program ⟨214⟩, ⟨13798⟩⟩
def mergeEvent : Nat := 41896
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }
def rhsRaw : List Term := Proof.Events048.exact12484RawTerms
def group : MergeGroup := .relation 41895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41895) (rhsResult := 12484)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7846⟩⟩) ⟨6777⟩ 12484) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41896

namespace LeftMerge41897
def owner : Owner := ⟨.program ⟨214⟩, ⟨13798⟩⟩
def mergeEvent : Nat := 41897
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }
def leftRaw : List Term := Proof.Events163.exact41888RawTerms
def rightRaw : List Term := Proof.Events048.exact12514RawTerms
def group : MergeGroup := .operator 41888 12514
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41888) (leftOrdinal := 0)
    (rightResult := 12514) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7846⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41897

namespace LeftMerge41902
def owner : Owner := ⟨.program ⟨214⟩, ⟨13799⟩⟩
def mergeEvent : Nat := 41902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }
def leftRaw : List Term := Proof.Events163.exact41898RawTerms
def rightRaw : List Term := Proof.Events163.exact41868RawTerms
def group : MergeGroup := .operator 41898 41868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41898) (leftOrdinal := 1)
    (rightResult := 41868) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41902

namespace LeftMerge41910
def owner : Owner := ⟨.program ⟨214⟩, ⟨25923⟩⟩
def mergeEvent : Nat := 41910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩] } }
def leftRaw : List Term := Proof.Events163.exact41904RawTerms
def rightRaw : List Term := Proof.Events163.exact41840RawTerms
def group : MergeGroup := .operator 41904 41840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41904) (leftOrdinal := 1)
    (rightResult := 41840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25922⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41910

namespace LeftMerge41912
def owner : Owner := ⟨.program ⟨214⟩, ⟨25923⟩⟩
def mergeEvent : Nat := 41912
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23504⟩⟩] } }
def rhsRaw : List Term := Proof.Events163.exact41837RawTerms
def group : MergeGroup := .relation 41911
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41911) (rhsResult := 41837)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25922⟩⟩) ⟨23504⟩ 41837) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23504⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨23504⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41912

namespace LeftMerge41913
def owner : Owner := ⟨.program ⟨214⟩, ⟨25923⟩⟩
def mergeEvent : Nat := 41913
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩] } }
def leftRaw : List Term := Proof.Events163.exact41904RawTerms
def rightRaw : List Term := Proof.Events163.exact41840RawTerms
def group : MergeGroup := .operator 41904 41840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41904) (leftOrdinal := 0)
    (rightResult := 41840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25922⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41913

namespace LeftMerge41927
def owner : Owner := ⟨.program ⟨214⟩, ⟨19395⟩⟩
def mergeEvent : Nat := 41927
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19392⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events163.exact41921RawTerms
def group : MergeGroup := .operator 36137 41921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 41921) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19392⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19392⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41927

namespace LeftMerge42006
def owner : Owner := ⟨.program ⟨214⟩, ⟨13792⟩⟩
def mergeEvent : Nat := 42006
def frameStart : Nat := 41976
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events164.exact42002RawTerms
def rightRaw : List Term := Proof.Events164.exact41999RawTerms
def group : MergeGroup := .operator 42002 41999
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42002) (leftOrdinal := 0)
    (rightResult := 41999) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11309⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42006

namespace LeftMerge42036
def owner : Owner := ⟨.program ⟨214⟩, ⟨13890⟩⟩
def mergeEvent : Nat := 42036
def frameStart : Nat := 41976
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42032RawTerms
def rightRaw : List Term := Proof.Events164.exact42030RawTerms
def group : MergeGroup := .operator 42032 42030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42032) (leftOrdinal := 0)
    (rightResult := 42030) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42036

namespace LeftMerge42059
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def mergeEvent : Nat := 42059
def frameStart : Nat := 41976
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42055RawTerms
def rightRaw : List Term := Proof.Events164.exact42052RawTerms
def group : MergeGroup := .operator 42055 42052
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42055) (leftOrdinal := 0)
    (rightResult := 42052) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7846⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42059

namespace LeftMerge42068
def owner : Owner := ⟨.program ⟨214⟩, ⟨25925⟩⟩
def mergeEvent : Nat := 42068
def frameStart : Nat := 41976
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42064RawTerms
def rightRaw : List Term := Proof.Events164.exact42021RawTerms
def group : MergeGroup := .operator 42064 42021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42064) (leftOrdinal := 0)
    (rightResult := 42021) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25922⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42068

namespace LeftMerge42069
def owner : Owner := ⟨.program ⟨214⟩, ⟨25925⟩⟩
def mergeEvent : Nat := 42069
def frameStart : Nat := 41976
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42064RawTerms
def rightRaw : List Term := Proof.Events164.exact42021RawTerms
def group : MergeGroup := .operator 42064 42021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42064) (leftOrdinal := 1)
    (rightResult := 42021) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25922⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42069

namespace LeftMerge42071
def owner : Owner := ⟨.program ⟨214⟩, ⟨25925⟩⟩
def mergeEvent : Nat := 42071
def frameStart : Nat := 41976
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23504⟩⟩] } }
def rhsRaw : List Term := Proof.Events164.exact42018RawTerms
def group : MergeGroup := .relation 42070
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42070) (rhsResult := 42018)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25922⟩⟩) ⟨23504⟩ 42018) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23504⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨23504⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42071

namespace LeftMerge42079
def owner : Owner := ⟨.program ⟨214⟩, ⟨15712⟩⟩
def mergeEvent : Nat := 42079
def frameStart : Nat := 41976
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15710⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42032RawTerms
def rightRaw : List Term := Proof.Events164.exact42075RawTerms
def group : MergeGroup := .operator 42032 42075
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42032) (leftOrdinal := 0)
    (rightResult := 42075) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15710⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15710⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42079

namespace LeftMerge42096
def owner : Owner := ⟨.program ⟨214⟩, ⟨19395⟩⟩
def mergeEvent : Nat := 42096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }
def rhsRaw : List Term := Proof.Events164.exact42093RawTerms
def group : MergeGroup := .relation 42095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42095) (rhsResult := 42093)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19392⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 42094 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19392⟩⟩]⟩) (none) 42093) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42096

namespace LeftMerge42097
def owner : Owner := ⟨.program ⟨214⟩, ⟨19395⟩⟩
def mergeEvent : Nat := 42097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩] } }
def rhsRaw : List Term := Proof.Events164.exact42093RawTerms
def group : MergeGroup := .relation 42095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42095) (rhsResult := 42093)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19392⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 42094 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19392⟩⟩]⟩) (none) 42093) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42097

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
