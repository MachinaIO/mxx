import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge35995
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 35995
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35995

namespace LeftMerge35996
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 35996
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35996

namespace LeftMerge35997
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 35997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35997

namespace LeftMerge35998
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 35998
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35998

namespace LeftMerge35999
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 35999
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35999

namespace LeftMerge36000
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36000
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36000

namespace LeftMerge36001
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36001
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36001

namespace LeftMerge36002
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36002
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36002

namespace LeftMerge36003
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36003
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36003

namespace LeftMerge36004
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36004
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36004

namespace LeftMerge36005
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36005
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36005

namespace LeftMerge36006
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36006

namespace LeftMerge36007
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36007
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36007

namespace LeftMerge36008
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36008

namespace LeftMerge36009
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def mergeEvent : Nat := 36009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35930RawTerms
def rightRaw : List Term := Proof.Events008.exact2300RawTerms
def group : MergeGroup := .operator 35930 2300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35930) (leftOrdinal := 1)
    (rightResult := 2300) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6750⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36009

namespace LeftMerge36044
def owner : Owner := ⟨.program ⟨214⟩, ⟨6569⟩⟩
def mergeEvent : Nat := 36044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events000.exact2RawTerms
def group : MergeGroup := .operator 35915 2
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 2) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36044

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
