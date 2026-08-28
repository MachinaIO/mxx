import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events323

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event82688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31650⟩⟩, .operator (⟨82682, 1⟩, ⟨3411, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event82689 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31650⟩⟩, .operator (⟨82682, 0⟩, ⟨3411, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def exact82690RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact82690RawTermsValid :
    exact82690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31650⟩⟩) exact82690RawTerms .large 82685 (.finite 5111808) (some (82687))

def event82691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31651⟩⟩) 0 ⟨31647⟩ 3411

def event82692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31651⟩⟩) 1 ⟨10328⟩ 75903

def event82693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31651⟩⟩) (.tensor (.predecessor 0 82691 .coefficient) (.predecessor 1 82692 .coefficient) true false)

def event82694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31651⟩⟩, .operator (⟨3411, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact82695RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82695RawTermsValid :
    exact82695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31651⟩⟩) exact82695RawTerms .large 82693 .exactZero (none)

def event82696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10345⟩⟩) 0 ⟨10327⟩ 75773

def event82697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10345⟩⟩) 1 ⟨7287⟩ 24135

def event82698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10345⟩⟩) (.product (.predecessor 0 82696 .coefficient) (.predecessor 1 82697 .coefficient) (⟨false, false, none, none, none⟩))

def event82699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10345⟩⟩, .operator (⟨75773, 0⟩, ⟨24135, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩)

def exact82700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩]

theorem exact82700RawTermsValid :
    exact82700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10345⟩⟩) exact82700RawTerms .large 82698 .exactZero (none)

def event82701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31652⟩⟩) 0 ⟨10345⟩ 82700

def event82702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31652⟩⟩) 1 ⟨31651⟩ 82695

def event82703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31652⟩⟩) (.sum [.predecessor 0 82701 .coefficient, .predecessor 1 82702 .coefficient])

def exact82704RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82704RawTermsValid :
    exact82704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31652⟩⟩) exact82704RawTerms .large 82703 .exactZero (none)

def event82705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31653⟩⟩) 0 ⟨31652⟩ 82704

def event82706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31653⟩⟩) 1 ⟨113⟩ 24127

def event82707 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31653⟩⟩) (.sum [.predecessor 0 82705 .coefficient, .predecessor 1 82706 .coefficient])

def event82708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31653⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨113⟩⟩]⟩) [⟨.result 24127 .coefficient, false, none⟩])

def event82709 : Event := .survivorFold (1) 82708

def exact82710RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82710RawTermsValid :
    exact82710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82710 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31653⟩⟩) exact82710RawTerms .large 82707 (.finite 26) (some (82708))

def event82711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31654⟩⟩) 0 ⟨31653⟩ 82710

def event82712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31654⟩⟩) 1 ⟨9578⟩ 24124

def event82713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31654⟩⟩) (.product (.predecessor 0 82711 .coefficient) (.predecessor 1 82712 .coefficient) (⟨false, false, none, none, none⟩))

def event82714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31654⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) [⟨.result 24120 .coefficient, false, none⟩])

def event82715 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31654⟩⟩) (.product (.result 82710 .summary) (.transfer 82714) (⟨false, false, none, none, none⟩))

def event82716 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31654⟩⟩, .operator (⟨82710, 1⟩, ⟨24124, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (-1)⟩)

def event82717 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31654⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9577⟩⟩) ⟨7307⟩ 24094)

def event82718 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31654⟩⟩, .relation 82717 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (-1)⟩)

def event82719 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31654⟩⟩, .operator (⟨82710, 0⟩, ⟨24124, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩)

def exact82720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (-1)⟩]

theorem exact82720RawTermsValid :
    exact82720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31654⟩⟩) exact82720RawTerms .large 82713 (.finite 279172874240) (some (82715))

def event82721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31655⟩⟩) 0 ⟨31654⟩ 82720

def event82722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31655⟩⟩) 1 ⟨31650⟩ 82690

def event82723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31655⟩⟩) (.sum [.predecessor 0 82721 .coefficient, .predecessor 1 82722 .coefficient])

def event82724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31655⟩⟩, .operator (⟨82720, 1⟩, ⟨82690, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def event82725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31655⟩⟩) (.sum [.result 82720 .summary, .result 82690 .summary])

def exact82726RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82726RawTermsValid :
    exact82726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31655⟩⟩) exact82726RawTerms .large 82723 (.finite 279177986048) (some (82725))

def event82727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33526⟩⟩) 0 ⟨31655⟩ 82726

def event82728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33526⟩⟩) 1 ⟨33525⟩ 82662

def event82729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33526⟩⟩) (.product (.predecessor 0 82727 .coefficient) (.predecessor 1 82728 .coefficient) (⟨false, false, none, none, none⟩))

def event82730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33526⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩) [⟨.result 82662 .coefficient, false, none⟩])

def event82731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33526⟩⟩) (.product (.result 82726 .summary) (.transfer 82730) (⟨false, false, none, none, none⟩))

def event82732 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33526⟩⟩, .operator (⟨82726, 1⟩, ⟨82662, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (-1)⟩)

def event82733 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33526⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33525⟩⟩) ⟨32985⟩ 82659)

def event82734 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33526⟩⟩, .relation 82733 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (-1)⟩)

def event82735 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33526⟩⟩, .operator (⟨82726, 0⟩, ⟨82662, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (1)⟩)

def exact82736RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (-1)⟩]

theorem exact82736RawTermsValid :
    exact82736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33526⟩⟩) exact82736RawTerms .large 82729 (.finite 2997650799598260715520) (some (82731))

def event82737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32449⟩⟩) 0 ⟨31649⟩ 3419

def event82738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32449⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact82739RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩, (1)⟩]

theorem exact82739RawTermsValid :
    exact82739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82739 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32449⟩⟩) exact82739RawTerms (.finite 5647228698) 82738 .exactZero (none)

def event82740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32451⟩⟩) 0 ⟨32449⟩ 82739

def event82741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32451⟩⟩) 1 ⟨2370⟩ 4

def event82742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32451⟩⟩) (.scale (.predecessor 0 82740 .coefficient) (.value (.predecessor 1 82741 .coefficient)))

def exact82743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩, (1)⟩]

theorem exact82743RawTermsValid :
    exact82743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32451⟩⟩) exact82743RawTerms (.finite 5647228698) 82742 .exactZero (none)

def event82744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32452⟩⟩) 0 ⟨10368⟩ 75995

def event82745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32452⟩⟩) 1 ⟨32451⟩ 82743

def event82746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32452⟩⟩) (.product (.predecessor 0 82744 .coefficient) (.predecessor 1 82745 .coefficient) (⟨false, false, none, none, none⟩))

def event82747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32452⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩) [⟨.result 82739 .coefficient, false, none⟩])

def event82748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32452⟩⟩) (.product (.result 75995 .summary) (.transfer 82747) (⟨false, false, none, none, none⟩))

def event82749 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32452⟩⟩, .operator (⟨75995, 0⟩, ⟨82743, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩, (1)⟩)

def event82750 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨32450⟩⟩)

def event82751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event82752 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event82753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event82754 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event82755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event82756 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event82757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event82758 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event82759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 82758

def event82760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 82756

def event82761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 82759 .coefficient) (.value (.predecessor 1 82760 .coefficient)))

def event82762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event82763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 82762

def event82764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 82754

def event82765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 82763 .coefficient, .predecessor 1 82764 .coefficient])

def event82766 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event82767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 82766

def event82768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 82752

def event82769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 82768 .coefficient))

def event82770 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event82771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24362⟩⟩) 0 ⟨10325⟩ 82770

def event82772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24362⟩⟩) (.authority (.programFamilyFact))

def exact82773RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩], []⟩, (1)⟩]

theorem exact82773RawTermsValid :
    exact82773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24362⟩⟩) exact82773RawTerms (.finite 6) 82772 .exactZero (none)

def event82774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31647⟩⟩) 0 ⟨10325⟩ 82770

def event82775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31647⟩⟩) (.authority (.programFamilyFact))

def exact82776RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩]

theorem exact82776RawTermsValid :
    exact82776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82776 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31647⟩⟩) exact82776RawTerms (.finite 6) 82775 .exactZero (none)

def event82777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31648⟩⟩) 0 ⟨31647⟩ 82776

def event82778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31648⟩⟩) 1 ⟨24362⟩ 82773

def event82779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31648⟩⟩) (.product (.predecessor 0 82777 .coefficient) (.predecessor 1 82778 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31648⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩) [⟨.result 82776 .coefficient, true, some 1⟩, ⟨.result 82773 .coefficient, true, some 1⟩])

def event82781 : Event := .survivorFold (1) 82780

def exact82782RawTerms : List Term := []

theorem exact82782RawTermsValid :
    exact82782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31648⟩⟩) exact82782RawTerms (.finite 36) 82779 (.finite 36) (some (82780))

def event82783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31649⟩⟩) 0 ⟨31648⟩ 82782

def event82784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31649⟩⟩) (.identity (.predecessor 0 82783 .coefficient))

def event82785 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31649⟩⟩) (.finite 36)

def event82786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32449⟩⟩) 0 ⟨31649⟩ 82785

def event82787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32449⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact82788RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩, (1)⟩]

theorem exact82788RawTermsValid :
    exact82788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32449⟩⟩) exact82788RawTerms (.finite 5647228698) 82787 .exactZero (none)

def event82789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact82790RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact82790RawTermsValid :
    exact82790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact82790RawTerms .large 82789 .exactZero (none)

def event82791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32450⟩⟩) 0 ⟨35⟩ 82790

def event82792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32450⟩⟩) 1 ⟨32449⟩ 82788

def event82793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32450⟩⟩) (.product (.predecessor 0 82791 .coefficient) (.predecessor 1 82792 .coefficient) (⟨false, false, none, none, none⟩))

def event82794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32450⟩⟩, .operator (⟨82790, 0⟩, ⟨82788, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩, (1)⟩)

def exact82795RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩, (1)⟩]

theorem exact82795RawTermsValid :
    exact82795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32450⟩⟩) exact82795RawTerms .large 82793 .exactZero (none)

def event82796 : Event := .preFoldPolynomial 82795 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩, (1)⟩] .exactZero none

def exact82797RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩, (1)⟩]

def event82797 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨32450⟩⟩) 82796 exact82797RawTerms .large 82793 .exactZero (none)

def event82798 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨33529⟩⟩)

def event82799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event82800 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event82801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event82802 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event82803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event82804 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event82805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event82806 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event82807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 82806

def event82808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 82804

def event82809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 82807 .coefficient) (.value (.predecessor 1 82808 .coefficient)))

def event82810 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event82811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 82810

def event82812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 82802

def event82813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 82811 .coefficient, .predecessor 1 82812 .coefficient])

def event82814 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event82815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 82814

def event82816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 82800

def event82817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 82816 .coefficient))

def event82818 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event82819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24362⟩⟩) 0 ⟨10325⟩ 82818

def event82820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24362⟩⟩) (.authority (.programFamilyFact))

def exact82821RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩], []⟩, (1)⟩]

theorem exact82821RawTermsValid :
    exact82821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82821 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24362⟩⟩) exact82821RawTerms (.finite 6) 82820 .exactZero (none)

def event82822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31647⟩⟩) 0 ⟨10325⟩ 82818

def event82823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31647⟩⟩) (.authority (.programFamilyFact))

def exact82824RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩]

theorem exact82824RawTermsValid :
    exact82824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82824 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31647⟩⟩) exact82824RawTerms (.finite 6) 82823 .exactZero (none)

def event82825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31648⟩⟩) 0 ⟨31647⟩ 82824

def event82826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31648⟩⟩) 1 ⟨24362⟩ 82821

def event82827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31648⟩⟩) (.product (.predecessor 0 82825 .coefficient) (.predecessor 1 82826 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82828 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31648⟩⟩, .operator (⟨82824, 0⟩, ⟨82821, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩)

def exact82829RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩]

theorem exact82829RawTermsValid :
    exact82829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31648⟩⟩) exact82829RawTerms (.finite 36) 82827 .exactZero (none)

def event82830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31649⟩⟩) 0 ⟨31648⟩ 82829

def event82831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31649⟩⟩) (.identity (.predecessor 0 82830 .coefficient))

def event82832 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31649⟩⟩) (.finite 36)

def event82833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32984⟩⟩) 0 ⟨31649⟩ 82832

def event82834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32984⟩⟩) (.authority (.programFamilyFact))

def event82835 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨32984⟩⟩) (.finite 3720)

def event82836 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event82837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32985⟩⟩) 0 ⟨7177⟩ 82836

def event82838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32985⟩⟩) 1 ⟨32984⟩ 82835

def event82839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32985⟩⟩) (.authority (.operator))

def exact82840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (1)⟩]

theorem exact82840RawTermsValid :
    exact82840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32985⟩⟩) exact82840RawTerms .large 82839 .exactZero (none)

def event82841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33525⟩⟩) 0 ⟨32985⟩ 82840

def event82842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33525⟩⟩) (.authority (.operator))

def exact82843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (1)⟩]

theorem exact82843RawTermsValid :
    exact82843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33525⟩⟩) exact82843RawTerms (.finite 8192) 82842 .exactZero (none)

def event82844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event82845 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event82846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33250⟩⟩) 0 ⟨31649⟩ 82832

def event82847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33250⟩⟩) 1 ⟨136⟩ 82845

def event82848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33250⟩⟩) (.sum [.predecessor 0 82846 .coefficient, .predecessor 1 82847 .coefficient])

def event82849 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33250⟩⟩) (.finite 36)

def event82850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33251⟩⟩) 0 ⟨33250⟩ 82849

def event82851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33251⟩⟩) (.identity (.predecessor 0 82850 .coefficient))

def exact82852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩]

theorem exact82852RawTermsValid :
    exact82852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33251⟩⟩) exact82852RawTerms (.finite 36) 82851 .exactZero (none)

def event82853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact82854RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82854RawTermsValid :
    exact82854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82854 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact82854RawTerms .large 82853 .exactZero (none)

def event82855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33252⟩⟩) 0 ⟨6908⟩ 82854

def event82856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33252⟩⟩) 1 ⟨33251⟩ 82852

def event82857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33252⟩⟩) (.product (.predecessor 0 82855 .coefficient) (.predecessor 1 82856 .coefficient) (⟨false, false, none, none, none⟩))

def event82858 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33252⟩⟩, .operator (⟨82854, 0⟩, ⟨82852, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact82859RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82859RawTermsValid :
    exact82859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33252⟩⟩) exact82859RawTerms .large 82857 .exactZero (none)

def event82860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event82861 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event82862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 82836

def event82863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact82864RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact82864RawTermsValid :
    exact82864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82864 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact82864RawTerms .large 82863 .exactZero (none)

def event82865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7307⟩⟩) 0 ⟨7178⟩ 82864

def event82866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7307⟩⟩) (.identity (.predecessor 0 82865 .coefficient))

def exact82867RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact82867RawTermsValid :
    exact82867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7307⟩⟩) exact82867RawTerms .large 82866 .exactZero (none)

def event82868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9577⟩⟩) 0 ⟨7307⟩ 82867

def event82869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9577⟩⟩) (.authority (.operator))

def exact82870RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact82870RawTermsValid :
    exact82870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9577⟩⟩) exact82870RawTerms (.finite 8192) 82869 .exactZero (none)

def event82871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 0 ⟨9577⟩ 82870

def event82872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 1 ⟨2370⟩ 82861

def event82873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9578⟩⟩) (.scale (.predecessor 0 82871 .coefficient) (.value (.predecessor 1 82872 .coefficient)))

def exact82874RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact82874RawTermsValid :
    exact82874RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82874 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9578⟩⟩) exact82874RawTerms (.finite 8192) 82873 .exactZero (none)

def event82875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7287⟩⟩) 0 ⟨7178⟩ 82864

def event82876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7287⟩⟩) (.identity (.predecessor 0 82875 .coefficient))

def exact82877RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩]

theorem exact82877RawTermsValid :
    exact82877RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82877 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7287⟩⟩) exact82877RawTerms .large 82876 .exactZero (none)

def event82878 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 0 ⟨7287⟩ 82877

def event82879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 1 ⟨9578⟩ 82874

def event82880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9579⟩⟩) (.product (.predecessor 0 82878 .coefficient) (.predecessor 1 82879 .coefficient) (⟨false, false, none, none, none⟩))

def event82881 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9579⟩⟩, .operator (⟨82877, 0⟩, ⟨82874, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩)

def exact82882RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact82882RawTermsValid :
    exact82882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82882 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9579⟩⟩) exact82882RawTerms .large 82880 .exactZero (none)

def event82883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33253⟩⟩) 0 ⟨9579⟩ 82882

def event82884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33253⟩⟩) 1 ⟨33252⟩ 82859

def event82885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33253⟩⟩) (.sum [.predecessor 0 82883 .coefficient, .predecessor 1 82884 .coefficient])

def exact82886RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82886RawTermsValid :
    exact82886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33253⟩⟩) exact82886RawTerms .large 82885 .exactZero (none)

def event82887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33528⟩⟩) 0 ⟨33253⟩ 82886

def event82888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33528⟩⟩) 1 ⟨33525⟩ 82843

def event82889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33528⟩⟩) (.product (.predecessor 0 82887 .coefficient) (.predecessor 1 82888 .coefficient) (⟨false, false, none, none, none⟩))

def event82890 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33528⟩⟩, .operator (⟨82886, 0⟩, ⟨82843, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (1)⟩)

def event82891 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33528⟩⟩, .operator (⟨82886, 1⟩, ⟨82843, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (-1)⟩)

def event82892 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33528⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33525⟩⟩) ⟨32985⟩ 82840)

def event82893 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33528⟩⟩, .relation 82892 0, ⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (-1)⟩)

def exact82894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (-1)⟩]

theorem exact82894RawTermsValid :
    exact82894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82894 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33528⟩⟩) exact82894RawTerms .large 82889 .exactZero (none)

def event82895 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31876⟩⟩) 0 ⟨31649⟩ 82832

def event82896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31876⟩⟩) (.authority (.programFamilyFact))

def exact82897RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], []⟩, (1)⟩]

theorem exact82897RawTermsValid :
    exact82897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82897 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31876⟩⟩) exact82897RawTerms (.finite 6) 82896 .exactZero (none)

def event82898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31878⟩⟩) 0 ⟨6908⟩ 82854

def event82899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31878⟩⟩) 1 ⟨31876⟩ 82897

def event82900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31878⟩⟩) (.product (.predecessor 0 82898 .coefficient) (.predecessor 1 82899 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82901 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31878⟩⟩, .operator (⟨82854, 0⟩, ⟨82897, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact82902RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82902RawTermsValid :
    exact82902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82902 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31878⟩⟩) exact82902RawTerms .large 82900 .exactZero (none)

def event82903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7182⟩⟩) 0 ⟨7177⟩ 82836

def event82904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7182⟩⟩) (.authority (.operator))

def exact82905RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩]

theorem exact82905RawTermsValid :
    exact82905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7182⟩⟩) exact82905RawTerms .large 82904 .exactZero (none)

def event82906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31879⟩⟩) 0 ⟨7182⟩ 82905

def event82907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31879⟩⟩) 1 ⟨31878⟩ 82902

def event82908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31879⟩⟩) (.sum [.predecessor 0 82906 .coefficient, .predecessor 1 82907 .coefficient])

def exact82909RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82909RawTermsValid :
    exact82909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31879⟩⟩) exact82909RawTerms .large 82908 .exactZero (none)

def event82910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33529⟩⟩) 0 ⟨31879⟩ 82909

def event82911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33529⟩⟩) 1 ⟨33528⟩ 82894

def event82912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33529⟩⟩) (.sum [.predecessor 0 82910 .coefficient, .predecessor 1 82911 .coefficient])

def exact82913RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82913RawTermsValid :
    exact82913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33529⟩⟩) exact82913RawTerms .large 82912 .exactZero (none)

def event82914 : Event := .preFoldPolynomial 82913 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact82915RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event82915 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨33529⟩⟩) 82914 exact82915RawTerms .large 82912 .exactZero (none)

def event82916 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨31649⟩⟩) ⟨⟨61⟩, ⟨39⟩, ⟨135⟩⟩ ⟨82750, 82916⟩

def event82917 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨32452⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩) (1) 0 2 (.universal 82916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32449⟩⟩]⟩) (none) 82915)

def event82918 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32452⟩⟩, .relation 82917 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩)

def event82919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32452⟩⟩, .relation 82917 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (-1)⟩)

def event82920 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32452⟩⟩, .relation 82917 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (1)⟩)

def event82921 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32452⟩⟩, .relation 82917 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact82922RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82922RawTermsValid :
    exact82922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32452⟩⟩) exact82922RawTerms .large 82746 (.finite 202072841853861888) (some (82748))

def event82923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33527⟩⟩) 0 ⟨32452⟩ 82922

def event82924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33527⟩⟩) 1 ⟨33526⟩ 82736

def event82925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33527⟩⟩) (.sum [.predecessor 0 82923 .coefficient, .predecessor 1 82924 .coefficient])

def event82926 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33527⟩⟩, .operator (⟨82922, 2⟩, ⟨82736, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (-1)⟩)

def event82927 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33527⟩⟩, .operator (⟨82922, 1⟩, ⟨82736, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (1)⟩)

def event82928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33527⟩⟩) (.sum [.result 82922 .summary, .result 82736 .summary])

def exact82929RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82929RawTermsValid :
    exact82929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82929 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33527⟩⟩) exact82929RawTerms .large 82925 (.finite 2997852872440114577408) (some (82928))

def event82930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34080⟩⟩) 0 ⟨33527⟩ 82929

def event82931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34080⟩⟩) 1 ⟨34078⟩ 82652

def event82932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34080⟩⟩) (.product (.predecessor 0 82930 .coefficient) (.predecessor 1 82931 .coefficient) (⟨false, false, none, none, none⟩))

def event82933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34080⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩) [⟨.result 82652 .coefficient, false, none⟩])

def event82934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34080⟩⟩) (.product (.result 82929 .summary) (.transfer 82933) (⟨false, false, none, none, none⟩))

def event82935 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34080⟩⟩, .operator (⟨82929, 0⟩, ⟨82652, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (1)⟩)

def event82936 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34080⟩⟩, .operator (⟨82929, 1⟩, ⟨82652, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (-1)⟩)

def event82937 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨34080⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34078⟩⟩) ⟨33155⟩ 82649)

def event82938 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34080⟩⟩, .relation 82937 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (-1)⟩)

def exact82939RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (-1)⟩]

theorem exact82939RawTermsValid :
    exact82939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82939 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34080⟩⟩) exact82939RawTerms .large 82932 (.finite 32189200113374879571150551121920) (some (82934))

def event82940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32816⟩⟩) 0 ⟨31877⟩ 3425

def event82941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32816⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact82942RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩, (1)⟩]

theorem exact82942RawTermsValid :
    exact82942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82942 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32816⟩⟩) exact82942RawTerms (.finite 5647228698) 82941 .exactZero (none)

def event82943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32818⟩⟩) 0 ⟨32816⟩ 82942

def eventLeaf5168 : Array AnnotatedEvent := #[
  { event := event82688
    frameStart := 0 },
  { event := event82689
    frameStart := 0 },
  { event := event82690
    frameStart := 0 },
  { event := event82691
    frameStart := 0 },
  { event := event82692
    frameStart := 0 },
  { event := event82693
    frameStart := 0 },
  { event := event82694
    frameStart := 0 },
  { event := event82695
    frameStart := 0 },
  { event := event82696
    frameStart := 0 },
  { event := event82697
    frameStart := 0 },
  { event := event82698
    frameStart := 0 },
  { event := event82699
    frameStart := 0 },
  { event := event82700
    frameStart := 0 },
  { event := event82701
    frameStart := 0 },
  { event := event82702
    frameStart := 0 },
  { event := event82703
    frameStart := 0 }
]

def eventLeaf5169 : Array AnnotatedEvent := #[
  { event := event82704
    frameStart := 0 },
  { event := event82705
    frameStart := 0 },
  { event := event82706
    frameStart := 0 },
  { event := event82707
    frameStart := 0 },
  { event := event82708
    frameStart := 0 },
  { event := event82709
    frameStart := 0 },
  { event := event82710
    frameStart := 0 },
  { event := event82711
    frameStart := 0 },
  { event := event82712
    frameStart := 0 },
  { event := event82713
    frameStart := 0 },
  { event := event82714
    frameStart := 0 },
  { event := event82715
    frameStart := 0 },
  { event := event82716
    frameStart := 0 },
  { event := event82717
    frameStart := 0 },
  { event := event82718
    frameStart := 0 },
  { event := event82719
    frameStart := 0 }
]

def eventLeaf5170 : Array AnnotatedEvent := #[
  { event := event82720
    frameStart := 0 },
  { event := event82721
    frameStart := 0 },
  { event := event82722
    frameStart := 0 },
  { event := event82723
    frameStart := 0 },
  { event := event82724
    frameStart := 0 },
  { event := event82725
    frameStart := 0 },
  { event := event82726
    frameStart := 0 },
  { event := event82727
    frameStart := 0 },
  { event := event82728
    frameStart := 0 },
  { event := event82729
    frameStart := 0 },
  { event := event82730
    frameStart := 0 },
  { event := event82731
    frameStart := 0 },
  { event := event82732
    frameStart := 0 },
  { event := event82733
    frameStart := 0 },
  { event := event82734
    frameStart := 0 },
  { event := event82735
    frameStart := 0 }
]

def eventLeaf5171 : Array AnnotatedEvent := #[
  { event := event82736
    frameStart := 0 },
  { event := event82737
    frameStart := 0 },
  { event := event82738
    frameStart := 0 },
  { event := event82739
    frameStart := 0 },
  { event := event82740
    frameStart := 0 },
  { event := event82741
    frameStart := 0 },
  { event := event82742
    frameStart := 0 },
  { event := event82743
    frameStart := 0 },
  { event := event82744
    frameStart := 0 },
  { event := event82745
    frameStart := 0 },
  { event := event82746
    frameStart := 0 },
  { event := event82747
    frameStart := 0 },
  { event := event82748
    frameStart := 0 },
  { event := event82749
    frameStart := 0 },
  { event := event82750
    frameStart := 82750 },
  { event := event82751
    frameStart := 82750 }
]

def eventLeaf5172 : Array AnnotatedEvent := #[
  { event := event82752
    frameStart := 82750 },
  { event := event82753
    frameStart := 82750 },
  { event := event82754
    frameStart := 82750 },
  { event := event82755
    frameStart := 82750 },
  { event := event82756
    frameStart := 82750 },
  { event := event82757
    frameStart := 82750 },
  { event := event82758
    frameStart := 82750 },
  { event := event82759
    frameStart := 82750 },
  { event := event82760
    frameStart := 82750 },
  { event := event82761
    frameStart := 82750 },
  { event := event82762
    frameStart := 82750 },
  { event := event82763
    frameStart := 82750 },
  { event := event82764
    frameStart := 82750 },
  { event := event82765
    frameStart := 82750 },
  { event := event82766
    frameStart := 82750 },
  { event := event82767
    frameStart := 82750 }
]

def eventLeaf5173 : Array AnnotatedEvent := #[
  { event := event82768
    frameStart := 82750 },
  { event := event82769
    frameStart := 82750 },
  { event := event82770
    frameStart := 82750 },
  { event := event82771
    frameStart := 82750 },
  { event := event82772
    frameStart := 82750 },
  { event := event82773
    frameStart := 82750 },
  { event := event82774
    frameStart := 82750 },
  { event := event82775
    frameStart := 82750 },
  { event := event82776
    frameStart := 82750 },
  { event := event82777
    frameStart := 82750 },
  { event := event82778
    frameStart := 82750 },
  { event := event82779
    frameStart := 82750 },
  { event := event82780
    frameStart := 82750 },
  { event := event82781
    frameStart := 82750 },
  { event := event82782
    frameStart := 82750 },
  { event := event82783
    frameStart := 82750 }
]

def eventLeaf5174 : Array AnnotatedEvent := #[
  { event := event82784
    frameStart := 82750 },
  { event := event82785
    frameStart := 82750 },
  { event := event82786
    frameStart := 82750 },
  { event := event82787
    frameStart := 82750 },
  { event := event82788
    frameStart := 82750 },
  { event := event82789
    frameStart := 82750 },
  { event := event82790
    frameStart := 82750 },
  { event := event82791
    frameStart := 82750 },
  { event := event82792
    frameStart := 82750 },
  { event := event82793
    frameStart := 82750 },
  { event := event82794
    frameStart := 82750 },
  { event := event82795
    frameStart := 82750 },
  { event := event82796
    frameStart := 82750 },
  { event := event82797
    frameStart := 82750 },
  { event := event82798
    frameStart := 82798 },
  { event := event82799
    frameStart := 82798 }
]

def eventLeaf5175 : Array AnnotatedEvent := #[
  { event := event82800
    frameStart := 82798 },
  { event := event82801
    frameStart := 82798 },
  { event := event82802
    frameStart := 82798 },
  { event := event82803
    frameStart := 82798 },
  { event := event82804
    frameStart := 82798 },
  { event := event82805
    frameStart := 82798 },
  { event := event82806
    frameStart := 82798 },
  { event := event82807
    frameStart := 82798 },
  { event := event82808
    frameStart := 82798 },
  { event := event82809
    frameStart := 82798 },
  { event := event82810
    frameStart := 82798 },
  { event := event82811
    frameStart := 82798 },
  { event := event82812
    frameStart := 82798 },
  { event := event82813
    frameStart := 82798 },
  { event := event82814
    frameStart := 82798 },
  { event := event82815
    frameStart := 82798 }
]

def eventLeaf5176 : Array AnnotatedEvent := #[
  { event := event82816
    frameStart := 82798 },
  { event := event82817
    frameStart := 82798 },
  { event := event82818
    frameStart := 82798 },
  { event := event82819
    frameStart := 82798 },
  { event := event82820
    frameStart := 82798 },
  { event := event82821
    frameStart := 82798 },
  { event := event82822
    frameStart := 82798 },
  { event := event82823
    frameStart := 82798 },
  { event := event82824
    frameStart := 82798 },
  { event := event82825
    frameStart := 82798 },
  { event := event82826
    frameStart := 82798 },
  { event := event82827
    frameStart := 82798 },
  { event := event82828
    frameStart := 82798 },
  { event := event82829
    frameStart := 82798 },
  { event := event82830
    frameStart := 82798 },
  { event := event82831
    frameStart := 82798 }
]

def eventLeaf5177 : Array AnnotatedEvent := #[
  { event := event82832
    frameStart := 82798 },
  { event := event82833
    frameStart := 82798 },
  { event := event82834
    frameStart := 82798 },
  { event := event82835
    frameStart := 82798 },
  { event := event82836
    frameStart := 82798 },
  { event := event82837
    frameStart := 82798 },
  { event := event82838
    frameStart := 82798 },
  { event := event82839
    frameStart := 82798 },
  { event := event82840
    frameStart := 82798 },
  { event := event82841
    frameStart := 82798 },
  { event := event82842
    frameStart := 82798 },
  { event := event82843
    frameStart := 82798 },
  { event := event82844
    frameStart := 82798 },
  { event := event82845
    frameStart := 82798 },
  { event := event82846
    frameStart := 82798 },
  { event := event82847
    frameStart := 82798 }
]

def eventLeaf5178 : Array AnnotatedEvent := #[
  { event := event82848
    frameStart := 82798 },
  { event := event82849
    frameStart := 82798 },
  { event := event82850
    frameStart := 82798 },
  { event := event82851
    frameStart := 82798 },
  { event := event82852
    frameStart := 82798 },
  { event := event82853
    frameStart := 82798 },
  { event := event82854
    frameStart := 82798 },
  { event := event82855
    frameStart := 82798 },
  { event := event82856
    frameStart := 82798 },
  { event := event82857
    frameStart := 82798 },
  { event := event82858
    frameStart := 82798 },
  { event := event82859
    frameStart := 82798 },
  { event := event82860
    frameStart := 82798 },
  { event := event82861
    frameStart := 82798 },
  { event := event82862
    frameStart := 82798 },
  { event := event82863
    frameStart := 82798 }
]

def eventLeaf5179 : Array AnnotatedEvent := #[
  { event := event82864
    frameStart := 82798 },
  { event := event82865
    frameStart := 82798 },
  { event := event82866
    frameStart := 82798 },
  { event := event82867
    frameStart := 82798 },
  { event := event82868
    frameStart := 82798 },
  { event := event82869
    frameStart := 82798 },
  { event := event82870
    frameStart := 82798 },
  { event := event82871
    frameStart := 82798 },
  { event := event82872
    frameStart := 82798 },
  { event := event82873
    frameStart := 82798 },
  { event := event82874
    frameStart := 82798 },
  { event := event82875
    frameStart := 82798 },
  { event := event82876
    frameStart := 82798 },
  { event := event82877
    frameStart := 82798 },
  { event := event82878
    frameStart := 82798 },
  { event := event82879
    frameStart := 82798 }
]

def eventLeaf5180 : Array AnnotatedEvent := #[
  { event := event82880
    frameStart := 82798 },
  { event := event82881
    frameStart := 82798 },
  { event := event82882
    frameStart := 82798 },
  { event := event82883
    frameStart := 82798 },
  { event := event82884
    frameStart := 82798 },
  { event := event82885
    frameStart := 82798 },
  { event := event82886
    frameStart := 82798 },
  { event := event82887
    frameStart := 82798 },
  { event := event82888
    frameStart := 82798 },
  { event := event82889
    frameStart := 82798 },
  { event := event82890
    frameStart := 82798 },
  { event := event82891
    frameStart := 82798 },
  { event := event82892
    frameStart := 82798 },
  { event := event82893
    frameStart := 82798 },
  { event := event82894
    frameStart := 82798 },
  { event := event82895
    frameStart := 82798 }
]

def eventLeaf5181 : Array AnnotatedEvent := #[
  { event := event82896
    frameStart := 82798 },
  { event := event82897
    frameStart := 82798 },
  { event := event82898
    frameStart := 82798 },
  { event := event82899
    frameStart := 82798 },
  { event := event82900
    frameStart := 82798 },
  { event := event82901
    frameStart := 82798 },
  { event := event82902
    frameStart := 82798 },
  { event := event82903
    frameStart := 82798 },
  { event := event82904
    frameStart := 82798 },
  { event := event82905
    frameStart := 82798 },
  { event := event82906
    frameStart := 82798 },
  { event := event82907
    frameStart := 82798 },
  { event := event82908
    frameStart := 82798 },
  { event := event82909
    frameStart := 82798 },
  { event := event82910
    frameStart := 82798 },
  { event := event82911
    frameStart := 82798 }
]

def eventLeaf5182 : Array AnnotatedEvent := #[
  { event := event82912
    frameStart := 82798 },
  { event := event82913
    frameStart := 82798 },
  { event := event82914
    frameStart := 82798 },
  { event := event82915
    frameStart := 82798 },
  { event := event82916
    frameStart := 0 },
  { event := event82917
    frameStart := 0 },
  { event := event82918
    frameStart := 0 },
  { event := event82919
    frameStart := 0 },
  { event := event82920
    frameStart := 0 },
  { event := event82921
    frameStart := 0 },
  { event := event82922
    frameStart := 0 },
  { event := event82923
    frameStart := 0 },
  { event := event82924
    frameStart := 0 },
  { event := event82925
    frameStart := 0 },
  { event := event82926
    frameStart := 0 },
  { event := event82927
    frameStart := 0 }
]

def eventLeaf5183 : Array AnnotatedEvent := #[
  { event := event82928
    frameStart := 0 },
  { event := event82929
    frameStart := 0 },
  { event := event82930
    frameStart := 0 },
  { event := event82931
    frameStart := 0 },
  { event := event82932
    frameStart := 0 },
  { event := event82933
    frameStart := 0 },
  { event := event82934
    frameStart := 0 },
  { event := event82935
    frameStart := 0 },
  { event := event82936
    frameStart := 0 },
  { event := event82937
    frameStart := 0 },
  { event := event82938
    frameStart := 0 },
  { event := event82939
    frameStart := 0 },
  { event := event82940
    frameStart := 0 },
  { event := event82941
    frameStart := 0 },
  { event := event82942
    frameStart := 0 },
  { event := event82943
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events323
