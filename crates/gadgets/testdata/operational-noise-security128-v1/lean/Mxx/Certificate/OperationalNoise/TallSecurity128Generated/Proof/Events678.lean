import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events678

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact173568RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], []⟩, (1)⟩]

theorem exact173568RawTermsValid :
    exact173568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66885⟩⟩) exact173568RawTerms (.finite 744) 173567 .exactZero (none)

def event173569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66886⟩⟩) 0 ⟨66885⟩ 173568

def event173570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66886⟩⟩) 1 ⟨37695⟩ 173221

def event173571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66886⟩⟩) (.sum [.predecessor 0 173569 .coefficient, .predecessor 1 173570 .coefficient])

def exact173572RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], []⟩, (1)⟩]

theorem exact173572RawTermsValid :
    exact173572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173572 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66886⟩⟩) exact173572RawTerms (.finite 807) 173571 .exactZero (none)

def event173573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66887⟩⟩) 0 ⟨66886⟩ 173572

def event173574 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66887⟩⟩) 1 ⟨40371⟩ 173198

def event173575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66887⟩⟩) (.sum [.predecessor 0 173573 .coefficient, .predecessor 1 173574 .coefficient])

def exact173576RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], []⟩, (1)⟩]

theorem exact173576RawTermsValid :
    exact173576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66887⟩⟩) exact173576RawTerms (.finite 870) 173575 .exactZero (none)

def event173577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66888⟩⟩) 0 ⟨66887⟩ 173576

def event173578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66888⟩⟩) 1 ⟨43051⟩ 173175

def event173579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66888⟩⟩) (.sum [.predecessor 0 173577 .coefficient, .predecessor 1 173578 .coefficient])

def exact173580RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], []⟩, (1)⟩]

theorem exact173580RawTermsValid :
    exact173580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173580 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66888⟩⟩) exact173580RawTerms (.finite 933) 173579 .exactZero (none)

def event173581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66889⟩⟩) 0 ⟨66888⟩ 173580

def event173582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66889⟩⟩) 1 ⟨45735⟩ 173152

def event173583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66889⟩⟩) (.sum [.predecessor 0 173581 .coefficient, .predecessor 1 173582 .coefficient])

def exact173584RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], []⟩, (1)⟩]

theorem exact173584RawTermsValid :
    exact173584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173584 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66889⟩⟩) exact173584RawTerms (.finite 996) 173583 .exactZero (none)

def event173585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66890⟩⟩) 0 ⟨66889⟩ 173584

def event173586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66890⟩⟩) 1 ⟨48415⟩ 173129

def event173587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66890⟩⟩) (.sum [.predecessor 0 173585 .coefficient, .predecessor 1 173586 .coefficient])

def exact173588RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], []⟩, (1)⟩]

theorem exact173588RawTermsValid :
    exact173588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173588 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66890⟩⟩) exact173588RawTerms (.finite 1059) 173587 .exactZero (none)

def event173589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66891⟩⟩) 0 ⟨66890⟩ 173588

def event173590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66891⟩⟩) (.identity (.predecessor 0 173589 .coefficient))

def event173591 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨66891⟩⟩) (.finite 1059)

def event173592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68853⟩⟩) 0 ⟨66891⟩ 173591

def event173593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68853⟩⟩) (.authority (.programFamilyFact))

def event173594 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68853⟩⟩) (.finite 1152)

def event173595 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event173596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68854⟩⟩) 0 ⟨7177⟩ 173595

def event173597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68854⟩⟩) 1 ⟨68853⟩ 173594

def event173598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68854⟩⟩) (.authority (.operator))

def exact173599RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (1)⟩]

theorem exact173599RawTermsValid :
    exact173599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173599 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68854⟩⟩) exact173599RawTerms .large 173598 .exactZero (none)

def event173600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71365⟩⟩) 0 ⟨68854⟩ 173599

def event173601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71365⟩⟩) (.authority (.operator))

def exact173602RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩]

theorem exact173602RawTermsValid :
    exact173602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71365⟩⟩) exact173602RawTerms (.finite 8192) 173601 .exactZero (none)

def event173603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event173604 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event173605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69103⟩⟩) 0 ⟨66891⟩ 173591

def event173606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69103⟩⟩) 1 ⟨136⟩ 173604

def event173607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69103⟩⟩) (.sum [.predecessor 0 173605 .coefficient, .predecessor 1 173606 .coefficient])

def event173608 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69103⟩⟩) (.finite 1059)

def event173609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69104⟩⟩) 0 ⟨69103⟩ 173608

def event173610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69104⟩⟩) (.identity (.predecessor 0 173609 .coefficient))

def exact173611RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], []⟩, (1)⟩]

theorem exact173611RawTermsValid :
    exact173611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69104⟩⟩) exact173611RawTerms (.finite 1059) 173610 .exactZero (none)

def event173612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact173613RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact173613RawTermsValid :
    exact173613RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173613 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact173613RawTerms .large 173612 .exactZero (none)

def event173614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69105⟩⟩) 0 ⟨6908⟩ 173613

def event173615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69105⟩⟩) 1 ⟨69104⟩ 173611

def event173616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69105⟩⟩) (.product (.predecessor 0 173614 .coefficient) (.predecessor 1 173615 .coefficient) (⟨false, false, none, none, none⟩))

def event173617 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173618 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173619 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173620 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173621 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173622 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173624 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173625 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173626 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173627 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173628 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173629 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173630 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173631 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173632 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173633 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event173634 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69105⟩⟩, .operator (⟨173613, 0⟩, ⟨173611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact173635RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact173635RawTermsValid :
    exact173635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69105⟩⟩) exact173635RawTerms .large 173616 .exactZero (none)

def event173636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 173595

def event173637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact173638RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact173638RawTermsValid :
    exact173638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173638 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact173638RawTerms .large 173637 .exactZero (none)

def event173639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 173595

def event173640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact173641RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact173641RawTermsValid :
    exact173641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173641 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact173641RawTerms .large 173640 .exactZero (none)

def event173642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 173595

def event173643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact173644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact173644RawTermsValid :
    exact173644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173644 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact173644RawTerms .large 173643 .exactZero (none)

def event173645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 173595

def event173646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact173647RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact173647RawTermsValid :
    exact173647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact173647RawTerms .large 173646 .exactZero (none)

def event173648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 173595

def event173649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact173650RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact173650RawTermsValid :
    exact173650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact173650RawTerms .large 173649 .exactZero (none)

def event173651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7222⟩⟩) 0 ⟨7177⟩ 173595

def event173652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7222⟩⟩) (.authority (.operator))

def exact173653RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact173653RawTermsValid :
    exact173653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173653 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7222⟩⟩) exact173653RawTerms .large 173652 .exactZero (none)

def event173654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 173595

def event173655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact173656RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact173656RawTermsValid :
    exact173656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact173656RawTerms .large 173655 .exactZero (none)

def event173657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 173595

def event173658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact173659RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact173659RawTermsValid :
    exact173659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact173659RawTerms .large 173658 .exactZero (none)

def event173660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 173595

def event173661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact173662RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact173662RawTermsValid :
    exact173662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173662 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact173662RawTerms .large 173661 .exactZero (none)

def event173663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7214⟩⟩) 0 ⟨7177⟩ 173595

def event173664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7214⟩⟩) (.authority (.operator))

def exact173665RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact173665RawTermsValid :
    exact173665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173665 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7214⟩⟩) exact173665RawTerms .large 173664 .exactZero (none)

def event173666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 173595

def event173667 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact173668RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact173668RawTermsValid :
    exact173668RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173668 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact173668RawTerms .large 173667 .exactZero (none)

def event173669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 173595

def event173670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact173671RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact173671RawTermsValid :
    exact173671RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173671 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact173671RawTerms .large 173670 .exactZero (none)

def event173672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 173595

def event173673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact173674RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact173674RawTermsValid :
    exact173674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact173674RawTerms .large 173673 .exactZero (none)

def event173675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 173595

def event173676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact173677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact173677RawTermsValid :
    exact173677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact173677RawTerms .large 173676 .exactZero (none)

def event173678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 173595

def event173679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact173680RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact173680RawTermsValid :
    exact173680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173680 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact173680RawTerms .large 173679 .exactZero (none)

def event173681 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7202⟩⟩) 0 ⟨7177⟩ 173595

def event173682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7202⟩⟩) (.authority (.operator))

def exact173683RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact173683RawTermsValid :
    exact173683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7202⟩⟩) exact173683RawTerms .large 173682 .exactZero (none)

def event173684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 173595

def event173685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact173686RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact173686RawTermsValid :
    exact173686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173686 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact173686RawTerms .large 173685 .exactZero (none)

def event173687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 173595

def event173688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact173689RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact173689RawTermsValid :
    exact173689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173689 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact173689RawTerms .large 173688 .exactZero (none)

def event173690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 0 ⟨7198⟩ 173689

def event173691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 1 ⟨7200⟩ 173686

def event173692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7309⟩⟩) (.sum [.predecessor 0 173690 .coefficient, .predecessor 1 173691 .coefficient])

def exact173693RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact173693RawTermsValid :
    exact173693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7309⟩⟩) exact173693RawTerms .large 173692 .exactZero (none)

def event173694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 0 ⟨7309⟩ 173693

def event173695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 1 ⟨7202⟩ 173683

def event173696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7310⟩⟩) (.sum [.predecessor 0 173694 .coefficient, .predecessor 1 173695 .coefficient])

def exact173697RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact173697RawTermsValid :
    exact173697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173697 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7310⟩⟩) exact173697RawTerms .large 173696 .exactZero (none)

def event173698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 0 ⟨7310⟩ 173697

def event173699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 1 ⟨7204⟩ 173680

def event173700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7311⟩⟩) (.sum [.predecessor 0 173698 .coefficient, .predecessor 1 173699 .coefficient])

def exact173701RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact173701RawTermsValid :
    exact173701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7311⟩⟩) exact173701RawTerms .large 173700 .exactZero (none)

def event173702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 0 ⟨7311⟩ 173701

def event173703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 1 ⟨7206⟩ 173677

def event173704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7312⟩⟩) (.sum [.predecessor 0 173702 .coefficient, .predecessor 1 173703 .coefficient])

def exact173705RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact173705RawTermsValid :
    exact173705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173705 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7312⟩⟩) exact173705RawTerms .large 173704 .exactZero (none)

def event173706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 0 ⟨7312⟩ 173705

def event173707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 1 ⟨7208⟩ 173674

def event173708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7313⟩⟩) (.sum [.predecessor 0 173706 .coefficient, .predecessor 1 173707 .coefficient])

def exact173709RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact173709RawTermsValid :
    exact173709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173709 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7313⟩⟩) exact173709RawTerms .large 173708 .exactZero (none)

def event173710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 0 ⟨7313⟩ 173709

def event173711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 1 ⟨7210⟩ 173671

def event173712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7314⟩⟩) (.sum [.predecessor 0 173710 .coefficient, .predecessor 1 173711 .coefficient])

def exact173713RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact173713RawTermsValid :
    exact173713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173713 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7314⟩⟩) exact173713RawTerms .large 173712 .exactZero (none)

def event173714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 0 ⟨7314⟩ 173713

def event173715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 1 ⟨7212⟩ 173668

def event173716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7315⟩⟩) (.sum [.predecessor 0 173714 .coefficient, .predecessor 1 173715 .coefficient])

def exact173717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact173717RawTermsValid :
    exact173717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7315⟩⟩) exact173717RawTerms .large 173716 .exactZero (none)

def event173718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 0 ⟨7315⟩ 173717

def event173719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 1 ⟨7214⟩ 173665

def event173720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7316⟩⟩) (.sum [.predecessor 0 173718 .coefficient, .predecessor 1 173719 .coefficient])

def exact173721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact173721RawTermsValid :
    exact173721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173721 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7316⟩⟩) exact173721RawTerms .large 173720 .exactZero (none)

def event173722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 0 ⟨7316⟩ 173721

def event173723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 1 ⟨7216⟩ 173662

def event173724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7317⟩⟩) (.sum [.predecessor 0 173722 .coefficient, .predecessor 1 173723 .coefficient])

def exact173725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact173725RawTermsValid :
    exact173725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7317⟩⟩) exact173725RawTerms .large 173724 .exactZero (none)

def event173726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 0 ⟨7317⟩ 173725

def event173727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 1 ⟨7218⟩ 173659

def event173728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7318⟩⟩) (.sum [.predecessor 0 173726 .coefficient, .predecessor 1 173727 .coefficient])

def exact173729RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact173729RawTermsValid :
    exact173729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7318⟩⟩) exact173729RawTerms .large 173728 .exactZero (none)

def event173730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 0 ⟨7318⟩ 173729

def event173731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 1 ⟨7220⟩ 173656

def event173732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7319⟩⟩) (.sum [.predecessor 0 173730 .coefficient, .predecessor 1 173731 .coefficient])

def exact173733RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact173733RawTermsValid :
    exact173733RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173733 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7319⟩⟩) exact173733RawTerms .large 173732 .exactZero (none)

def event173734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 0 ⟨7319⟩ 173733

def event173735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 1 ⟨7222⟩ 173653

def event173736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7320⟩⟩) (.sum [.predecessor 0 173734 .coefficient, .predecessor 1 173735 .coefficient])

def exact173737RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact173737RawTermsValid :
    exact173737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173737 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7320⟩⟩) exact173737RawTerms .large 173736 .exactZero (none)

def event173738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 0 ⟨7320⟩ 173737

def event173739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 1 ⟨7224⟩ 173650

def event173740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7321⟩⟩) (.sum [.predecessor 0 173738 .coefficient, .predecessor 1 173739 .coefficient])

def exact173741RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact173741RawTermsValid :
    exact173741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173741 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7321⟩⟩) exact173741RawTerms .large 173740 .exactZero (none)

def event173742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 0 ⟨7321⟩ 173741

def event173743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 1 ⟨7226⟩ 173647

def event173744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7322⟩⟩) (.sum [.predecessor 0 173742 .coefficient, .predecessor 1 173743 .coefficient])

def exact173745RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact173745RawTermsValid :
    exact173745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7322⟩⟩) exact173745RawTerms .large 173744 .exactZero (none)

def event173746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 0 ⟨7322⟩ 173745

def event173747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 1 ⟨7228⟩ 173644

def event173748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7323⟩⟩) (.sum [.predecessor 0 173746 .coefficient, .predecessor 1 173747 .coefficient])

def exact173749RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact173749RawTermsValid :
    exact173749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173749 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7323⟩⟩) exact173749RawTerms .large 173748 .exactZero (none)

def event173750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 0 ⟨7323⟩ 173749

def event173751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 1 ⟨7230⟩ 173641

def event173752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7324⟩⟩) (.sum [.predecessor 0 173750 .coefficient, .predecessor 1 173751 .coefficient])

def exact173753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact173753RawTermsValid :
    exact173753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7324⟩⟩) exact173753RawTerms .large 173752 .exactZero (none)

def event173754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 0 ⟨7324⟩ 173753

def event173755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 1 ⟨7232⟩ 173638

def event173756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7325⟩⟩) (.sum [.predecessor 0 173754 .coefficient, .predecessor 1 173755 .coefficient])

def exact173757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact173757RawTermsValid :
    exact173757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7325⟩⟩) exact173757RawTerms .large 173756 .exactZero (none)

def event173758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69106⟩⟩) 0 ⟨7325⟩ 173757

def event173759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69106⟩⟩) 1 ⟨69105⟩ 173635

def event173760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69106⟩⟩) (.sum [.predecessor 0 173758 .coefficient, .predecessor 1 173759 .coefficient])

def exact173761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact173761RawTermsValid :
    exact173761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event173761 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69106⟩⟩) exact173761RawTerms .large 173760 .exactZero (none)

def event173762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71366⟩⟩) 0 ⟨69106⟩ 173761

def event173763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71366⟩⟩) 1 ⟨71365⟩ 173602

def event173764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71366⟩⟩) (.product (.predecessor 0 173762 .coefficient) (.predecessor 1 173763 .coefficient) (⟨false, false, none, none, none⟩))

def event173765 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 17⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173766 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 16⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173767 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 15⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173768 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 14⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173769 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 13⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173770 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 12⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 11⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173772 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 10⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173773 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 9⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 8⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 7⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 6⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 5⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173778 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 4⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 3⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173780 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 2⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173781 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 1⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173782 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 0⟩, ⟨173602, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event173783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 29⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173784 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173785 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173784 0, ⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 28⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173787 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173788 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173787 0, ⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 27⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173790 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173790 0, ⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 26⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173793 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173793 0, ⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 25⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173796 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173796 0, ⟨[⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173798 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 24⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173799 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173800 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173799 0, ⟨[⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 22⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173802 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173802 0, ⟨[⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 21⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173805 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173805 0, ⟨[⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 35⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173808 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173808 0, ⟨[⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 34⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173811 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173811 0, ⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173813 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 33⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173814 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173815 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173814 0, ⟨[⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173816 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 32⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173817 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173817 0, ⟨[⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173819 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 31⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173820 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def event173821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .relation 173820 0, ⟨[⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event173822 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71366⟩⟩, .operator (⟨173761, 30⟩, ⟨173602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event173823 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71366⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599)

def eventLeaf10848 : Array AnnotatedEvent := #[
  { event := event173568
    frameStart := 173086 },
  { event := event173569
    frameStart := 173086 },
  { event := event173570
    frameStart := 173086 },
  { event := event173571
    frameStart := 173086 },
  { event := event173572
    frameStart := 173086 },
  { event := event173573
    frameStart := 173086 },
  { event := event173574
    frameStart := 173086 },
  { event := event173575
    frameStart := 173086 },
  { event := event173576
    frameStart := 173086 },
  { event := event173577
    frameStart := 173086 },
  { event := event173578
    frameStart := 173086 },
  { event := event173579
    frameStart := 173086 },
  { event := event173580
    frameStart := 173086 },
  { event := event173581
    frameStart := 173086 },
  { event := event173582
    frameStart := 173086 },
  { event := event173583
    frameStart := 173086 }
]

def eventLeaf10849 : Array AnnotatedEvent := #[
  { event := event173584
    frameStart := 173086 },
  { event := event173585
    frameStart := 173086 },
  { event := event173586
    frameStart := 173086 },
  { event := event173587
    frameStart := 173086 },
  { event := event173588
    frameStart := 173086 },
  { event := event173589
    frameStart := 173086 },
  { event := event173590
    frameStart := 173086 },
  { event := event173591
    frameStart := 173086 },
  { event := event173592
    frameStart := 173086 },
  { event := event173593
    frameStart := 173086 },
  { event := event173594
    frameStart := 173086 },
  { event := event173595
    frameStart := 173086 },
  { event := event173596
    frameStart := 173086 },
  { event := event173597
    frameStart := 173086 },
  { event := event173598
    frameStart := 173086 },
  { event := event173599
    frameStart := 173086 }
]

def eventLeaf10850 : Array AnnotatedEvent := #[
  { event := event173600
    frameStart := 173086 },
  { event := event173601
    frameStart := 173086 },
  { event := event173602
    frameStart := 173086 },
  { event := event173603
    frameStart := 173086 },
  { event := event173604
    frameStart := 173086 },
  { event := event173605
    frameStart := 173086 },
  { event := event173606
    frameStart := 173086 },
  { event := event173607
    frameStart := 173086 },
  { event := event173608
    frameStart := 173086 },
  { event := event173609
    frameStart := 173086 },
  { event := event173610
    frameStart := 173086 },
  { event := event173611
    frameStart := 173086 },
  { event := event173612
    frameStart := 173086 },
  { event := event173613
    frameStart := 173086 },
  { event := event173614
    frameStart := 173086 },
  { event := event173615
    frameStart := 173086 }
]

def eventLeaf10851 : Array AnnotatedEvent := #[
  { event := event173616
    frameStart := 173086 },
  { event := event173617
    frameStart := 173086 },
  { event := event173618
    frameStart := 173086 },
  { event := event173619
    frameStart := 173086 },
  { event := event173620
    frameStart := 173086 },
  { event := event173621
    frameStart := 173086 },
  { event := event173622
    frameStart := 173086 },
  { event := event173623
    frameStart := 173086 },
  { event := event173624
    frameStart := 173086 },
  { event := event173625
    frameStart := 173086 },
  { event := event173626
    frameStart := 173086 },
  { event := event173627
    frameStart := 173086 },
  { event := event173628
    frameStart := 173086 },
  { event := event173629
    frameStart := 173086 },
  { event := event173630
    frameStart := 173086 },
  { event := event173631
    frameStart := 173086 }
]

def eventLeaf10852 : Array AnnotatedEvent := #[
  { event := event173632
    frameStart := 173086 },
  { event := event173633
    frameStart := 173086 },
  { event := event173634
    frameStart := 173086 },
  { event := event173635
    frameStart := 173086 },
  { event := event173636
    frameStart := 173086 },
  { event := event173637
    frameStart := 173086 },
  { event := event173638
    frameStart := 173086 },
  { event := event173639
    frameStart := 173086 },
  { event := event173640
    frameStart := 173086 },
  { event := event173641
    frameStart := 173086 },
  { event := event173642
    frameStart := 173086 },
  { event := event173643
    frameStart := 173086 },
  { event := event173644
    frameStart := 173086 },
  { event := event173645
    frameStart := 173086 },
  { event := event173646
    frameStart := 173086 },
  { event := event173647
    frameStart := 173086 }
]

def eventLeaf10853 : Array AnnotatedEvent := #[
  { event := event173648
    frameStart := 173086 },
  { event := event173649
    frameStart := 173086 },
  { event := event173650
    frameStart := 173086 },
  { event := event173651
    frameStart := 173086 },
  { event := event173652
    frameStart := 173086 },
  { event := event173653
    frameStart := 173086 },
  { event := event173654
    frameStart := 173086 },
  { event := event173655
    frameStart := 173086 },
  { event := event173656
    frameStart := 173086 },
  { event := event173657
    frameStart := 173086 },
  { event := event173658
    frameStart := 173086 },
  { event := event173659
    frameStart := 173086 },
  { event := event173660
    frameStart := 173086 },
  { event := event173661
    frameStart := 173086 },
  { event := event173662
    frameStart := 173086 },
  { event := event173663
    frameStart := 173086 }
]

def eventLeaf10854 : Array AnnotatedEvent := #[
  { event := event173664
    frameStart := 173086 },
  { event := event173665
    frameStart := 173086 },
  { event := event173666
    frameStart := 173086 },
  { event := event173667
    frameStart := 173086 },
  { event := event173668
    frameStart := 173086 },
  { event := event173669
    frameStart := 173086 },
  { event := event173670
    frameStart := 173086 },
  { event := event173671
    frameStart := 173086 },
  { event := event173672
    frameStart := 173086 },
  { event := event173673
    frameStart := 173086 },
  { event := event173674
    frameStart := 173086 },
  { event := event173675
    frameStart := 173086 },
  { event := event173676
    frameStart := 173086 },
  { event := event173677
    frameStart := 173086 },
  { event := event173678
    frameStart := 173086 },
  { event := event173679
    frameStart := 173086 }
]

def eventLeaf10855 : Array AnnotatedEvent := #[
  { event := event173680
    frameStart := 173086 },
  { event := event173681
    frameStart := 173086 },
  { event := event173682
    frameStart := 173086 },
  { event := event173683
    frameStart := 173086 },
  { event := event173684
    frameStart := 173086 },
  { event := event173685
    frameStart := 173086 },
  { event := event173686
    frameStart := 173086 },
  { event := event173687
    frameStart := 173086 },
  { event := event173688
    frameStart := 173086 },
  { event := event173689
    frameStart := 173086 },
  { event := event173690
    frameStart := 173086 },
  { event := event173691
    frameStart := 173086 },
  { event := event173692
    frameStart := 173086 },
  { event := event173693
    frameStart := 173086 },
  { event := event173694
    frameStart := 173086 },
  { event := event173695
    frameStart := 173086 }
]

def eventLeaf10856 : Array AnnotatedEvent := #[
  { event := event173696
    frameStart := 173086 },
  { event := event173697
    frameStart := 173086 },
  { event := event173698
    frameStart := 173086 },
  { event := event173699
    frameStart := 173086 },
  { event := event173700
    frameStart := 173086 },
  { event := event173701
    frameStart := 173086 },
  { event := event173702
    frameStart := 173086 },
  { event := event173703
    frameStart := 173086 },
  { event := event173704
    frameStart := 173086 },
  { event := event173705
    frameStart := 173086 },
  { event := event173706
    frameStart := 173086 },
  { event := event173707
    frameStart := 173086 },
  { event := event173708
    frameStart := 173086 },
  { event := event173709
    frameStart := 173086 },
  { event := event173710
    frameStart := 173086 },
  { event := event173711
    frameStart := 173086 }
]

def eventLeaf10857 : Array AnnotatedEvent := #[
  { event := event173712
    frameStart := 173086 },
  { event := event173713
    frameStart := 173086 },
  { event := event173714
    frameStart := 173086 },
  { event := event173715
    frameStart := 173086 },
  { event := event173716
    frameStart := 173086 },
  { event := event173717
    frameStart := 173086 },
  { event := event173718
    frameStart := 173086 },
  { event := event173719
    frameStart := 173086 },
  { event := event173720
    frameStart := 173086 },
  { event := event173721
    frameStart := 173086 },
  { event := event173722
    frameStart := 173086 },
  { event := event173723
    frameStart := 173086 },
  { event := event173724
    frameStart := 173086 },
  { event := event173725
    frameStart := 173086 },
  { event := event173726
    frameStart := 173086 },
  { event := event173727
    frameStart := 173086 }
]

def eventLeaf10858 : Array AnnotatedEvent := #[
  { event := event173728
    frameStart := 173086 },
  { event := event173729
    frameStart := 173086 },
  { event := event173730
    frameStart := 173086 },
  { event := event173731
    frameStart := 173086 },
  { event := event173732
    frameStart := 173086 },
  { event := event173733
    frameStart := 173086 },
  { event := event173734
    frameStart := 173086 },
  { event := event173735
    frameStart := 173086 },
  { event := event173736
    frameStart := 173086 },
  { event := event173737
    frameStart := 173086 },
  { event := event173738
    frameStart := 173086 },
  { event := event173739
    frameStart := 173086 },
  { event := event173740
    frameStart := 173086 },
  { event := event173741
    frameStart := 173086 },
  { event := event173742
    frameStart := 173086 },
  { event := event173743
    frameStart := 173086 }
]

def eventLeaf10859 : Array AnnotatedEvent := #[
  { event := event173744
    frameStart := 173086 },
  { event := event173745
    frameStart := 173086 },
  { event := event173746
    frameStart := 173086 },
  { event := event173747
    frameStart := 173086 },
  { event := event173748
    frameStart := 173086 },
  { event := event173749
    frameStart := 173086 },
  { event := event173750
    frameStart := 173086 },
  { event := event173751
    frameStart := 173086 },
  { event := event173752
    frameStart := 173086 },
  { event := event173753
    frameStart := 173086 },
  { event := event173754
    frameStart := 173086 },
  { event := event173755
    frameStart := 173086 },
  { event := event173756
    frameStart := 173086 },
  { event := event173757
    frameStart := 173086 },
  { event := event173758
    frameStart := 173086 },
  { event := event173759
    frameStart := 173086 }
]

def eventLeaf10860 : Array AnnotatedEvent := #[
  { event := event173760
    frameStart := 173086 },
  { event := event173761
    frameStart := 173086 },
  { event := event173762
    frameStart := 173086 },
  { event := event173763
    frameStart := 173086 },
  { event := event173764
    frameStart := 173086 },
  { event := event173765
    frameStart := 173086 },
  { event := event173766
    frameStart := 173086 },
  { event := event173767
    frameStart := 173086 },
  { event := event173768
    frameStart := 173086 },
  { event := event173769
    frameStart := 173086 },
  { event := event173770
    frameStart := 173086 },
  { event := event173771
    frameStart := 173086 },
  { event := event173772
    frameStart := 173086 },
  { event := event173773
    frameStart := 173086 },
  { event := event173774
    frameStart := 173086 },
  { event := event173775
    frameStart := 173086 }
]

def eventLeaf10861 : Array AnnotatedEvent := #[
  { event := event173776
    frameStart := 173086 },
  { event := event173777
    frameStart := 173086 },
  { event := event173778
    frameStart := 173086 },
  { event := event173779
    frameStart := 173086 },
  { event := event173780
    frameStart := 173086 },
  { event := event173781
    frameStart := 173086 },
  { event := event173782
    frameStart := 173086 },
  { event := event173783
    frameStart := 173086 },
  { event := event173784
    frameStart := 173086 },
  { event := event173785
    frameStart := 173086 },
  { event := event173786
    frameStart := 173086 },
  { event := event173787
    frameStart := 173086 },
  { event := event173788
    frameStart := 173086 },
  { event := event173789
    frameStart := 173086 },
  { event := event173790
    frameStart := 173086 },
  { event := event173791
    frameStart := 173086 }
]

def eventLeaf10862 : Array AnnotatedEvent := #[
  { event := event173792
    frameStart := 173086 },
  { event := event173793
    frameStart := 173086 },
  { event := event173794
    frameStart := 173086 },
  { event := event173795
    frameStart := 173086 },
  { event := event173796
    frameStart := 173086 },
  { event := event173797
    frameStart := 173086 },
  { event := event173798
    frameStart := 173086 },
  { event := event173799
    frameStart := 173086 },
  { event := event173800
    frameStart := 173086 },
  { event := event173801
    frameStart := 173086 },
  { event := event173802
    frameStart := 173086 },
  { event := event173803
    frameStart := 173086 },
  { event := event173804
    frameStart := 173086 },
  { event := event173805
    frameStart := 173086 },
  { event := event173806
    frameStart := 173086 },
  { event := event173807
    frameStart := 173086 }
]

def eventLeaf10863 : Array AnnotatedEvent := #[
  { event := event173808
    frameStart := 173086 },
  { event := event173809
    frameStart := 173086 },
  { event := event173810
    frameStart := 173086 },
  { event := event173811
    frameStart := 173086 },
  { event := event173812
    frameStart := 173086 },
  { event := event173813
    frameStart := 173086 },
  { event := event173814
    frameStart := 173086 },
  { event := event173815
    frameStart := 173086 },
  { event := event173816
    frameStart := 173086 },
  { event := event173817
    frameStart := 173086 },
  { event := event173818
    frameStart := 173086 },
  { event := event173819
    frameStart := 173086 },
  { event := event173820
    frameStart := 173086 },
  { event := event173821
    frameStart := 173086 },
  { event := event173822
    frameStart := 173086 },
  { event := event173823
    frameStart := 173086 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events678
