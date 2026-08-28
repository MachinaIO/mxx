import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events049

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event12544 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨66261⟩⟩) (.finite 1059)

def event12545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67363⟩⟩) 0 ⟨66261⟩ 12544

def event12546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67363⟩⟩) (.authority (.programFamilyFact))

def exact12547RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67363⟩⟩], []⟩, (1)⟩]

theorem exact12547RawTermsValid :
    exact12547RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12547 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67363⟩⟩) exact12547RawTerms (.finite 18) 12546 .exactZero (none)

def event12548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67364⟩⟩) 0 ⟨67363⟩ 12547

def event12549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67364⟩⟩) 1 ⟨6774⟩ 36

def event12550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67364⟩⟩) (.product (.predecessor 0 12548 .coefficient) (.predecessor 1 12549 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12551 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67364⟩⟩, .operator (⟨12547, 0⟩, ⟨36, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], []⟩, (1)⟩)

def exact12552RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], []⟩, (1)⟩]

theorem exact12552RawTermsValid :
    exact12552RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12552 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67364⟩⟩) exact12552RawTerms (.finite 4222381728938650955397720) 12550 .exactZero (none)

def event12553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48294⟩⟩) 0 ⟨48109⟩ 12079

def event12554 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48294⟩⟩) (.authority (.programFamilyFact))

def exact12555RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48294⟩⟩], []⟩, (1)⟩]

theorem exact12555RawTermsValid :
    exact12555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12555 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48294⟩⟩) exact12555RawTerms (.finite 60) 12554 .exactZero (none)

def event12556 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48295⟩⟩) 0 ⟨48294⟩ 12555

def event12557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48295⟩⟩) 1 ⟨6800⟩ 543

def event12558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48295⟩⟩) (.product (.predecessor 0 12556 .coefficient) (.predecessor 1 12557 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12559 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48295⟩⟩, .operator (⟨12555, 0⟩, ⟨543, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], []⟩, (1)⟩)

def exact12560RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], []⟩, (1)⟩]

theorem exact12560RawTermsValid :
    exact12560RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12560 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48295⟩⟩) exact12560RawTerms (.finite 230731242018505516688400) 12558 .exactZero (none)

def event12561 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45614⟩⟩) 0 ⟨45429⟩ 12102

def event12562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45614⟩⟩) (.authority (.programFamilyFact))

def exact12563RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩, (1)⟩]

theorem exact12563RawTermsValid :
    exact12563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45614⟩⟩) exact12563RawTerms (.finite 58) 12562 .exactZero (none)

def event12564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45615⟩⟩) 0 ⟨45614⟩ 12563

def event12565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45615⟩⟩) 1 ⟨6807⟩ 553

def event12566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45615⟩⟩) (.product (.predecessor 0 12564 .coefficient) (.predecessor 1 12565 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12567 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45615⟩⟩, .operator (⟨12563, 0⟩, ⟨553, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩, (1)⟩)

def exact12568RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩, (1)⟩]

theorem exact12568RawTermsValid :
    exact12568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45615⟩⟩) exact12568RawTerms (.finite 230600885384596756509480) 12566 .exactZero (none)

def event12569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42937⟩⟩) 0 ⟨42749⟩ 12125

def event12570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42937⟩⟩) (.authority (.programFamilyFact))

def exact12571RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩]

theorem exact12571RawTermsValid :
    exact12571RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12571 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42937⟩⟩) exact12571RawTerms (.finite 52) 12570 .exactZero (none)

def event12572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42938⟩⟩) 0 ⟨42937⟩ 12571

def event12573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42938⟩⟩) 1 ⟨6817⟩ 563

def event12574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42938⟩⟩) (.product (.predecessor 0 12572 .coefficient) (.predecessor 1 12573 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12575 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42938⟩⟩, .operator (⟨12571, 0⟩, ⟨563, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩)

def exact12576RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩]

theorem exact12576RawTermsValid :
    exact12576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42938⟩⟩) exact12576RawTerms (.finite 230150786063741980797360) 12574 .exactZero (none)

def event12577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40257⟩⟩) 0 ⟨40069⟩ 12148

def event12578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40257⟩⟩) (.authority (.programFamilyFact))

def exact12579RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩]

theorem exact12579RawTermsValid :
    exact12579RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12579 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40257⟩⟩) exact12579RawTerms (.finite 46) 12578 .exactZero (none)

def event12580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40258⟩⟩) 0 ⟨40257⟩ 12579

def event12581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40258⟩⟩) 1 ⟨6828⟩ 573

def event12582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40258⟩⟩) (.product (.predecessor 0 12580 .coefficient) (.predecessor 1 12581 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12583 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40258⟩⟩, .operator (⟨12579, 0⟩, ⟨573, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩)

def exact12584RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩]

theorem exact12584RawTermsValid :
    exact12584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12584 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40258⟩⟩) exact12584RawTerms (.finite 229585767767349815541720) 12582 .exactZero (none)

def event12585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37574⟩⟩) 0 ⟨37389⟩ 12171

def event12586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37574⟩⟩) (.authority (.programFamilyFact))

def exact12587RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩]

theorem exact12587RawTermsValid :
    exact12587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12587 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37574⟩⟩) exact12587RawTerms (.finite 42) 12586 .exactZero (none)

def event12588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37575⟩⟩) 0 ⟨37574⟩ 12587

def event12589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37575⟩⟩) 1 ⟨6838⟩ 583

def event12590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37575⟩⟩) (.product (.predecessor 0 12588 .coefficient) (.predecessor 1 12589 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12591 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37575⟩⟩, .operator (⟨12587, 0⟩, ⟨583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩)

def exact12592RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩]

theorem exact12592RawTermsValid :
    exact12592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12592 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37575⟩⟩) exact12592RawTerms (.finite 229121489167213617734760) 12590 .exactZero (none)

def event12593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34894⟩⟩) 0 ⟨34709⟩ 12194

def event12594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34894⟩⟩) (.authority (.programFamilyFact))

def exact12595RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩]

theorem exact12595RawTermsValid :
    exact12595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34894⟩⟩) exact12595RawTerms (.finite 40) 12594 .exactZero (none)

def event12596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34895⟩⟩) 0 ⟨34894⟩ 12595

def event12597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34895⟩⟩) 1 ⟨6842⟩ 593

def event12598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34895⟩⟩) (.product (.predecessor 0 12596 .coefficient) (.predecessor 1 12597 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12599 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34895⟩⟩, .operator (⟨12595, 0⟩, ⟨593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩)

def exact12600RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩]

theorem exact12600RawTermsValid :
    exact12600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34895⟩⟩) exact12600RawTerms (.finite 228855378262257504357600) 12598 .exactZero (none)

def event12601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29237⟩⟩) 0 ⟨29049⟩ 12217

def event12602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29237⟩⟩) (.authority (.programFamilyFact))

def exact12603RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩]

theorem exact12603RawTermsValid :
    exact12603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12603 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29237⟩⟩) exact12603RawTerms (.finite 36) 12602 .exactZero (none)

def event12604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29238⟩⟩) 0 ⟨29237⟩ 12603

def event12605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29238⟩⟩) 1 ⟨6857⟩ 603

def event12606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29238⟩⟩) (.product (.predecessor 0 12604 .coefficient) (.predecessor 1 12605 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12607 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29238⟩⟩, .operator (⟨12603, 0⟩, ⟨603, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩)

def exact12608RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩]

theorem exact12608RawTermsValid :
    exact12608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29238⟩⟩) exact12608RawTerms (.finite 228236850212900051643120) 12606 .exactZero (none)

def event12609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26557⟩⟩) 0 ⟨26369⟩ 12240

def event12610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26557⟩⟩) (.authority (.programFamilyFact))

def exact12611RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩]

theorem exact12611RawTermsValid :
    exact12611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26557⟩⟩) exact12611RawTerms (.finite 30) 12610 .exactZero (none)

def event12612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26558⟩⟩) 0 ⟨26557⟩ 12611

def event12613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26558⟩⟩) 1 ⟨6860⟩ 613

def event12614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26558⟩⟩) (.product (.predecessor 0 12612 .coefficient) (.predecessor 1 12613 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12615 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26558⟩⟩, .operator (⟨12611, 0⟩, ⟨613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩)

def exact12616RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩]

theorem exact12616RawTermsValid :
    exact12616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12616 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26558⟩⟩) exact12616RawTerms (.finite 227009770373045750290200) 12614 .exactZero (none)

def event12617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66238⟩⟩) 0 ⟨65749⟩ 12263

def event12618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66238⟩⟩) (.authority (.programFamilyFact))

def exact12619RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12619RawTermsValid :
    exact12619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12619 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66238⟩⟩) exact12619RawTerms (.finite 28) 12618 .exactZero (none)

def event12620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66239⟩⟩) 0 ⟨66238⟩ 12619

def event12621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66239⟩⟩) 1 ⟨6870⟩ 623

def event12622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66239⟩⟩) (.product (.predecessor 0 12620 .coefficient) (.predecessor 1 12621 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨66239⟩⟩, .operator (⟨12619, 0⟩, ⟨623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩)

def exact12624RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12624RawTermsValid :
    exact12624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66239⟩⟩) exact12624RawTerms (.finite 226487908831958288795280) 12622 .exactZero (none)

def event12625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62990⟩⟩) 0 ⟨62769⟩ 12286

def event12626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62990⟩⟩) (.authority (.programFamilyFact))

def exact12627RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩]

theorem exact12627RawTermsValid :
    exact12627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62990⟩⟩) exact12627RawTerms (.finite 22) 12626 .exactZero (none)

def event12628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62991⟩⟩) 0 ⟨62990⟩ 12627

def event12629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62991⟩⟩) 1 ⟨6732⟩ 633

def event12630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62991⟩⟩) (.product (.predecessor 0 12628 .coefficient) (.predecessor 1 12629 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12631 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62991⟩⟩, .operator (⟨12627, 0⟩, ⟨633, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩)

def exact12632RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩]

theorem exact12632RawTermsValid :
    exact12632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62991⟩⟩) exact12632RawTerms (.finite 224377773035387248837560) 12630 .exactZero (none)

def event12633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60010⟩⟩) 0 ⟨59789⟩ 12309

def event12634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60010⟩⟩) (.authority (.programFamilyFact))

def exact12635RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩]

theorem exact12635RawTermsValid :
    exact12635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60010⟩⟩) exact12635RawTerms (.finite 18) 12634 .exactZero (none)

def event12636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60011⟩⟩) 0 ⟨60010⟩ 12635

def event12637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60011⟩⟩) 1 ⟨6736⟩ 643

def event12638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60011⟩⟩) (.product (.predecessor 0 12636 .coefficient) (.predecessor 1 12637 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12639 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60011⟩⟩, .operator (⟨12635, 0⟩, ⟨643, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩)

def exact12640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩]

theorem exact12640RawTermsValid :
    exact12640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60011⟩⟩) exact12640RawTerms (.finite 222230617312560576599880) 12638 .exactZero (none)

def event12641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57030⟩⟩) 0 ⟨56809⟩ 12332

def event12642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57030⟩⟩) (.authority (.programFamilyFact))

def exact12643RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩]

theorem exact12643RawTermsValid :
    exact12643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12643 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57030⟩⟩) exact12643RawTerms (.finite 16) 12642 .exactZero (none)

def event12644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57031⟩⟩) 0 ⟨57030⟩ 12643

def event12645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57031⟩⟩) 1 ⟨6741⟩ 653

def event12646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57031⟩⟩) (.product (.predecessor 0 12644 .coefficient) (.predecessor 1 12645 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12647 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57031⟩⟩, .operator (⟨12643, 0⟩, ⟨653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩)

def exact12648RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩]

theorem exact12648RawTermsValid :
    exact12648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12648 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57031⟩⟩) exact12648RawTerms (.finite 220778129617707239497920) 12646 .exactZero (none)

def event12649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54050⟩⟩) 0 ⟨53829⟩ 12355

def event12650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54050⟩⟩) (.authority (.programFamilyFact))

def exact12651RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩]

theorem exact12651RawTermsValid :
    exact12651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12651 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54050⟩⟩) exact12651RawTerms (.finite 12) 12650 .exactZero (none)

def event12652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54051⟩⟩) 0 ⟨54050⟩ 12651

def event12653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54051⟩⟩) 1 ⟨6757⟩ 663

def event12654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54051⟩⟩) (.product (.predecessor 0 12652 .coefficient) (.predecessor 1 12653 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12655 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54051⟩⟩, .operator (⟨12651, 0⟩, ⟨663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩)

def exact12656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩]

theorem exact12656RawTermsValid :
    exact12656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54051⟩⟩) exact12656RawTerms (.finite 216532396355828254122960) 12654 .exactZero (none)

def event12657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51070⟩⟩) 0 ⟨50849⟩ 12378

def event12658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51070⟩⟩) (.authority (.programFamilyFact))

def exact12659RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩]

theorem exact12659RawTermsValid :
    exact12659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51070⟩⟩) exact12659RawTerms (.finite 10) 12658 .exactZero (none)

def event12660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51071⟩⟩) 0 ⟨51070⟩ 12659

def event12661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51071⟩⟩) 1 ⟨6768⟩ 673

def event12662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51071⟩⟩) (.product (.predecessor 0 12660 .coefficient) (.predecessor 1 12661 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51071⟩⟩, .operator (⟨12659, 0⟩, ⟨673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩)

def exact12664RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩]

theorem exact12664RawTermsValid :
    exact12664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51071⟩⟩) exact12664RawTerms (.finite 213251602471649038151400) 12662 .exactZero (none)

def event12665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32006⟩⟩) 0 ⟨31789⟩ 12401

def event12666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32006⟩⟩) (.authority (.programFamilyFact))

def exact12667RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩]

theorem exact12667RawTermsValid :
    exact12667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32006⟩⟩) exact12667RawTerms (.finite 6) 12666 .exactZero (none)

def event12668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32007⟩⟩) 0 ⟨32006⟩ 12667

def event12669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32007⟩⟩) 1 ⟨6794⟩ 683

def event12670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32007⟩⟩) (.product (.predecessor 0 12668 .coefficient) (.predecessor 1 12669 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12671 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32007⟩⟩, .operator (⟨12667, 0⟩, ⟨683, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩)

def exact12672RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩]

theorem exact12672RawTermsValid :
    exact12672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32007⟩⟩) exact12672RawTerms (.finite 201065796616126235971320) 12670 .exactZero (none)

def event12673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21986⟩⟩) 0 ⟨21769⟩ 12424

def event12674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21986⟩⟩) (.authority (.programFamilyFact))

def exact12675RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩]

theorem exact12675RawTermsValid :
    exact12675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12675 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21986⟩⟩) exact12675RawTerms (.finite 4) 12674 .exactZero (none)

def event12676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21987⟩⟩) 0 ⟨21986⟩ 12675

def event12677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21987⟩⟩) 1 ⟨6822⟩ 693

def event12678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21987⟩⟩) (.product (.predecessor 0 12676 .coefficient) (.predecessor 1 12677 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21987⟩⟩, .operator (⟨12675, 0⟩, ⟨693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩)

def exact12680RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩]

theorem exact12680RawTermsValid :
    exact12680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12680 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21987⟩⟩) exact12680RawTerms (.finite 187661410175051153573232) 12678 .exactZero (none)

def event12681 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18766⟩⟩) 0 ⟨18549⟩ 12447

def event12682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18766⟩⟩) (.authority (.programFamilyFact))

def exact12683RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩]

theorem exact12683RawTermsValid :
    exact12683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18766⟩⟩) exact12683RawTerms (.finite 3) 12682 .exactZero (none)

def event12684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18767⟩⟩) 0 ⟨18766⟩ 12683

def event12685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18767⟩⟩) 1 ⟨6846⟩ 703

def event12686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18767⟩⟩) (.product (.predecessor 0 12684 .coefficient) (.predecessor 1 12685 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18767⟩⟩, .operator (⟨12683, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩)

def exact12688RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩]

theorem exact12688RawTermsValid :
    exact12688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12688 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18767⟩⟩) exact12688RawTerms (.finite 175932572039110456474905) 12686 .exactZero (none)

def event12689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15950⟩⟩) 0 ⟨15749⟩ 12470

def event12690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15950⟩⟩) (.authority (.programFamilyFact))

def exact12691RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12691RawTermsValid :
    exact12691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12691 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15950⟩⟩) exact12691RawTerms (.finite 2) 12690 .exactZero (none)

def event12692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15951⟩⟩) 0 ⟨15950⟩ 12691

def event12693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15951⟩⟩) 1 ⟨6863⟩ 713

def event12694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15951⟩⟩) (.product (.predecessor 0 12692 .coefficient) (.predecessor 1 12693 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15951⟩⟩, .operator (⟨12691, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩)

def exact12696RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12696RawTermsValid :
    exact12696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12696 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15951⟩⟩) exact12696RawTerms (.finite 156384508479209294644360) 12694 .exactZero (none)

def event12697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15952⟩⟩) 0 ⟨6728⟩ 728

def event12698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15952⟩⟩) 1 ⟨15951⟩ 12696

def event12699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15952⟩⟩) (.sum [.predecessor 0 12697 .coefficient, .predecessor 1 12698 .coefficient])

def exact12700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12700RawTermsValid :
    exact12700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15952⟩⟩) exact12700RawTerms (.finite 156384508479209294644360) 12699 .exactZero (none)

def event12701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18768⟩⟩) 0 ⟨15952⟩ 12700

def event12702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18768⟩⟩) 1 ⟨18767⟩ 12688

def event12703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18768⟩⟩) (.sum [.predecessor 0 12701 .coefficient, .predecessor 1 12702 .coefficient])

def exact12704RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12704RawTermsValid :
    exact12704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18768⟩⟩) exact12704RawTerms (.finite 332317080518319751119265) 12703 .exactZero (none)

def event12705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21988⟩⟩) 0 ⟨18768⟩ 12704

def event12706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21988⟩⟩) 1 ⟨21987⟩ 12680

def event12707 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21988⟩⟩) (.sum [.predecessor 0 12705 .coefficient, .predecessor 1 12706 .coefficient])

def exact12708RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12708RawTermsValid :
    exact12708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12708 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21988⟩⟩) exact12708RawTerms (.finite 519978490693370904692497) 12707 .exactZero (none)

def event12709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32008⟩⟩) 0 ⟨21988⟩ 12708

def event12710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32008⟩⟩) 1 ⟨32007⟩ 12672

def event12711 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32008⟩⟩) (.sum [.predecessor 0 12709 .coefficient, .predecessor 1 12710 .coefficient])

def exact12712RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12712RawTermsValid :
    exact12712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12712 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32008⟩⟩) exact12712RawTerms (.finite 721044287309497140663817) 12711 .exactZero (none)

def event12713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51072⟩⟩) 0 ⟨32008⟩ 12712

def event12714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51072⟩⟩) 1 ⟨51071⟩ 12664

def event12715 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51072⟩⟩) (.sum [.predecessor 0 12713 .coefficient, .predecessor 1 12714 .coefficient])

def exact12716RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12716RawTermsValid :
    exact12716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12716 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51072⟩⟩) exact12716RawTerms (.finite 934295889781146178815217) 12715 .exactZero (none)

def event12717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54052⟩⟩) 0 ⟨51072⟩ 12716

def event12718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54052⟩⟩) 1 ⟨54051⟩ 12656

def event12719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54052⟩⟩) (.sum [.predecessor 0 12717 .coefficient, .predecessor 1 12718 .coefficient])

def exact12720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12720RawTermsValid :
    exact12720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54052⟩⟩) exact12720RawTerms (.finite 1150828286136974432938177) 12719 .exactZero (none)

def event12721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57032⟩⟩) 0 ⟨54052⟩ 12720

def event12722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57032⟩⟩) 1 ⟨57031⟩ 12648

def event12723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57032⟩⟩) (.sum [.predecessor 0 12721 .coefficient, .predecessor 1 12722 .coefficient])

def exact12724RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12724RawTermsValid :
    exact12724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57032⟩⟩) exact12724RawTerms (.finite 1371606415754681672436097) 12723 .exactZero (none)

def event12725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60012⟩⟩) 0 ⟨57032⟩ 12724

def event12726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60012⟩⟩) 1 ⟨60011⟩ 12640

def event12727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60012⟩⟩) (.sum [.predecessor 0 12725 .coefficient, .predecessor 1 12726 .coefficient])

def exact12728RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12728RawTermsValid :
    exact12728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60012⟩⟩) exact12728RawTerms (.finite 1593837033067242249035977) 12727 .exactZero (none)

def event12729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62992⟩⟩) 0 ⟨60012⟩ 12728

def event12730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62992⟩⟩) 1 ⟨62991⟩ 12632

def event12731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62992⟩⟩) (.sum [.predecessor 0 12729 .coefficient, .predecessor 1 12730 .coefficient])

def exact12732RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩]

theorem exact12732RawTermsValid :
    exact12732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12732 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62992⟩⟩) exact12732RawTerms (.finite 1818214806102629497873537) 12731 .exactZero (none)

def event12733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66240⟩⟩) 0 ⟨62992⟩ 12732

def event12734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66240⟩⟩) 1 ⟨66239⟩ 12624

def event12735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66240⟩⟩) (.sum [.predecessor 0 12733 .coefficient, .predecessor 1 12734 .coefficient])

def exact12736RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12736RawTermsValid :
    exact12736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66240⟩⟩) exact12736RawTerms (.finite 2044702714934587786668817) 12735 .exactZero (none)

def event12737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66241⟩⟩) 0 ⟨66240⟩ 12736

def event12738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66241⟩⟩) 1 ⟨26558⟩ 12616

def event12739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66241⟩⟩) (.sum [.predecessor 0 12737 .coefficient, .predecessor 1 12738 .coefficient])

def exact12740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12740RawTermsValid :
    exact12740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66241⟩⟩) exact12740RawTerms (.finite 2271712485307633536959017) 12739 .exactZero (none)

def event12741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66242⟩⟩) 0 ⟨66241⟩ 12740

def event12742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66242⟩⟩) 1 ⟨29238⟩ 12608

def event12743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66242⟩⟩) (.sum [.predecessor 0 12741 .coefficient, .predecessor 1 12742 .coefficient])

def exact12744RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12744RawTermsValid :
    exact12744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12744 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66242⟩⟩) exact12744RawTerms (.finite 2499949335520533588602137) 12743 .exactZero (none)

def event12745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66243⟩⟩) 0 ⟨66242⟩ 12744

def event12746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66243⟩⟩) 1 ⟨34895⟩ 12600

def event12747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66243⟩⟩) (.sum [.predecessor 0 12745 .coefficient, .predecessor 1 12746 .coefficient])

def exact12748RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12748RawTermsValid :
    exact12748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12748 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66243⟩⟩) exact12748RawTerms (.finite 2728804713782791092959737) 12747 .exactZero (none)

def event12749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66244⟩⟩) 0 ⟨66243⟩ 12748

def event12750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66244⟩⟩) 1 ⟨37575⟩ 12592

def event12751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66244⟩⟩) (.sum [.predecessor 0 12749 .coefficient, .predecessor 1 12750 .coefficient])

def exact12752RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12752RawTermsValid :
    exact12752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12752 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66244⟩⟩) exact12752RawTerms (.finite 2957926202950004710694497) 12751 .exactZero (none)

def event12753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66245⟩⟩) 0 ⟨66244⟩ 12752

def event12754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66245⟩⟩) 1 ⟨40258⟩ 12584

def event12755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66245⟩⟩) (.sum [.predecessor 0 12753 .coefficient, .predecessor 1 12754 .coefficient])

def exact12756RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12756RawTermsValid :
    exact12756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66245⟩⟩) exact12756RawTerms (.finite 3187511970717354526236217) 12755 .exactZero (none)

def event12757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66246⟩⟩) 0 ⟨66245⟩ 12756

def event12758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66246⟩⟩) 1 ⟨42938⟩ 12576

def event12759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66246⟩⟩) (.sum [.predecessor 0 12757 .coefficient, .predecessor 1 12758 .coefficient])

def exact12760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12760RawTermsValid :
    exact12760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66246⟩⟩) exact12760RawTerms (.finite 3417662756781096507033577) 12759 .exactZero (none)

def event12761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66247⟩⟩) 0 ⟨66246⟩ 12760

def event12762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66247⟩⟩) 1 ⟨45615⟩ 12568

def event12763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66247⟩⟩) (.sum [.predecessor 0 12761 .coefficient, .predecessor 1 12762 .coefficient])

def exact12764RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12764RawTermsValid :
    exact12764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66247⟩⟩) exact12764RawTerms (.finite 3648263642165693263543057) 12763 .exactZero (none)

def event12765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66248⟩⟩) 0 ⟨66247⟩ 12764

def event12766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66248⟩⟩) 1 ⟨48295⟩ 12560

def event12767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66248⟩⟩) (.sum [.predecessor 0 12765 .coefficient, .predecessor 1 12766 .coefficient])

def exact12768RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12768RawTermsValid :
    exact12768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12768 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66248⟩⟩) exact12768RawTerms (.finite 3878994884184198780231457) 12767 .exactZero (none)

def event12769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67366⟩⟩) 0 ⟨66248⟩ 12768

def event12770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67366⟩⟩) 1 ⟨67364⟩ 12552

def event12771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67366⟩⟩) (.sum [.predecessor 0 12769 .coefficient, .predecessor 1 12770 .coefficient])

def exact12772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12772RawTermsValid :
    exact12772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67366⟩⟩) exact12772RawTerms (.finite 8101376613122849735629177) 12771 .exactZero (none)

def event12773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67367⟩⟩) 0 ⟨67366⟩ 12772

def event12774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67367⟩⟩) 1 ⟨6739⟩ 12049

def event12775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67367⟩⟩) (.product (.predecessor 0 12773 .coefficient) (.predecessor 1 12774 .coefficient) (⟨false, true, none, none, some 1⟩))

def event12776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 5⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], []⟩, (-1)⟩)

def event12777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 7⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], []⟩, (1)⟩)

def event12778 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 8⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩, (1)⟩)

def event12779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 9⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩)

def event12780 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 11⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩)

def event12781 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 12⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩)

def event12782 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 13⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩)

def event12783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 15⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩)

def event12784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 16⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩)

def event12785 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 18⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩)

def event12786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 0⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩)

def event12787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 1⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩)

def event12788 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 2⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩)

def event12789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 3⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩)

def event12790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 4⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩)

def event12791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 6⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩)

def event12792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 10⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩)

def event12793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 14⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩)

def event12794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67367⟩⟩, .operator (⟨12772, 17⟩, ⟨12049, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩)

def exact12795RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩, (1)⟩]

theorem exact12795RawTermsValid :
    exact12795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67367⟩⟩) exact12795RawTerms (.finite 337289362800481729748408277773111159218767369973532755787043822584521038233677360097003240996386234053757127521394400923608720140807233890449037750467931042325029210850157327936709881458776973737984) 12775 .exactZero (none)

def event12796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6826⟩⟩) (.authority (.factStore))

def exact12797RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩], []⟩, (1)⟩]

theorem exact12797RawTermsValid :
    exact12797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12797 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6826⟩⟩) exact12797RawTerms (.finite 486143007796660399719643631064108216347957931107960438472841836277675575486982344957071379179671130251989976255561662493972360674322280794500019565719323150507961475646) 12796 .exactZero (none)

def event12798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨387⟩⟩) (.authority (.operator))

def event12799 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨387⟩⟩) (.finite 2)

def eventLeaf784 : Array AnnotatedEvent := #[
  { event := event12544
    frameStart := 0 },
  { event := event12545
    frameStart := 0 },
  { event := event12546
    frameStart := 0 },
  { event := event12547
    frameStart := 0 },
  { event := event12548
    frameStart := 0 },
  { event := event12549
    frameStart := 0 },
  { event := event12550
    frameStart := 0 },
  { event := event12551
    frameStart := 0 },
  { event := event12552
    frameStart := 0 },
  { event := event12553
    frameStart := 0 },
  { event := event12554
    frameStart := 0 },
  { event := event12555
    frameStart := 0 },
  { event := event12556
    frameStart := 0 },
  { event := event12557
    frameStart := 0 },
  { event := event12558
    frameStart := 0 },
  { event := event12559
    frameStart := 0 }
]

def eventLeaf785 : Array AnnotatedEvent := #[
  { event := event12560
    frameStart := 0 },
  { event := event12561
    frameStart := 0 },
  { event := event12562
    frameStart := 0 },
  { event := event12563
    frameStart := 0 },
  { event := event12564
    frameStart := 0 },
  { event := event12565
    frameStart := 0 },
  { event := event12566
    frameStart := 0 },
  { event := event12567
    frameStart := 0 },
  { event := event12568
    frameStart := 0 },
  { event := event12569
    frameStart := 0 },
  { event := event12570
    frameStart := 0 },
  { event := event12571
    frameStart := 0 },
  { event := event12572
    frameStart := 0 },
  { event := event12573
    frameStart := 0 },
  { event := event12574
    frameStart := 0 },
  { event := event12575
    frameStart := 0 }
]

def eventLeaf786 : Array AnnotatedEvent := #[
  { event := event12576
    frameStart := 0 },
  { event := event12577
    frameStart := 0 },
  { event := event12578
    frameStart := 0 },
  { event := event12579
    frameStart := 0 },
  { event := event12580
    frameStart := 0 },
  { event := event12581
    frameStart := 0 },
  { event := event12582
    frameStart := 0 },
  { event := event12583
    frameStart := 0 },
  { event := event12584
    frameStart := 0 },
  { event := event12585
    frameStart := 0 },
  { event := event12586
    frameStart := 0 },
  { event := event12587
    frameStart := 0 },
  { event := event12588
    frameStart := 0 },
  { event := event12589
    frameStart := 0 },
  { event := event12590
    frameStart := 0 },
  { event := event12591
    frameStart := 0 }
]

def eventLeaf787 : Array AnnotatedEvent := #[
  { event := event12592
    frameStart := 0 },
  { event := event12593
    frameStart := 0 },
  { event := event12594
    frameStart := 0 },
  { event := event12595
    frameStart := 0 },
  { event := event12596
    frameStart := 0 },
  { event := event12597
    frameStart := 0 },
  { event := event12598
    frameStart := 0 },
  { event := event12599
    frameStart := 0 },
  { event := event12600
    frameStart := 0 },
  { event := event12601
    frameStart := 0 },
  { event := event12602
    frameStart := 0 },
  { event := event12603
    frameStart := 0 },
  { event := event12604
    frameStart := 0 },
  { event := event12605
    frameStart := 0 },
  { event := event12606
    frameStart := 0 },
  { event := event12607
    frameStart := 0 }
]

def eventLeaf788 : Array AnnotatedEvent := #[
  { event := event12608
    frameStart := 0 },
  { event := event12609
    frameStart := 0 },
  { event := event12610
    frameStart := 0 },
  { event := event12611
    frameStart := 0 },
  { event := event12612
    frameStart := 0 },
  { event := event12613
    frameStart := 0 },
  { event := event12614
    frameStart := 0 },
  { event := event12615
    frameStart := 0 },
  { event := event12616
    frameStart := 0 },
  { event := event12617
    frameStart := 0 },
  { event := event12618
    frameStart := 0 },
  { event := event12619
    frameStart := 0 },
  { event := event12620
    frameStart := 0 },
  { event := event12621
    frameStart := 0 },
  { event := event12622
    frameStart := 0 },
  { event := event12623
    frameStart := 0 }
]

def eventLeaf789 : Array AnnotatedEvent := #[
  { event := event12624
    frameStart := 0 },
  { event := event12625
    frameStart := 0 },
  { event := event12626
    frameStart := 0 },
  { event := event12627
    frameStart := 0 },
  { event := event12628
    frameStart := 0 },
  { event := event12629
    frameStart := 0 },
  { event := event12630
    frameStart := 0 },
  { event := event12631
    frameStart := 0 },
  { event := event12632
    frameStart := 0 },
  { event := event12633
    frameStart := 0 },
  { event := event12634
    frameStart := 0 },
  { event := event12635
    frameStart := 0 },
  { event := event12636
    frameStart := 0 },
  { event := event12637
    frameStart := 0 },
  { event := event12638
    frameStart := 0 },
  { event := event12639
    frameStart := 0 }
]

def eventLeaf790 : Array AnnotatedEvent := #[
  { event := event12640
    frameStart := 0 },
  { event := event12641
    frameStart := 0 },
  { event := event12642
    frameStart := 0 },
  { event := event12643
    frameStart := 0 },
  { event := event12644
    frameStart := 0 },
  { event := event12645
    frameStart := 0 },
  { event := event12646
    frameStart := 0 },
  { event := event12647
    frameStart := 0 },
  { event := event12648
    frameStart := 0 },
  { event := event12649
    frameStart := 0 },
  { event := event12650
    frameStart := 0 },
  { event := event12651
    frameStart := 0 },
  { event := event12652
    frameStart := 0 },
  { event := event12653
    frameStart := 0 },
  { event := event12654
    frameStart := 0 },
  { event := event12655
    frameStart := 0 }
]

def eventLeaf791 : Array AnnotatedEvent := #[
  { event := event12656
    frameStart := 0 },
  { event := event12657
    frameStart := 0 },
  { event := event12658
    frameStart := 0 },
  { event := event12659
    frameStart := 0 },
  { event := event12660
    frameStart := 0 },
  { event := event12661
    frameStart := 0 },
  { event := event12662
    frameStart := 0 },
  { event := event12663
    frameStart := 0 },
  { event := event12664
    frameStart := 0 },
  { event := event12665
    frameStart := 0 },
  { event := event12666
    frameStart := 0 },
  { event := event12667
    frameStart := 0 },
  { event := event12668
    frameStart := 0 },
  { event := event12669
    frameStart := 0 },
  { event := event12670
    frameStart := 0 },
  { event := event12671
    frameStart := 0 }
]

def eventLeaf792 : Array AnnotatedEvent := #[
  { event := event12672
    frameStart := 0 },
  { event := event12673
    frameStart := 0 },
  { event := event12674
    frameStart := 0 },
  { event := event12675
    frameStart := 0 },
  { event := event12676
    frameStart := 0 },
  { event := event12677
    frameStart := 0 },
  { event := event12678
    frameStart := 0 },
  { event := event12679
    frameStart := 0 },
  { event := event12680
    frameStart := 0 },
  { event := event12681
    frameStart := 0 },
  { event := event12682
    frameStart := 0 },
  { event := event12683
    frameStart := 0 },
  { event := event12684
    frameStart := 0 },
  { event := event12685
    frameStart := 0 },
  { event := event12686
    frameStart := 0 },
  { event := event12687
    frameStart := 0 }
]

def eventLeaf793 : Array AnnotatedEvent := #[
  { event := event12688
    frameStart := 0 },
  { event := event12689
    frameStart := 0 },
  { event := event12690
    frameStart := 0 },
  { event := event12691
    frameStart := 0 },
  { event := event12692
    frameStart := 0 },
  { event := event12693
    frameStart := 0 },
  { event := event12694
    frameStart := 0 },
  { event := event12695
    frameStart := 0 },
  { event := event12696
    frameStart := 0 },
  { event := event12697
    frameStart := 0 },
  { event := event12698
    frameStart := 0 },
  { event := event12699
    frameStart := 0 },
  { event := event12700
    frameStart := 0 },
  { event := event12701
    frameStart := 0 },
  { event := event12702
    frameStart := 0 },
  { event := event12703
    frameStart := 0 }
]

def eventLeaf794 : Array AnnotatedEvent := #[
  { event := event12704
    frameStart := 0 },
  { event := event12705
    frameStart := 0 },
  { event := event12706
    frameStart := 0 },
  { event := event12707
    frameStart := 0 },
  { event := event12708
    frameStart := 0 },
  { event := event12709
    frameStart := 0 },
  { event := event12710
    frameStart := 0 },
  { event := event12711
    frameStart := 0 },
  { event := event12712
    frameStart := 0 },
  { event := event12713
    frameStart := 0 },
  { event := event12714
    frameStart := 0 },
  { event := event12715
    frameStart := 0 },
  { event := event12716
    frameStart := 0 },
  { event := event12717
    frameStart := 0 },
  { event := event12718
    frameStart := 0 },
  { event := event12719
    frameStart := 0 }
]

def eventLeaf795 : Array AnnotatedEvent := #[
  { event := event12720
    frameStart := 0 },
  { event := event12721
    frameStart := 0 },
  { event := event12722
    frameStart := 0 },
  { event := event12723
    frameStart := 0 },
  { event := event12724
    frameStart := 0 },
  { event := event12725
    frameStart := 0 },
  { event := event12726
    frameStart := 0 },
  { event := event12727
    frameStart := 0 },
  { event := event12728
    frameStart := 0 },
  { event := event12729
    frameStart := 0 },
  { event := event12730
    frameStart := 0 },
  { event := event12731
    frameStart := 0 },
  { event := event12732
    frameStart := 0 },
  { event := event12733
    frameStart := 0 },
  { event := event12734
    frameStart := 0 },
  { event := event12735
    frameStart := 0 }
]

def eventLeaf796 : Array AnnotatedEvent := #[
  { event := event12736
    frameStart := 0 },
  { event := event12737
    frameStart := 0 },
  { event := event12738
    frameStart := 0 },
  { event := event12739
    frameStart := 0 },
  { event := event12740
    frameStart := 0 },
  { event := event12741
    frameStart := 0 },
  { event := event12742
    frameStart := 0 },
  { event := event12743
    frameStart := 0 },
  { event := event12744
    frameStart := 0 },
  { event := event12745
    frameStart := 0 },
  { event := event12746
    frameStart := 0 },
  { event := event12747
    frameStart := 0 },
  { event := event12748
    frameStart := 0 },
  { event := event12749
    frameStart := 0 },
  { event := event12750
    frameStart := 0 },
  { event := event12751
    frameStart := 0 }
]

def eventLeaf797 : Array AnnotatedEvent := #[
  { event := event12752
    frameStart := 0 },
  { event := event12753
    frameStart := 0 },
  { event := event12754
    frameStart := 0 },
  { event := event12755
    frameStart := 0 },
  { event := event12756
    frameStart := 0 },
  { event := event12757
    frameStart := 0 },
  { event := event12758
    frameStart := 0 },
  { event := event12759
    frameStart := 0 },
  { event := event12760
    frameStart := 0 },
  { event := event12761
    frameStart := 0 },
  { event := event12762
    frameStart := 0 },
  { event := event12763
    frameStart := 0 },
  { event := event12764
    frameStart := 0 },
  { event := event12765
    frameStart := 0 },
  { event := event12766
    frameStart := 0 },
  { event := event12767
    frameStart := 0 }
]

def eventLeaf798 : Array AnnotatedEvent := #[
  { event := event12768
    frameStart := 0 },
  { event := event12769
    frameStart := 0 },
  { event := event12770
    frameStart := 0 },
  { event := event12771
    frameStart := 0 },
  { event := event12772
    frameStart := 0 },
  { event := event12773
    frameStart := 0 },
  { event := event12774
    frameStart := 0 },
  { event := event12775
    frameStart := 0 },
  { event := event12776
    frameStart := 0 },
  { event := event12777
    frameStart := 0 },
  { event := event12778
    frameStart := 0 },
  { event := event12779
    frameStart := 0 },
  { event := event12780
    frameStart := 0 },
  { event := event12781
    frameStart := 0 },
  { event := event12782
    frameStart := 0 },
  { event := event12783
    frameStart := 0 }
]

def eventLeaf799 : Array AnnotatedEvent := #[
  { event := event12784
    frameStart := 0 },
  { event := event12785
    frameStart := 0 },
  { event := event12786
    frameStart := 0 },
  { event := event12787
    frameStart := 0 },
  { event := event12788
    frameStart := 0 },
  { event := event12789
    frameStart := 0 },
  { event := event12790
    frameStart := 0 },
  { event := event12791
    frameStart := 0 },
  { event := event12792
    frameStart := 0 },
  { event := event12793
    frameStart := 0 },
  { event := event12794
    frameStart := 0 },
  { event := event12795
    frameStart := 0 },
  { event := event12796
    frameStart := 0 },
  { event := event12797
    frameStart := 0 },
  { event := event12798
    frameStart := 0 },
  { event := event12799
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events049
