import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1135

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event290560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68794⟩⟩) 0 ⟨7177⟩ 290559

def event290561 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68794⟩⟩) 1 ⟨68793⟩ 290558

def event290562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68794⟩⟩) (.authority (.operator))

def exact290563RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩]

theorem exact290563RawTermsValid :
    exact290563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68794⟩⟩) exact290563RawTerms .large 290562 .exactZero (none)

def event290564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71048⟩⟩) 0 ⟨68794⟩ 290563

def event290565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71048⟩⟩) (.authority (.operator))

def exact290566RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩]

theorem exact290566RawTermsValid :
    exact290566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290566 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71048⟩⟩) exact290566RawTerms (.finite 8192) 290565 .exactZero (none)

def event290567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event290568 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event290569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69063⟩⟩) 0 ⟨66191⟩ 290555

def event290570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69063⟩⟩) 1 ⟨136⟩ 290568

def event290571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69063⟩⟩) (.sum [.predecessor 0 290569 .coefficient, .predecessor 1 290570 .coefficient])

def event290572 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69063⟩⟩) (.finite 1059)

def event290573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69064⟩⟩) 0 ⟨69063⟩ 290572

def event290574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69064⟩⟩) (.identity (.predecessor 0 290573 .coefficient))

def exact290575RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], []⟩, (1)⟩]

theorem exact290575RawTermsValid :
    exact290575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290575 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69064⟩⟩) exact290575RawTerms (.finite 1059) 290574 .exactZero (none)

def event290576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact290577RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact290577RawTermsValid :
    exact290577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290577 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact290577RawTerms .large 290576 .exactZero (none)

def event290578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69065⟩⟩) 0 ⟨6908⟩ 290577

def event290579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69065⟩⟩) 1 ⟨69064⟩ 290575

def event290580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69065⟩⟩) (.product (.predecessor 0 290578 .coefficient) (.predecessor 1 290579 .coefficient) (⟨false, false, none, none, none⟩))

def event290581 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290582 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290583 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290584 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290585 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290586 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290587 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290588 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290589 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290590 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290591 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290592 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290593 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290594 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290595 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290596 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290597 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event290598 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69065⟩⟩, .operator (⟨290577, 0⟩, ⟨290575, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact290599RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact290599RawTermsValid :
    exact290599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290599 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69065⟩⟩) exact290599RawTerms .large 290580 .exactZero (none)

def event290600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 290559

def event290601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact290602RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact290602RawTermsValid :
    exact290602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact290602RawTerms .large 290601 .exactZero (none)

def event290603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 290559

def event290604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact290605RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact290605RawTermsValid :
    exact290605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact290605RawTerms .large 290604 .exactZero (none)

def event290606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 290559

def event290607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact290608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact290608RawTermsValid :
    exact290608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact290608RawTerms .large 290607 .exactZero (none)

def event290609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 290559

def event290610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact290611RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact290611RawTermsValid :
    exact290611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact290611RawTerms .large 290610 .exactZero (none)

def event290612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 290559

def event290613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact290614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact290614RawTermsValid :
    exact290614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact290614RawTerms .large 290613 .exactZero (none)

def event290615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7222⟩⟩) 0 ⟨7177⟩ 290559

def event290616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7222⟩⟩) (.authority (.operator))

def exact290617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact290617RawTermsValid :
    exact290617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7222⟩⟩) exact290617RawTerms .large 290616 .exactZero (none)

def event290618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 290559

def event290619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact290620RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact290620RawTermsValid :
    exact290620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact290620RawTerms .large 290619 .exactZero (none)

def event290621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 290559

def event290622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact290623RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact290623RawTermsValid :
    exact290623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290623 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact290623RawTerms .large 290622 .exactZero (none)

def event290624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 290559

def event290625 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact290626RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact290626RawTermsValid :
    exact290626RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290626 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact290626RawTerms .large 290625 .exactZero (none)

def event290627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7214⟩⟩) 0 ⟨7177⟩ 290559

def event290628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7214⟩⟩) (.authority (.operator))

def exact290629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact290629RawTermsValid :
    exact290629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290629 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7214⟩⟩) exact290629RawTerms .large 290628 .exactZero (none)

def event290630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 290559

def event290631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact290632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact290632RawTermsValid :
    exact290632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact290632RawTerms .large 290631 .exactZero (none)

def event290633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 290559

def event290634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact290635RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact290635RawTermsValid :
    exact290635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact290635RawTerms .large 290634 .exactZero (none)

def event290636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 290559

def event290637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact290638RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact290638RawTermsValid :
    exact290638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290638 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact290638RawTerms .large 290637 .exactZero (none)

def event290639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 290559

def event290640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact290641RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact290641RawTermsValid :
    exact290641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290641 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact290641RawTerms .large 290640 .exactZero (none)

def event290642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 290559

def event290643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact290644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact290644RawTermsValid :
    exact290644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290644 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact290644RawTerms .large 290643 .exactZero (none)

def event290645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7202⟩⟩) 0 ⟨7177⟩ 290559

def event290646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7202⟩⟩) (.authority (.operator))

def exact290647RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact290647RawTermsValid :
    exact290647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7202⟩⟩) exact290647RawTerms .large 290646 .exactZero (none)

def event290648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 290559

def event290649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact290650RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact290650RawTermsValid :
    exact290650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact290650RawTerms .large 290649 .exactZero (none)

def event290651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 290559

def event290652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact290653RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact290653RawTermsValid :
    exact290653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290653 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact290653RawTerms .large 290652 .exactZero (none)

def event290654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 0 ⟨7198⟩ 290653

def event290655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 1 ⟨7200⟩ 290650

def event290656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7309⟩⟩) (.sum [.predecessor 0 290654 .coefficient, .predecessor 1 290655 .coefficient])

def exact290657RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact290657RawTermsValid :
    exact290657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290657 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7309⟩⟩) exact290657RawTerms .large 290656 .exactZero (none)

def event290658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 0 ⟨7309⟩ 290657

def event290659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 1 ⟨7202⟩ 290647

def event290660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7310⟩⟩) (.sum [.predecessor 0 290658 .coefficient, .predecessor 1 290659 .coefficient])

def exact290661RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact290661RawTermsValid :
    exact290661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290661 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7310⟩⟩) exact290661RawTerms .large 290660 .exactZero (none)

def event290662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 0 ⟨7310⟩ 290661

def event290663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 1 ⟨7204⟩ 290644

def event290664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7311⟩⟩) (.sum [.predecessor 0 290662 .coefficient, .predecessor 1 290663 .coefficient])

def exact290665RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact290665RawTermsValid :
    exact290665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290665 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7311⟩⟩) exact290665RawTerms .large 290664 .exactZero (none)

def event290666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 0 ⟨7311⟩ 290665

def event290667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 1 ⟨7206⟩ 290641

def event290668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7312⟩⟩) (.sum [.predecessor 0 290666 .coefficient, .predecessor 1 290667 .coefficient])

def exact290669RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact290669RawTermsValid :
    exact290669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290669 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7312⟩⟩) exact290669RawTerms .large 290668 .exactZero (none)

def event290670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 0 ⟨7312⟩ 290669

def event290671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 1 ⟨7208⟩ 290638

def event290672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7313⟩⟩) (.sum [.predecessor 0 290670 .coefficient, .predecessor 1 290671 .coefficient])

def exact290673RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact290673RawTermsValid :
    exact290673RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290673 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7313⟩⟩) exact290673RawTerms .large 290672 .exactZero (none)

def event290674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 0 ⟨7313⟩ 290673

def event290675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 1 ⟨7210⟩ 290635

def event290676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7314⟩⟩) (.sum [.predecessor 0 290674 .coefficient, .predecessor 1 290675 .coefficient])

def exact290677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact290677RawTermsValid :
    exact290677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7314⟩⟩) exact290677RawTerms .large 290676 .exactZero (none)

def event290678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 0 ⟨7314⟩ 290677

def event290679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 1 ⟨7212⟩ 290632

def event290680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7315⟩⟩) (.sum [.predecessor 0 290678 .coefficient, .predecessor 1 290679 .coefficient])

def exact290681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact290681RawTermsValid :
    exact290681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7315⟩⟩) exact290681RawTerms .large 290680 .exactZero (none)

def event290682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 0 ⟨7315⟩ 290681

def event290683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 1 ⟨7214⟩ 290629

def event290684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7316⟩⟩) (.sum [.predecessor 0 290682 .coefficient, .predecessor 1 290683 .coefficient])

def exact290685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact290685RawTermsValid :
    exact290685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290685 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7316⟩⟩) exact290685RawTerms .large 290684 .exactZero (none)

def event290686 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 0 ⟨7316⟩ 290685

def event290687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 1 ⟨7216⟩ 290626

def event290688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7317⟩⟩) (.sum [.predecessor 0 290686 .coefficient, .predecessor 1 290687 .coefficient])

def exact290689RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact290689RawTermsValid :
    exact290689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290689 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7317⟩⟩) exact290689RawTerms .large 290688 .exactZero (none)

def event290690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 0 ⟨7317⟩ 290689

def event290691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 1 ⟨7218⟩ 290623

def event290692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7318⟩⟩) (.sum [.predecessor 0 290690 .coefficient, .predecessor 1 290691 .coefficient])

def exact290693RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact290693RawTermsValid :
    exact290693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7318⟩⟩) exact290693RawTerms .large 290692 .exactZero (none)

def event290694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 0 ⟨7318⟩ 290693

def event290695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 1 ⟨7220⟩ 290620

def event290696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7319⟩⟩) (.sum [.predecessor 0 290694 .coefficient, .predecessor 1 290695 .coefficient])

def exact290697RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact290697RawTermsValid :
    exact290697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290697 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7319⟩⟩) exact290697RawTerms .large 290696 .exactZero (none)

def event290698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 0 ⟨7319⟩ 290697

def event290699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 1 ⟨7222⟩ 290617

def event290700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7320⟩⟩) (.sum [.predecessor 0 290698 .coefficient, .predecessor 1 290699 .coefficient])

def exact290701RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact290701RawTermsValid :
    exact290701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7320⟩⟩) exact290701RawTerms .large 290700 .exactZero (none)

def event290702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 0 ⟨7320⟩ 290701

def event290703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 1 ⟨7224⟩ 290614

def event290704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7321⟩⟩) (.sum [.predecessor 0 290702 .coefficient, .predecessor 1 290703 .coefficient])

def exact290705RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact290705RawTermsValid :
    exact290705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290705 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7321⟩⟩) exact290705RawTerms .large 290704 .exactZero (none)

def event290706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 0 ⟨7321⟩ 290705

def event290707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 1 ⟨7226⟩ 290611

def event290708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7322⟩⟩) (.sum [.predecessor 0 290706 .coefficient, .predecessor 1 290707 .coefficient])

def exact290709RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact290709RawTermsValid :
    exact290709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290709 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7322⟩⟩) exact290709RawTerms .large 290708 .exactZero (none)

def event290710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 0 ⟨7322⟩ 290709

def event290711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 1 ⟨7228⟩ 290608

def event290712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7323⟩⟩) (.sum [.predecessor 0 290710 .coefficient, .predecessor 1 290711 .coefficient])

def exact290713RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact290713RawTermsValid :
    exact290713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290713 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7323⟩⟩) exact290713RawTerms .large 290712 .exactZero (none)

def event290714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 0 ⟨7323⟩ 290713

def event290715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 1 ⟨7230⟩ 290605

def event290716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7324⟩⟩) (.sum [.predecessor 0 290714 .coefficient, .predecessor 1 290715 .coefficient])

def exact290717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact290717RawTermsValid :
    exact290717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7324⟩⟩) exact290717RawTerms .large 290716 .exactZero (none)

def event290718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 0 ⟨7324⟩ 290717

def event290719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 1 ⟨7232⟩ 290602

def event290720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7325⟩⟩) (.sum [.predecessor 0 290718 .coefficient, .predecessor 1 290719 .coefficient])

def exact290721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact290721RawTermsValid :
    exact290721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290721 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7325⟩⟩) exact290721RawTerms .large 290720 .exactZero (none)

def event290722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69066⟩⟩) 0 ⟨7325⟩ 290721

def event290723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69066⟩⟩) 1 ⟨69065⟩ 290599

def event290724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69066⟩⟩) (.sum [.predecessor 0 290722 .coefficient, .predecessor 1 290723 .coefficient])

def exact290725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact290725RawTermsValid :
    exact290725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69066⟩⟩) exact290725RawTerms .large 290724 .exactZero (none)

def event290726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71049⟩⟩) 0 ⟨69066⟩ 290725

def event290727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71049⟩⟩) 1 ⟨71048⟩ 290566

def event290728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71049⟩⟩) (.product (.predecessor 0 290726 .coefficient) (.predecessor 1 290727 .coefficient) (⟨false, false, none, none, none⟩))

def event290729 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 17⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290730 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 16⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290731 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 15⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290732 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 14⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290733 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 13⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290734 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 12⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290735 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 11⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290736 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 10⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290737 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 9⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290738 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 8⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 7⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 6⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 5⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290742 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 4⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290743 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 3⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290744 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 2⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290745 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 1⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290746 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 0⟩, ⟨290566, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 29⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290748 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290749 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290748 0, ⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290750 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 28⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290751 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290752 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290751 0, ⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290753 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 27⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290754 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290755 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290754 0, ⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290756 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 26⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290757 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290757 0, ⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290759 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 25⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290760 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290761 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290760 0, ⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290762 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 24⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290763 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290764 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290763 0, ⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290765 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 22⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290766 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290767 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290766 0, ⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290768 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 21⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290769 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290770 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290769 0, ⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 35⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290772 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290773 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290772 0, ⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 34⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290775 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290775 0, ⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 33⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290778 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290778 0, ⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290780 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 32⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290781 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290782 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290781 0, ⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 31⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290784 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290785 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290784 0, ⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 30⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290787 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290788 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290787 0, ⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 23⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290790 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290790 0, ⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 20⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290793 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290793 0, ⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 19⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290796 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290796 0, ⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290798 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .operator (⟨290725, 18⟩, ⟨290566, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290799 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563)

def event290800 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71049⟩⟩, .relation 290799 0, ⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def exact290801RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩]

theorem exact290801RawTermsValid :
    exact290801RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290801 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71049⟩⟩) exact290801RawTerms .large 290728 .exactZero (none)

def event290802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67341⟩⟩) 0 ⟨66191⟩ 290555

def event290803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67341⟩⟩) (.authority (.programFamilyFact))

def exact290804RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67341⟩⟩], []⟩, (1)⟩]

theorem exact290804RawTermsValid :
    exact290804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290804 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67341⟩⟩) exact290804RawTerms (.finite 18) 290803 .exactZero (none)

def event290805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67343⟩⟩) 0 ⟨6908⟩ 290577

def event290806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67343⟩⟩) 1 ⟨67341⟩ 290804

def event290807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67343⟩⟩) (.product (.predecessor 0 290805 .coefficient) (.predecessor 1 290806 .coefficient) (⟨false, true, none, none, some 1⟩))

def event290808 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67343⟩⟩, .operator (⟨290577, 0⟩, ⟨290804, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact290809RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact290809RawTermsValid :
    exact290809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290809 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67343⟩⟩) exact290809RawTerms .large 290807 .exactZero (none)

def event290810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7233⟩⟩) 0 ⟨7177⟩ 290559

def event290811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7233⟩⟩) (.authority (.operator))

def exact290812RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩]

theorem exact290812RawTermsValid :
    exact290812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7233⟩⟩) exact290812RawTerms .large 290811 .exactZero (none)

def event290813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67349⟩⟩) 0 ⟨7233⟩ 290812

def event290814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67349⟩⟩) 1 ⟨67343⟩ 290809

def event290815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67349⟩⟩) (.sum [.predecessor 0 290813 .coefficient, .predecessor 1 290814 .coefficient])

def eventLeaf18160 : Array AnnotatedEvent := #[
  { event := event290560
    frameStart := 290050 },
  { event := event290561
    frameStart := 290050 },
  { event := event290562
    frameStart := 290050 },
  { event := event290563
    frameStart := 290050 },
  { event := event290564
    frameStart := 290050 },
  { event := event290565
    frameStart := 290050 },
  { event := event290566
    frameStart := 290050 },
  { event := event290567
    frameStart := 290050 },
  { event := event290568
    frameStart := 290050 },
  { event := event290569
    frameStart := 290050 },
  { event := event290570
    frameStart := 290050 },
  { event := event290571
    frameStart := 290050 },
  { event := event290572
    frameStart := 290050 },
  { event := event290573
    frameStart := 290050 },
  { event := event290574
    frameStart := 290050 },
  { event := event290575
    frameStart := 290050 }
]

def eventLeaf18161 : Array AnnotatedEvent := #[
  { event := event290576
    frameStart := 290050 },
  { event := event290577
    frameStart := 290050 },
  { event := event290578
    frameStart := 290050 },
  { event := event290579
    frameStart := 290050 },
  { event := event290580
    frameStart := 290050 },
  { event := event290581
    frameStart := 290050 },
  { event := event290582
    frameStart := 290050 },
  { event := event290583
    frameStart := 290050 },
  { event := event290584
    frameStart := 290050 },
  { event := event290585
    frameStart := 290050 },
  { event := event290586
    frameStart := 290050 },
  { event := event290587
    frameStart := 290050 },
  { event := event290588
    frameStart := 290050 },
  { event := event290589
    frameStart := 290050 },
  { event := event290590
    frameStart := 290050 },
  { event := event290591
    frameStart := 290050 }
]

def eventLeaf18162 : Array AnnotatedEvent := #[
  { event := event290592
    frameStart := 290050 },
  { event := event290593
    frameStart := 290050 },
  { event := event290594
    frameStart := 290050 },
  { event := event290595
    frameStart := 290050 },
  { event := event290596
    frameStart := 290050 },
  { event := event290597
    frameStart := 290050 },
  { event := event290598
    frameStart := 290050 },
  { event := event290599
    frameStart := 290050 },
  { event := event290600
    frameStart := 290050 },
  { event := event290601
    frameStart := 290050 },
  { event := event290602
    frameStart := 290050 },
  { event := event290603
    frameStart := 290050 },
  { event := event290604
    frameStart := 290050 },
  { event := event290605
    frameStart := 290050 },
  { event := event290606
    frameStart := 290050 },
  { event := event290607
    frameStart := 290050 }
]

def eventLeaf18163 : Array AnnotatedEvent := #[
  { event := event290608
    frameStart := 290050 },
  { event := event290609
    frameStart := 290050 },
  { event := event290610
    frameStart := 290050 },
  { event := event290611
    frameStart := 290050 },
  { event := event290612
    frameStart := 290050 },
  { event := event290613
    frameStart := 290050 },
  { event := event290614
    frameStart := 290050 },
  { event := event290615
    frameStart := 290050 },
  { event := event290616
    frameStart := 290050 },
  { event := event290617
    frameStart := 290050 },
  { event := event290618
    frameStart := 290050 },
  { event := event290619
    frameStart := 290050 },
  { event := event290620
    frameStart := 290050 },
  { event := event290621
    frameStart := 290050 },
  { event := event290622
    frameStart := 290050 },
  { event := event290623
    frameStart := 290050 }
]

def eventLeaf18164 : Array AnnotatedEvent := #[
  { event := event290624
    frameStart := 290050 },
  { event := event290625
    frameStart := 290050 },
  { event := event290626
    frameStart := 290050 },
  { event := event290627
    frameStart := 290050 },
  { event := event290628
    frameStart := 290050 },
  { event := event290629
    frameStart := 290050 },
  { event := event290630
    frameStart := 290050 },
  { event := event290631
    frameStart := 290050 },
  { event := event290632
    frameStart := 290050 },
  { event := event290633
    frameStart := 290050 },
  { event := event290634
    frameStart := 290050 },
  { event := event290635
    frameStart := 290050 },
  { event := event290636
    frameStart := 290050 },
  { event := event290637
    frameStart := 290050 },
  { event := event290638
    frameStart := 290050 },
  { event := event290639
    frameStart := 290050 }
]

def eventLeaf18165 : Array AnnotatedEvent := #[
  { event := event290640
    frameStart := 290050 },
  { event := event290641
    frameStart := 290050 },
  { event := event290642
    frameStart := 290050 },
  { event := event290643
    frameStart := 290050 },
  { event := event290644
    frameStart := 290050 },
  { event := event290645
    frameStart := 290050 },
  { event := event290646
    frameStart := 290050 },
  { event := event290647
    frameStart := 290050 },
  { event := event290648
    frameStart := 290050 },
  { event := event290649
    frameStart := 290050 },
  { event := event290650
    frameStart := 290050 },
  { event := event290651
    frameStart := 290050 },
  { event := event290652
    frameStart := 290050 },
  { event := event290653
    frameStart := 290050 },
  { event := event290654
    frameStart := 290050 },
  { event := event290655
    frameStart := 290050 }
]

def eventLeaf18166 : Array AnnotatedEvent := #[
  { event := event290656
    frameStart := 290050 },
  { event := event290657
    frameStart := 290050 },
  { event := event290658
    frameStart := 290050 },
  { event := event290659
    frameStart := 290050 },
  { event := event290660
    frameStart := 290050 },
  { event := event290661
    frameStart := 290050 },
  { event := event290662
    frameStart := 290050 },
  { event := event290663
    frameStart := 290050 },
  { event := event290664
    frameStart := 290050 },
  { event := event290665
    frameStart := 290050 },
  { event := event290666
    frameStart := 290050 },
  { event := event290667
    frameStart := 290050 },
  { event := event290668
    frameStart := 290050 },
  { event := event290669
    frameStart := 290050 },
  { event := event290670
    frameStart := 290050 },
  { event := event290671
    frameStart := 290050 }
]

def eventLeaf18167 : Array AnnotatedEvent := #[
  { event := event290672
    frameStart := 290050 },
  { event := event290673
    frameStart := 290050 },
  { event := event290674
    frameStart := 290050 },
  { event := event290675
    frameStart := 290050 },
  { event := event290676
    frameStart := 290050 },
  { event := event290677
    frameStart := 290050 },
  { event := event290678
    frameStart := 290050 },
  { event := event290679
    frameStart := 290050 },
  { event := event290680
    frameStart := 290050 },
  { event := event290681
    frameStart := 290050 },
  { event := event290682
    frameStart := 290050 },
  { event := event290683
    frameStart := 290050 },
  { event := event290684
    frameStart := 290050 },
  { event := event290685
    frameStart := 290050 },
  { event := event290686
    frameStart := 290050 },
  { event := event290687
    frameStart := 290050 }
]

def eventLeaf18168 : Array AnnotatedEvent := #[
  { event := event290688
    frameStart := 290050 },
  { event := event290689
    frameStart := 290050 },
  { event := event290690
    frameStart := 290050 },
  { event := event290691
    frameStart := 290050 },
  { event := event290692
    frameStart := 290050 },
  { event := event290693
    frameStart := 290050 },
  { event := event290694
    frameStart := 290050 },
  { event := event290695
    frameStart := 290050 },
  { event := event290696
    frameStart := 290050 },
  { event := event290697
    frameStart := 290050 },
  { event := event290698
    frameStart := 290050 },
  { event := event290699
    frameStart := 290050 },
  { event := event290700
    frameStart := 290050 },
  { event := event290701
    frameStart := 290050 },
  { event := event290702
    frameStart := 290050 },
  { event := event290703
    frameStart := 290050 }
]

def eventLeaf18169 : Array AnnotatedEvent := #[
  { event := event290704
    frameStart := 290050 },
  { event := event290705
    frameStart := 290050 },
  { event := event290706
    frameStart := 290050 },
  { event := event290707
    frameStart := 290050 },
  { event := event290708
    frameStart := 290050 },
  { event := event290709
    frameStart := 290050 },
  { event := event290710
    frameStart := 290050 },
  { event := event290711
    frameStart := 290050 },
  { event := event290712
    frameStart := 290050 },
  { event := event290713
    frameStart := 290050 },
  { event := event290714
    frameStart := 290050 },
  { event := event290715
    frameStart := 290050 },
  { event := event290716
    frameStart := 290050 },
  { event := event290717
    frameStart := 290050 },
  { event := event290718
    frameStart := 290050 },
  { event := event290719
    frameStart := 290050 }
]

def eventLeaf18170 : Array AnnotatedEvent := #[
  { event := event290720
    frameStart := 290050 },
  { event := event290721
    frameStart := 290050 },
  { event := event290722
    frameStart := 290050 },
  { event := event290723
    frameStart := 290050 },
  { event := event290724
    frameStart := 290050 },
  { event := event290725
    frameStart := 290050 },
  { event := event290726
    frameStart := 290050 },
  { event := event290727
    frameStart := 290050 },
  { event := event290728
    frameStart := 290050 },
  { event := event290729
    frameStart := 290050 },
  { event := event290730
    frameStart := 290050 },
  { event := event290731
    frameStart := 290050 },
  { event := event290732
    frameStart := 290050 },
  { event := event290733
    frameStart := 290050 },
  { event := event290734
    frameStart := 290050 },
  { event := event290735
    frameStart := 290050 }
]

def eventLeaf18171 : Array AnnotatedEvent := #[
  { event := event290736
    frameStart := 290050 },
  { event := event290737
    frameStart := 290050 },
  { event := event290738
    frameStart := 290050 },
  { event := event290739
    frameStart := 290050 },
  { event := event290740
    frameStart := 290050 },
  { event := event290741
    frameStart := 290050 },
  { event := event290742
    frameStart := 290050 },
  { event := event290743
    frameStart := 290050 },
  { event := event290744
    frameStart := 290050 },
  { event := event290745
    frameStart := 290050 },
  { event := event290746
    frameStart := 290050 },
  { event := event290747
    frameStart := 290050 },
  { event := event290748
    frameStart := 290050 },
  { event := event290749
    frameStart := 290050 },
  { event := event290750
    frameStart := 290050 },
  { event := event290751
    frameStart := 290050 }
]

def eventLeaf18172 : Array AnnotatedEvent := #[
  { event := event290752
    frameStart := 290050 },
  { event := event290753
    frameStart := 290050 },
  { event := event290754
    frameStart := 290050 },
  { event := event290755
    frameStart := 290050 },
  { event := event290756
    frameStart := 290050 },
  { event := event290757
    frameStart := 290050 },
  { event := event290758
    frameStart := 290050 },
  { event := event290759
    frameStart := 290050 },
  { event := event290760
    frameStart := 290050 },
  { event := event290761
    frameStart := 290050 },
  { event := event290762
    frameStart := 290050 },
  { event := event290763
    frameStart := 290050 },
  { event := event290764
    frameStart := 290050 },
  { event := event290765
    frameStart := 290050 },
  { event := event290766
    frameStart := 290050 },
  { event := event290767
    frameStart := 290050 }
]

def eventLeaf18173 : Array AnnotatedEvent := #[
  { event := event290768
    frameStart := 290050 },
  { event := event290769
    frameStart := 290050 },
  { event := event290770
    frameStart := 290050 },
  { event := event290771
    frameStart := 290050 },
  { event := event290772
    frameStart := 290050 },
  { event := event290773
    frameStart := 290050 },
  { event := event290774
    frameStart := 290050 },
  { event := event290775
    frameStart := 290050 },
  { event := event290776
    frameStart := 290050 },
  { event := event290777
    frameStart := 290050 },
  { event := event290778
    frameStart := 290050 },
  { event := event290779
    frameStart := 290050 },
  { event := event290780
    frameStart := 290050 },
  { event := event290781
    frameStart := 290050 },
  { event := event290782
    frameStart := 290050 },
  { event := event290783
    frameStart := 290050 }
]

def eventLeaf18174 : Array AnnotatedEvent := #[
  { event := event290784
    frameStart := 290050 },
  { event := event290785
    frameStart := 290050 },
  { event := event290786
    frameStart := 290050 },
  { event := event290787
    frameStart := 290050 },
  { event := event290788
    frameStart := 290050 },
  { event := event290789
    frameStart := 290050 },
  { event := event290790
    frameStart := 290050 },
  { event := event290791
    frameStart := 290050 },
  { event := event290792
    frameStart := 290050 },
  { event := event290793
    frameStart := 290050 },
  { event := event290794
    frameStart := 290050 },
  { event := event290795
    frameStart := 290050 },
  { event := event290796
    frameStart := 290050 },
  { event := event290797
    frameStart := 290050 },
  { event := event290798
    frameStart := 290050 },
  { event := event290799
    frameStart := 290050 }
]

def eventLeaf18175 : Array AnnotatedEvent := #[
  { event := event290800
    frameStart := 290050 },
  { event := event290801
    frameStart := 290050 },
  { event := event290802
    frameStart := 290050 },
  { event := event290803
    frameStart := 290050 },
  { event := event290804
    frameStart := 290050 },
  { event := event290805
    frameStart := 290050 },
  { event := event290806
    frameStart := 290050 },
  { event := event290807
    frameStart := 290050 },
  { event := event290808
    frameStart := 290050 },
  { event := event290809
    frameStart := 290050 },
  { event := event290810
    frameStart := 290050 },
  { event := event290811
    frameStart := 290050 },
  { event := event290812
    frameStart := 290050 },
  { event := event290813
    frameStart := 290050 },
  { event := event290814
    frameStart := 290050 },
  { event := event290815
    frameStart := 290050 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1135
