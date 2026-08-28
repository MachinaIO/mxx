import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events280

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event71680 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event71681 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48002⟩⟩) 0 ⟨10749⟩ 71680

def event71682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48002⟩⟩) (.authority (.programFamilyFact))

def exact71683RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩, (1)⟩]

theorem exact71683RawTermsValid :
    exact71683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48002⟩⟩) exact71683RawTerms (.finite 60) 71682 .exactZero (none)

def event71684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15186⟩⟩) 0 ⟨10749⟩ 71680

def event71685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15186⟩⟩) (.authority (.programFamilyFact))

def exact71686RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩], []⟩, (1)⟩]

theorem exact71686RawTermsValid :
    exact71686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71686 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15186⟩⟩) exact71686RawTerms (.finite 60) 71685 .exactZero (none)

def event71687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48003⟩⟩) 0 ⟨15186⟩ 71686

def event71688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48003⟩⟩) 1 ⟨48002⟩ 71683

def event71689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48003⟩⟩) (.product (.predecessor 0 71687 .coefficient) (.predecessor 1 71688 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48003⟩⟩, .operator (⟨71686, 0⟩, ⟨71683, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩, (1)⟩)

def exact71691RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩, (1)⟩]

theorem exact71691RawTermsValid :
    exact71691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71691 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48003⟩⟩) exact71691RawTerms (.finite 3600) 71689 .exactZero (none)

def event71692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48004⟩⟩) 0 ⟨48003⟩ 71691

def event71693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48004⟩⟩) (.identity (.predecessor 0 71692 .coefficient))

def event71694 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48004⟩⟩) (.finite 3600)

def event71695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48204⟩⟩) 0 ⟨48004⟩ 71694

def event71696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48204⟩⟩) (.authority (.programFamilyFact))

def exact71697RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], []⟩, (1)⟩]

theorem exact71697RawTermsValid :
    exact71697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71697 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48204⟩⟩) exact71697RawTerms (.finite 60) 71696 .exactZero (none)

def event71698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48205⟩⟩) 0 ⟨48204⟩ 71697

def event71699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48205⟩⟩) (.identity (.predecessor 0 71698 .coefficient))

def event71700 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48205⟩⟩) (.finite 60)

def event71701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49362⟩⟩) 0 ⟨48205⟩ 71700

def event71702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49362⟩⟩) (.authority (.programFamilyFact))

def event71703 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49362⟩⟩) (.finite 3720)

def event71704 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event71705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49363⟩⟩) 0 ⟨7177⟩ 71704

def event71706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49363⟩⟩) 1 ⟨49362⟩ 71703

def event71707 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49363⟩⟩) (.authority (.operator))

def exact71708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (1)⟩]

theorem exact71708RawTermsValid :
    exact71708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71708 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49363⟩⟩) exact71708RawTerms .large 71707 .exactZero (none)

def event71709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50198⟩⟩) 0 ⟨49363⟩ 71708

def event71710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50198⟩⟩) (.authority (.operator))

def exact71711RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (1)⟩]

theorem exact71711RawTermsValid :
    exact71711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50198⟩⟩) exact71711RawTerms (.finite 8192) 71710 .exactZero (none)

def event71712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event71713 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event71714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49534⟩⟩) 0 ⟨48205⟩ 71700

def event71715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49534⟩⟩) 1 ⟨136⟩ 71713

def event71716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49534⟩⟩) (.sum [.predecessor 0 71714 .coefficient, .predecessor 1 71715 .coefficient])

def event71717 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49534⟩⟩) (.finite 60)

def event71718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49535⟩⟩) 0 ⟨49534⟩ 71717

def event71719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49535⟩⟩) (.identity (.predecessor 0 71718 .coefficient))

def exact71720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], []⟩, (1)⟩]

theorem exact71720RawTermsValid :
    exact71720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49535⟩⟩) exact71720RawTerms (.finite 60) 71719 .exactZero (none)

def event71721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact71722RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact71722RawTermsValid :
    exact71722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71722 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact71722RawTerms .large 71721 .exactZero (none)

def event71723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49536⟩⟩) 0 ⟨6908⟩ 71722

def event71724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49536⟩⟩) 1 ⟨49535⟩ 71720

def event71725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49536⟩⟩) (.product (.predecessor 0 71723 .coefficient) (.predecessor 1 71724 .coefficient) (⟨false, false, none, none, none⟩))

def event71726 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49536⟩⟩, .operator (⟨71722, 0⟩, ⟨71720, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact71727RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact71727RawTermsValid :
    exact71727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71727 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49536⟩⟩) exact71727RawTerms .large 71725 .exactZero (none)

def event71728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 71704

def event71729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact71730RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact71730RawTermsValid :
    exact71730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71730 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact71730RawTerms .large 71729 .exactZero (none)

def event71731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49537⟩⟩) 0 ⟨7196⟩ 71730

def event71732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49537⟩⟩) 1 ⟨49536⟩ 71727

def event71733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49537⟩⟩) (.sum [.predecessor 0 71731 .coefficient, .predecessor 1 71732 .coefficient])

def exact71734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact71734RawTermsValid :
    exact71734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49537⟩⟩) exact71734RawTerms .large 71733 .exactZero (none)

def event71735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50199⟩⟩) 0 ⟨49537⟩ 71734

def event71736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50199⟩⟩) 1 ⟨50198⟩ 71711

def event71737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50199⟩⟩) (.product (.predecessor 0 71735 .coefficient) (.predecessor 1 71736 .coefficient) (⟨false, false, none, none, none⟩))

def event71738 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50199⟩⟩, .operator (⟨71734, 0⟩, ⟨71711, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (1)⟩)

def event71739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50199⟩⟩, .operator (⟨71734, 1⟩, ⟨71711, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (-1)⟩)

def event71740 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50199⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50198⟩⟩) ⟨49363⟩ 71708)

def event71741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50199⟩⟩, .relation 71740 0, ⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (-1)⟩)

def exact71742RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (-1)⟩]

theorem exact71742RawTermsValid :
    exact71742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71742 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50199⟩⟩) exact71742RawTerms .large 71737 .exactZero (none)

def event71743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48450⟩⟩) 0 ⟨48205⟩ 71700

def event71744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48450⟩⟩) (.authority (.programFamilyFact))

def exact71745RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], []⟩, (1)⟩]

theorem exact71745RawTermsValid :
    exact71745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48450⟩⟩) exact71745RawTerms (.finite 60) 71744 .exactZero (none)

def event71746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48452⟩⟩) 0 ⟨6908⟩ 71722

def event71747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48452⟩⟩) 1 ⟨48450⟩ 71745

def event71748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48452⟩⟩) (.product (.predecessor 0 71746 .coefficient) (.predecessor 1 71747 .coefficient) (⟨false, true, none, none, some 1⟩))

def event71749 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48452⟩⟩, .operator (⟨71722, 0⟩, ⟨71745, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact71750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact71750RawTermsValid :
    exact71750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48452⟩⟩) exact71750RawTerms .large 71748 .exactZero (none)

def event71751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7231⟩⟩) 0 ⟨7177⟩ 71704

def event71752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7231⟩⟩) (.authority (.operator))

def exact71753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩]

theorem exact71753RawTermsValid :
    exact71753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7231⟩⟩) exact71753RawTerms .large 71752 .exactZero (none)

def event71754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48453⟩⟩) 0 ⟨7231⟩ 71753

def event71755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48453⟩⟩) 1 ⟨48452⟩ 71750

def event71756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48453⟩⟩) (.sum [.predecessor 0 71754 .coefficient, .predecessor 1 71755 .coefficient])

def exact71757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact71757RawTermsValid :
    exact71757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48453⟩⟩) exact71757RawTerms .large 71756 .exactZero (none)

def event71758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50203⟩⟩) 0 ⟨48453⟩ 71757

def event71759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50203⟩⟩) 1 ⟨50199⟩ 71742

def event71760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50203⟩⟩) (.sum [.predecessor 0 71758 .coefficient, .predecessor 1 71759 .coefficient])

def exact71761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact71761RawTermsValid :
    exact71761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71761 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50203⟩⟩) exact71761RawTerms .large 71760 .exactZero (none)

def event71762 : Event := .preFoldPolynomial 71761 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact71763RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event71763 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨50203⟩⟩) 71762 exact71763RawTerms .large 71760 .exactZero (none)

def event71764 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48205⟩⟩) ⟨⟨110⟩, ⟨93⟩, ⟨135⟩⟩ ⟨71606, 71764⟩

def event71765 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49035⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩) (1) 0 2 (.universal 71764 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩) (none) 71763)

def event71766 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49035⟩⟩, .relation 71765 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩)

def event71767 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49035⟩⟩, .relation 71765 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (-1)⟩)

def event71768 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49035⟩⟩, .relation 71765 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (1)⟩)

def event71769 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49035⟩⟩, .relation 71765 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact71770RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact71770RawTermsValid :
    exact71770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49035⟩⟩) exact71770RawTerms .large 71602 (.finite 202072841853861888) (some (71604))

def event71771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50201⟩⟩) 0 ⟨49035⟩ 71770

def event71772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50201⟩⟩) 1 ⟨50200⟩ 71592

def event71773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50201⟩⟩) (.sum [.predecessor 0 71771 .coefficient, .predecessor 1 71772 .coefficient])

def event71774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50201⟩⟩, .operator (⟨71770, 0⟩, ⟨71592, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩, (1)⟩)

def event71775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50201⟩⟩, .operator (⟨71770, 2⟩, ⟨71592, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩, (-1)⟩)

def event71776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50201⟩⟩) (.sum [.result 71770 .summary, .result 71592 .summary])

def exact71777RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact71777RawTermsValid :
    exact71777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50201⟩⟩) exact71777RawTerms .large 71773 (.finite 32194504275408640829496428331008) (some (71776))

def event71778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50202⟩⟩) 0 ⟨50201⟩ 71777

def event71779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50202⟩⟩) 1 ⟨7148⟩ 15542

def event71780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50202⟩⟩) (.product (.predecessor 0 71778 .coefficient) (.predecessor 1 71779 .coefficient) (⟨false, false, none, none, none⟩))

def event71781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50202⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) [⟨.result 15538 .coefficient, false, none⟩])

def event71782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50202⟩⟩) (.product (.result 71777 .summary) (.transfer 71781) (⟨false, false, none, none, none⟩))

def event71783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50202⟩⟩, .operator (⟨71777, 0⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event71784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50202⟩⟩, .operator (⟨71777, 1⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event71785 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50202⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7147⟩⟩) ⟨7039⟩ 15535)

def event71786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50202⟩⟩, .relation 71785 0, ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact71787RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩]

theorem exact71787RawTermsValid :
    exact71787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71787 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50202⟩⟩) exact71787RawTerms .large 71780 (.finite 345685857434530723496243679576218056785920) (some (71782))

def event71788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46683⟩⟩) 0 ⟨7177⟩ 15500

def event71789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46683⟩⟩) 1 ⟨46682⟩ 61754

def event71790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46683⟩⟩) (.authority (.operator))

def exact71791RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46683⟩⟩]⟩, (1)⟩]

theorem exact71791RawTermsValid :
    exact71791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71791 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46683⟩⟩) exact71791RawTerms .large 71790 .exactZero (none)

def event71792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47518⟩⟩) 0 ⟨46683⟩ 71791

def event71793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47518⟩⟩) (.authority (.operator))

def exact71794RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩, (1)⟩]

theorem exact71794RawTermsValid :
    exact71794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71794 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47518⟩⟩) exact71794RawTerms (.finite 8192) 71793 .exactZero (none)

def event71795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47520⟩⟩) 0 ⟨47058⟩ 62038

def event71796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47520⟩⟩) 1 ⟨47518⟩ 71794

def event71797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47520⟩⟩) (.product (.predecessor 0 71795 .coefficient) (.predecessor 1 71796 .coefficient) (⟨false, false, none, none, none⟩))

def event71798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47520⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩) [⟨.result 71794 .coefficient, false, none⟩])

def event71799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47520⟩⟩) (.product (.result 62038 .summary) (.transfer 71798) (⟨false, false, none, none, none⟩))

def event71800 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47520⟩⟩, .operator (⟨62038, 0⟩, ⟨71794, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩, (1)⟩)

def event71801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47520⟩⟩, .operator (⟨62038, 1⟩, ⟨71794, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩, (-1)⟩)

def event71802 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47520⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47518⟩⟩) ⟨46683⟩ 71791)

def event71803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47520⟩⟩, .relation 71802 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨46683⟩⟩]⟩, (-1)⟩)

def exact71804RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨46683⟩⟩]⟩, (-1)⟩]

theorem exact71804RawTermsValid :
    exact71804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71804 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47520⟩⟩) exact71804RawTerms .large 71797 (.finite 32194307824962751379413684715520) (some (71799))

def event71805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46352⟩⟩) 0 ⟨45525⟩ 2378

def event71806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46352⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact71807RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩, (1)⟩]

theorem exact71807RawTermsValid :
    exact71807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71807 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46352⟩⟩) exact71807RawTerms (.finite 5647228698) 71806 .exactZero (none)

def event71808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46354⟩⟩) 0 ⟨46352⟩ 71807

def event71809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46354⟩⟩) 1 ⟨2370⟩ 4

def event71810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46354⟩⟩) (.scale (.predecessor 0 71808 .coefficient) (.value (.predecessor 1 71809 .coefficient)))

def exact71811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩, (1)⟩]

theorem exact71811RawTermsValid :
    exact71811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46354⟩⟩) exact71811RawTerms (.finite 5647228698) 71810 .exactZero (none)

def event71812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46355⟩⟩) 0 ⟨10792⟩ 61370

def event71813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46355⟩⟩) 1 ⟨46354⟩ 71811

def event71814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46355⟩⟩) (.product (.predecessor 0 71812 .coefficient) (.predecessor 1 71813 .coefficient) (⟨false, false, none, none, none⟩))

def event71815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46355⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩) [⟨.result 71807 .coefficient, false, none⟩])

def event71816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46355⟩⟩) (.product (.result 61370 .summary) (.transfer 71815) (⟨false, false, none, none, none⟩))

def event71817 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46355⟩⟩, .operator (⟨61370, 0⟩, ⟨71811, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩, (1)⟩)

def event71818 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46353⟩⟩)

def event71819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event71820 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event71821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event71822 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event71823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event71824 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event71825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event71826 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event71827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 71826

def event71828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 71824

def event71829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 71827 .coefficient) (.value (.predecessor 1 71828 .coefficient)))

def event71830 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event71831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 71830

def event71832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 71822

def event71833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 71831 .coefficient, .predecessor 1 71832 .coefficient])

def event71834 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event71835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 71834

def event71836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 71820

def event71837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 71836 .coefficient))

def event71838 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event71839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45322⟩⟩) 0 ⟨10749⟩ 71838

def event71840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45322⟩⟩) (.authority (.programFamilyFact))

def exact71841RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩, (1)⟩]

theorem exact71841RawTermsValid :
    exact71841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71841 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45322⟩⟩) exact71841RawTerms (.finite 58) 71840 .exactZero (none)

def event71842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14886⟩⟩) 0 ⟨10749⟩ 71838

def event71843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14886⟩⟩) (.authority (.programFamilyFact))

def exact71844RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩], []⟩, (1)⟩]

theorem exact71844RawTermsValid :
    exact71844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14886⟩⟩) exact71844RawTerms (.finite 58) 71843 .exactZero (none)

def event71845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45323⟩⟩) 0 ⟨14886⟩ 71844

def event71846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45323⟩⟩) 1 ⟨45322⟩ 71841

def event71847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45323⟩⟩) (.product (.predecessor 0 71845 .coefficient) (.predecessor 1 71846 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45323⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩) [⟨.result 71844 .coefficient, true, some 1⟩, ⟨.result 71841 .coefficient, true, some 1⟩])

def event71849 : Event := .survivorFold (1) 71848

def exact71850RawTerms : List Term := []

theorem exact71850RawTermsValid :
    exact71850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71850 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45323⟩⟩) exact71850RawTerms (.finite 3364) 71847 (.finite 3364) (some (71848))

def event71851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45324⟩⟩) 0 ⟨45323⟩ 71850

def event71852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45324⟩⟩) (.identity (.predecessor 0 71851 .coefficient))

def event71853 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45324⟩⟩) (.finite 3364)

def event71854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45524⟩⟩) 0 ⟨45324⟩ 71853

def event71855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45524⟩⟩) (.authority (.programFamilyFact))

def exact71856RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], []⟩, (1)⟩]

theorem exact71856RawTermsValid :
    exact71856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71856 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45524⟩⟩) exact71856RawTerms (.finite 58) 71855 .exactZero (none)

def event71857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45525⟩⟩) 0 ⟨45524⟩ 71856

def event71858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45525⟩⟩) (.identity (.predecessor 0 71857 .coefficient))

def event71859 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45525⟩⟩) (.finite 58)

def event71860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46352⟩⟩) 0 ⟨45525⟩ 71859

def event71861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46352⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact71862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩, (1)⟩]

theorem exact71862RawTermsValid :
    exact71862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46352⟩⟩) exact71862RawTerms (.finite 5647228698) 71861 .exactZero (none)

def event71863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact71864RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact71864RawTermsValid :
    exact71864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71864 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact71864RawTerms .large 71863 .exactZero (none)

def event71865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46353⟩⟩) 0 ⟨35⟩ 71864

def event71866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46353⟩⟩) 1 ⟨46352⟩ 71862

def event71867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46353⟩⟩) (.product (.predecessor 0 71865 .coefficient) (.predecessor 1 71866 .coefficient) (⟨false, false, none, none, none⟩))

def event71868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46353⟩⟩, .operator (⟨71864, 0⟩, ⟨71862, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩, (1)⟩)

def exact71869RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩, (1)⟩]

theorem exact71869RawTermsValid :
    exact71869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46353⟩⟩) exact71869RawTerms .large 71867 .exactZero (none)

def event71870 : Event := .preFoldPolynomial 71869 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩, (1)⟩] .exactZero none

def exact71871RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩, (1)⟩]

def event71871 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46353⟩⟩) 71870 exact71871RawTerms .large 71867 .exactZero (none)

def event71872 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47523⟩⟩)

def event71873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event71874 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event71875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event71876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event71877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event71878 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event71879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event71880 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event71881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 71880

def event71882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 71878

def event71883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 71881 .coefficient) (.value (.predecessor 1 71882 .coefficient)))

def event71884 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event71885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 71884

def event71886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 71876

def event71887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 71885 .coefficient, .predecessor 1 71886 .coefficient])

def event71888 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event71889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 71888

def event71890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 71874

def event71891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 71890 .coefficient))

def event71892 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event71893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45322⟩⟩) 0 ⟨10749⟩ 71892

def event71894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45322⟩⟩) (.authority (.programFamilyFact))

def exact71895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩, (1)⟩]

theorem exact71895RawTermsValid :
    exact71895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45322⟩⟩) exact71895RawTerms (.finite 58) 71894 .exactZero (none)

def event71896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14886⟩⟩) 0 ⟨10749⟩ 71892

def event71897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14886⟩⟩) (.authority (.programFamilyFact))

def exact71898RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩], []⟩, (1)⟩]

theorem exact71898RawTermsValid :
    exact71898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14886⟩⟩) exact71898RawTerms (.finite 58) 71897 .exactZero (none)

def event71899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45323⟩⟩) 0 ⟨14886⟩ 71898

def event71900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45323⟩⟩) 1 ⟨45322⟩ 71895

def event71901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45323⟩⟩) (.product (.predecessor 0 71899 .coefficient) (.predecessor 1 71900 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45323⟩⟩, .operator (⟨71898, 0⟩, ⟨71895, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩, (1)⟩)

def exact71903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩, (1)⟩]

theorem exact71903RawTermsValid :
    exact71903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45323⟩⟩) exact71903RawTerms (.finite 3364) 71901 .exactZero (none)

def event71904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45324⟩⟩) 0 ⟨45323⟩ 71903

def event71905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45324⟩⟩) (.identity (.predecessor 0 71904 .coefficient))

def event71906 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45324⟩⟩) (.finite 3364)

def event71907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45524⟩⟩) 0 ⟨45324⟩ 71906

def event71908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45524⟩⟩) (.authority (.programFamilyFact))

def exact71909RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], []⟩, (1)⟩]

theorem exact71909RawTermsValid :
    exact71909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45524⟩⟩) exact71909RawTerms (.finite 58) 71908 .exactZero (none)

def event71910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45525⟩⟩) 0 ⟨45524⟩ 71909

def event71911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45525⟩⟩) (.identity (.predecessor 0 71910 .coefficient))

def event71912 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45525⟩⟩) (.finite 58)

def event71913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46682⟩⟩) 0 ⟨45525⟩ 71912

def event71914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46682⟩⟩) (.authority (.programFamilyFact))

def event71915 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46682⟩⟩) (.finite 3720)

def event71916 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event71917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46683⟩⟩) 0 ⟨7177⟩ 71916

def event71918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46683⟩⟩) 1 ⟨46682⟩ 71915

def event71919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46683⟩⟩) (.authority (.operator))

def exact71920RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46683⟩⟩]⟩, (1)⟩]

theorem exact71920RawTermsValid :
    exact71920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46683⟩⟩) exact71920RawTerms .large 71919 .exactZero (none)

def event71921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47518⟩⟩) 0 ⟨46683⟩ 71920

def event71922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47518⟩⟩) (.authority (.operator))

def exact71923RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩, (1)⟩]

theorem exact71923RawTermsValid :
    exact71923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47518⟩⟩) exact71923RawTerms (.finite 8192) 71922 .exactZero (none)

def event71924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event71925 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event71926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46854⟩⟩) 0 ⟨45525⟩ 71912

def event71927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46854⟩⟩) 1 ⟨136⟩ 71925

def event71928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46854⟩⟩) (.sum [.predecessor 0 71926 .coefficient, .predecessor 1 71927 .coefficient])

def event71929 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46854⟩⟩) (.finite 58)

def event71930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46855⟩⟩) 0 ⟨46854⟩ 71929

def event71931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46855⟩⟩) (.identity (.predecessor 0 71930 .coefficient))

def exact71932RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], []⟩, (1)⟩]

theorem exact71932RawTermsValid :
    exact71932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71932 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46855⟩⟩) exact71932RawTerms (.finite 58) 71931 .exactZero (none)

def event71933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact71934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact71934RawTermsValid :
    exact71934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact71934RawTerms .large 71933 .exactZero (none)

def event71935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46856⟩⟩) 0 ⟨6908⟩ 71934

def eventLeaf4480 : Array AnnotatedEvent := #[
  { event := event71680
    frameStart := 71660 },
  { event := event71681
    frameStart := 71660 },
  { event := event71682
    frameStart := 71660 },
  { event := event71683
    frameStart := 71660 },
  { event := event71684
    frameStart := 71660 },
  { event := event71685
    frameStart := 71660 },
  { event := event71686
    frameStart := 71660 },
  { event := event71687
    frameStart := 71660 },
  { event := event71688
    frameStart := 71660 },
  { event := event71689
    frameStart := 71660 },
  { event := event71690
    frameStart := 71660 },
  { event := event71691
    frameStart := 71660 },
  { event := event71692
    frameStart := 71660 },
  { event := event71693
    frameStart := 71660 },
  { event := event71694
    frameStart := 71660 },
  { event := event71695
    frameStart := 71660 }
]

def eventLeaf4481 : Array AnnotatedEvent := #[
  { event := event71696
    frameStart := 71660 },
  { event := event71697
    frameStart := 71660 },
  { event := event71698
    frameStart := 71660 },
  { event := event71699
    frameStart := 71660 },
  { event := event71700
    frameStart := 71660 },
  { event := event71701
    frameStart := 71660 },
  { event := event71702
    frameStart := 71660 },
  { event := event71703
    frameStart := 71660 },
  { event := event71704
    frameStart := 71660 },
  { event := event71705
    frameStart := 71660 },
  { event := event71706
    frameStart := 71660 },
  { event := event71707
    frameStart := 71660 },
  { event := event71708
    frameStart := 71660 },
  { event := event71709
    frameStart := 71660 },
  { event := event71710
    frameStart := 71660 },
  { event := event71711
    frameStart := 71660 }
]

def eventLeaf4482 : Array AnnotatedEvent := #[
  { event := event71712
    frameStart := 71660 },
  { event := event71713
    frameStart := 71660 },
  { event := event71714
    frameStart := 71660 },
  { event := event71715
    frameStart := 71660 },
  { event := event71716
    frameStart := 71660 },
  { event := event71717
    frameStart := 71660 },
  { event := event71718
    frameStart := 71660 },
  { event := event71719
    frameStart := 71660 },
  { event := event71720
    frameStart := 71660 },
  { event := event71721
    frameStart := 71660 },
  { event := event71722
    frameStart := 71660 },
  { event := event71723
    frameStart := 71660 },
  { event := event71724
    frameStart := 71660 },
  { event := event71725
    frameStart := 71660 },
  { event := event71726
    frameStart := 71660 },
  { event := event71727
    frameStart := 71660 }
]

def eventLeaf4483 : Array AnnotatedEvent := #[
  { event := event71728
    frameStart := 71660 },
  { event := event71729
    frameStart := 71660 },
  { event := event71730
    frameStart := 71660 },
  { event := event71731
    frameStart := 71660 },
  { event := event71732
    frameStart := 71660 },
  { event := event71733
    frameStart := 71660 },
  { event := event71734
    frameStart := 71660 },
  { event := event71735
    frameStart := 71660 },
  { event := event71736
    frameStart := 71660 },
  { event := event71737
    frameStart := 71660 },
  { event := event71738
    frameStart := 71660 },
  { event := event71739
    frameStart := 71660 },
  { event := event71740
    frameStart := 71660 },
  { event := event71741
    frameStart := 71660 },
  { event := event71742
    frameStart := 71660 },
  { event := event71743
    frameStart := 71660 }
]

def eventLeaf4484 : Array AnnotatedEvent := #[
  { event := event71744
    frameStart := 71660 },
  { event := event71745
    frameStart := 71660 },
  { event := event71746
    frameStart := 71660 },
  { event := event71747
    frameStart := 71660 },
  { event := event71748
    frameStart := 71660 },
  { event := event71749
    frameStart := 71660 },
  { event := event71750
    frameStart := 71660 },
  { event := event71751
    frameStart := 71660 },
  { event := event71752
    frameStart := 71660 },
  { event := event71753
    frameStart := 71660 },
  { event := event71754
    frameStart := 71660 },
  { event := event71755
    frameStart := 71660 },
  { event := event71756
    frameStart := 71660 },
  { event := event71757
    frameStart := 71660 },
  { event := event71758
    frameStart := 71660 },
  { event := event71759
    frameStart := 71660 }
]

def eventLeaf4485 : Array AnnotatedEvent := #[
  { event := event71760
    frameStart := 71660 },
  { event := event71761
    frameStart := 71660 },
  { event := event71762
    frameStart := 71660 },
  { event := event71763
    frameStart := 71660 },
  { event := event71764
    frameStart := 0 },
  { event := event71765
    frameStart := 0 },
  { event := event71766
    frameStart := 0 },
  { event := event71767
    frameStart := 0 },
  { event := event71768
    frameStart := 0 },
  { event := event71769
    frameStart := 0 },
  { event := event71770
    frameStart := 0 },
  { event := event71771
    frameStart := 0 },
  { event := event71772
    frameStart := 0 },
  { event := event71773
    frameStart := 0 },
  { event := event71774
    frameStart := 0 },
  { event := event71775
    frameStart := 0 }
]

def eventLeaf4486 : Array AnnotatedEvent := #[
  { event := event71776
    frameStart := 0 },
  { event := event71777
    frameStart := 0 },
  { event := event71778
    frameStart := 0 },
  { event := event71779
    frameStart := 0 },
  { event := event71780
    frameStart := 0 },
  { event := event71781
    frameStart := 0 },
  { event := event71782
    frameStart := 0 },
  { event := event71783
    frameStart := 0 },
  { event := event71784
    frameStart := 0 },
  { event := event71785
    frameStart := 0 },
  { event := event71786
    frameStart := 0 },
  { event := event71787
    frameStart := 0 },
  { event := event71788
    frameStart := 0 },
  { event := event71789
    frameStart := 0 },
  { event := event71790
    frameStart := 0 },
  { event := event71791
    frameStart := 0 }
]

def eventLeaf4487 : Array AnnotatedEvent := #[
  { event := event71792
    frameStart := 0 },
  { event := event71793
    frameStart := 0 },
  { event := event71794
    frameStart := 0 },
  { event := event71795
    frameStart := 0 },
  { event := event71796
    frameStart := 0 },
  { event := event71797
    frameStart := 0 },
  { event := event71798
    frameStart := 0 },
  { event := event71799
    frameStart := 0 },
  { event := event71800
    frameStart := 0 },
  { event := event71801
    frameStart := 0 },
  { event := event71802
    frameStart := 0 },
  { event := event71803
    frameStart := 0 },
  { event := event71804
    frameStart := 0 },
  { event := event71805
    frameStart := 0 },
  { event := event71806
    frameStart := 0 },
  { event := event71807
    frameStart := 0 }
]

def eventLeaf4488 : Array AnnotatedEvent := #[
  { event := event71808
    frameStart := 0 },
  { event := event71809
    frameStart := 0 },
  { event := event71810
    frameStart := 0 },
  { event := event71811
    frameStart := 0 },
  { event := event71812
    frameStart := 0 },
  { event := event71813
    frameStart := 0 },
  { event := event71814
    frameStart := 0 },
  { event := event71815
    frameStart := 0 },
  { event := event71816
    frameStart := 0 },
  { event := event71817
    frameStart := 0 },
  { event := event71818
    frameStart := 71818 },
  { event := event71819
    frameStart := 71818 },
  { event := event71820
    frameStart := 71818 },
  { event := event71821
    frameStart := 71818 },
  { event := event71822
    frameStart := 71818 },
  { event := event71823
    frameStart := 71818 }
]

def eventLeaf4489 : Array AnnotatedEvent := #[
  { event := event71824
    frameStart := 71818 },
  { event := event71825
    frameStart := 71818 },
  { event := event71826
    frameStart := 71818 },
  { event := event71827
    frameStart := 71818 },
  { event := event71828
    frameStart := 71818 },
  { event := event71829
    frameStart := 71818 },
  { event := event71830
    frameStart := 71818 },
  { event := event71831
    frameStart := 71818 },
  { event := event71832
    frameStart := 71818 },
  { event := event71833
    frameStart := 71818 },
  { event := event71834
    frameStart := 71818 },
  { event := event71835
    frameStart := 71818 },
  { event := event71836
    frameStart := 71818 },
  { event := event71837
    frameStart := 71818 },
  { event := event71838
    frameStart := 71818 },
  { event := event71839
    frameStart := 71818 }
]

def eventLeaf4490 : Array AnnotatedEvent := #[
  { event := event71840
    frameStart := 71818 },
  { event := event71841
    frameStart := 71818 },
  { event := event71842
    frameStart := 71818 },
  { event := event71843
    frameStart := 71818 },
  { event := event71844
    frameStart := 71818 },
  { event := event71845
    frameStart := 71818 },
  { event := event71846
    frameStart := 71818 },
  { event := event71847
    frameStart := 71818 },
  { event := event71848
    frameStart := 71818 },
  { event := event71849
    frameStart := 71818 },
  { event := event71850
    frameStart := 71818 },
  { event := event71851
    frameStart := 71818 },
  { event := event71852
    frameStart := 71818 },
  { event := event71853
    frameStart := 71818 },
  { event := event71854
    frameStart := 71818 },
  { event := event71855
    frameStart := 71818 }
]

def eventLeaf4491 : Array AnnotatedEvent := #[
  { event := event71856
    frameStart := 71818 },
  { event := event71857
    frameStart := 71818 },
  { event := event71858
    frameStart := 71818 },
  { event := event71859
    frameStart := 71818 },
  { event := event71860
    frameStart := 71818 },
  { event := event71861
    frameStart := 71818 },
  { event := event71862
    frameStart := 71818 },
  { event := event71863
    frameStart := 71818 },
  { event := event71864
    frameStart := 71818 },
  { event := event71865
    frameStart := 71818 },
  { event := event71866
    frameStart := 71818 },
  { event := event71867
    frameStart := 71818 },
  { event := event71868
    frameStart := 71818 },
  { event := event71869
    frameStart := 71818 },
  { event := event71870
    frameStart := 71818 },
  { event := event71871
    frameStart := 71818 }
]

def eventLeaf4492 : Array AnnotatedEvent := #[
  { event := event71872
    frameStart := 71872 },
  { event := event71873
    frameStart := 71872 },
  { event := event71874
    frameStart := 71872 },
  { event := event71875
    frameStart := 71872 },
  { event := event71876
    frameStart := 71872 },
  { event := event71877
    frameStart := 71872 },
  { event := event71878
    frameStart := 71872 },
  { event := event71879
    frameStart := 71872 },
  { event := event71880
    frameStart := 71872 },
  { event := event71881
    frameStart := 71872 },
  { event := event71882
    frameStart := 71872 },
  { event := event71883
    frameStart := 71872 },
  { event := event71884
    frameStart := 71872 },
  { event := event71885
    frameStart := 71872 },
  { event := event71886
    frameStart := 71872 },
  { event := event71887
    frameStart := 71872 }
]

def eventLeaf4493 : Array AnnotatedEvent := #[
  { event := event71888
    frameStart := 71872 },
  { event := event71889
    frameStart := 71872 },
  { event := event71890
    frameStart := 71872 },
  { event := event71891
    frameStart := 71872 },
  { event := event71892
    frameStart := 71872 },
  { event := event71893
    frameStart := 71872 },
  { event := event71894
    frameStart := 71872 },
  { event := event71895
    frameStart := 71872 },
  { event := event71896
    frameStart := 71872 },
  { event := event71897
    frameStart := 71872 },
  { event := event71898
    frameStart := 71872 },
  { event := event71899
    frameStart := 71872 },
  { event := event71900
    frameStart := 71872 },
  { event := event71901
    frameStart := 71872 },
  { event := event71902
    frameStart := 71872 },
  { event := event71903
    frameStart := 71872 }
]

def eventLeaf4494 : Array AnnotatedEvent := #[
  { event := event71904
    frameStart := 71872 },
  { event := event71905
    frameStart := 71872 },
  { event := event71906
    frameStart := 71872 },
  { event := event71907
    frameStart := 71872 },
  { event := event71908
    frameStart := 71872 },
  { event := event71909
    frameStart := 71872 },
  { event := event71910
    frameStart := 71872 },
  { event := event71911
    frameStart := 71872 },
  { event := event71912
    frameStart := 71872 },
  { event := event71913
    frameStart := 71872 },
  { event := event71914
    frameStart := 71872 },
  { event := event71915
    frameStart := 71872 },
  { event := event71916
    frameStart := 71872 },
  { event := event71917
    frameStart := 71872 },
  { event := event71918
    frameStart := 71872 },
  { event := event71919
    frameStart := 71872 }
]

def eventLeaf4495 : Array AnnotatedEvent := #[
  { event := event71920
    frameStart := 71872 },
  { event := event71921
    frameStart := 71872 },
  { event := event71922
    frameStart := 71872 },
  { event := event71923
    frameStart := 71872 },
  { event := event71924
    frameStart := 71872 },
  { event := event71925
    frameStart := 71872 },
  { event := event71926
    frameStart := 71872 },
  { event := event71927
    frameStart := 71872 },
  { event := event71928
    frameStart := 71872 },
  { event := event71929
    frameStart := 71872 },
  { event := event71930
    frameStart := 71872 },
  { event := event71931
    frameStart := 71872 },
  { event := event71932
    frameStart := 71872 },
  { event := event71933
    frameStart := 71872 },
  { event := event71934
    frameStart := 71872 },
  { event := event71935
    frameStart := 71872 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events280
