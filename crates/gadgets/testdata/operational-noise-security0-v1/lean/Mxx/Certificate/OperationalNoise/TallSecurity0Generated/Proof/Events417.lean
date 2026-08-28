import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events417

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event106752 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6712⟩⟩) (.authority (.operator))

def exact106753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩]

theorem exact106753RawTermsValid :
    exact106753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106753 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6712⟩⟩) exact106753RawTerms .large 106752 .exactZero (none)

def event106754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15199⟩⟩) 0 ⟨6712⟩ 106753

def event106755 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15199⟩⟩) 1 ⟨15198⟩ 106750

def event106756 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15199⟩⟩) (.sum [.predecessor 0 106754 .coefficient, .predecessor 1 106755 .coefficient])

def exact106757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106757RawTermsValid :
    exact106757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106757 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15199⟩⟩) exact106757RawTerms .large 106756 .exactZero (none)

def event106758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26745⟩⟩) 0 ⟨15199⟩ 106757

def event106759 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26745⟩⟩) 1 ⟨26740⟩ 106742

def event106760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26745⟩⟩) (.sum [.predecessor 0 106758 .coefficient, .predecessor 1 106759 .coefficient])

def exact106761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26739⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106761RawTermsValid :
    exact106761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106761 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26745⟩⟩) exact106761RawTerms .large 106760 .exactZero (none)

def event106762 : Event := .preFoldPolynomial 106761 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26739⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact106763RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26739⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event106763 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26745⟩⟩) 106762 exact106763RawTerms .large 106760 .exactZero (none)

def event106764 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15105⟩⟩) ⟨⟨125⟩, ⟨31⟩, ⟨109⟩⟩ ⟨106630, 106764⟩

def event106765 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20600⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20597⟩⟩]⟩) (1) 0 2 (.universal 106764 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20597⟩⟩]⟩) (none) 106763)

def event106766 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20600⟩⟩, .relation 106765 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩)

def event106767 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20600⟩⟩, .relation 106765 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26739⟩⟩]⟩, (-1)⟩)

def event106768 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20600⟩⟩, .relation 106765 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23837⟩⟩]⟩, (1)⟩)

def event106769 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20600⟩⟩, .relation 106765 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106770RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26739⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106770RawTermsValid :
    exact106770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106770 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20600⟩⟩) exact106770RawTerms .large 106626 (.finite 1811303510016) (some (106628))

def event106771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26742⟩⟩) 0 ⟨20600⟩ 106770

def event106772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26742⟩⟩) 1 ⟨26741⟩ 106616

def event106773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26742⟩⟩) (.sum [.predecessor 0 106771 .coefficient, .predecessor 1 106772 .coefficient])

def event106774 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26742⟩⟩, .operator (⟨106770, 0⟩, ⟨106616, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26739⟩⟩]⟩, (1)⟩)

def event106775 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26742⟩⟩, .operator (⟨106770, 2⟩, ⟨106616, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23837⟩⟩]⟩, (-1)⟩)

def event106776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26742⟩⟩) (.sum [.result 106770 .summary, .result 106616 .summary])

def exact106777RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106777RawTermsValid :
    exact106777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106777 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26742⟩⟩) exact106777RawTerms .large 106773 (.finite 1291911586824442228736) (some (106776))

def event106778 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26743⟩⟩) 0 ⟨26742⟩ 106777

def event106779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26743⟩⟩) 1 ⟨6664⟩ 5819

def event106780 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26743⟩⟩) (.product (.predecessor 0 106778 .coefficient) (.predecessor 1 106779 .coefficient) (⟨false, false, none, none, none⟩))

def event106781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26743⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩) [⟨.result 5815 .coefficient, false, none⟩])

def event106782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26743⟩⟩) (.product (.result 106777 .summary) (.transfer 106781) (⟨false, false, none, none, none⟩))

def event106783 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26743⟩⟩, .operator (⟨106777, 0⟩, ⟨5819, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩)

def event106784 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26743⟩⟩, .operator (⟨106777, 1⟩, ⟨5819, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (-1)⟩)

def event106785 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26743⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6663⟩⟩) ⟨6603⟩ 5812)

def event106786 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26743⟩⟩, .relation 106785 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106787RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106787RawTermsValid :
    exact106787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106787 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26743⟩⟩) exact106787RawTerms .large 106780 (.finite 4741336194231092170536779776) (some (106782))

def event106788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23774⟩⟩) 0 ⟨6689⟩ 5477

def event106789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23774⟩⟩) 1 ⟨23773⟩ 101308

def event106790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23774⟩⟩) (.authority (.operator))

def exact106791RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (1)⟩]

theorem exact106791RawTermsValid :
    exact106791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106791 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23774⟩⟩) exact106791RawTerms .large 106790 .exactZero (none)

def event106792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26522⟩⟩) 0 ⟨23774⟩ 106791

def event106793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26522⟩⟩) (.authority (.operator))

def exact106794RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (1)⟩]

theorem exact106794RawTermsValid :
    exact106794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106794 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26522⟩⟩) exact106794RawTerms (.finite 8192) 106793 .exactZero (none)

def event106795 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26524⟩⟩) 0 ⟨24977⟩ 101568

def event106796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26524⟩⟩) 1 ⟨26522⟩ 106794

def event106797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26524⟩⟩) (.product (.predecessor 0 106795 .coefficient) (.predecessor 1 106796 .coefficient) (⟨false, false, none, none, none⟩))

def event106798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26524⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩) [⟨.result 106794 .coefficient, false, none⟩])

def event106799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26524⟩⟩) (.product (.result 101568 .summary) (.transfer 106798) (⟨false, false, none, none, none⟩))

def event106800 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26524⟩⟩, .operator (⟨101568, 0⟩, ⟨106794, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (1)⟩)

def event106801 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26524⟩⟩, .operator (⟨101568, 1⟩, ⟨106794, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (-1)⟩)

def event106802 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26524⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26522⟩⟩) ⟨23774⟩ 106791)

def event106803 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26524⟩⟩, .relation 106802 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (-1)⟩)

def exact106804RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (-1)⟩]

theorem exact106804RawTermsValid :
    exact106804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106804 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26524⟩⟩) exact106804RawTerms .large 106797 (.finite 1291900378790628425728) (some (106799))

def event106805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20453⟩⟩) 0 ⟨14944⟩ 4951

def event106806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20453⟩⟩) (.authority (.relationPreimageSource ⟨29⟩))

def exact106807RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩, (1)⟩]

theorem exact106807RawTermsValid :
    exact106807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106807 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20453⟩⟩) exact106807RawTerms (.finite 136065468) 106806 .exactZero (none)

def event106808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20455⟩⟩) 0 ⟨20453⟩ 106807

def event106809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20455⟩⟩) 1 ⟨2348⟩ 4

def event106810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20455⟩⟩) (.scale (.predecessor 0 106808 .coefficient) (.value (.predecessor 1 106809 .coefficient)))

def exact106811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩, (1)⟩]

theorem exact106811RawTermsValid :
    exact106811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20455⟩⟩) exact106811RawTerms (.finite 136065468) 106810 .exactZero (none)

def event106812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20456⟩⟩) 0 ⟨5509⟩ 94462

def event106813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20456⟩⟩) 1 ⟨20455⟩ 106811

def event106814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20456⟩⟩) (.product (.predecessor 0 106812 .coefficient) (.predecessor 1 106813 .coefficient) (⟨false, false, none, none, none⟩))

def event106815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20456⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩) [⟨.result 106807 .coefficient, false, none⟩])

def event106816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20456⟩⟩) (.product (.result 94462 .summary) (.transfer 106815) (⟨false, false, none, none, none⟩))

def event106817 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20456⟩⟩, .operator (⟨94462, 0⟩, ⟨106811, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩, (1)⟩)

def event106818 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20454⟩⟩)

def event106819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event106820 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event106821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event106822 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event106823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 106822

def event106824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 106820

def event106825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 106823 .coefficient) (.value (.predecessor 1 106824 .coefficient)))

def event106826 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event106827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10652⟩⟩) 0 ⟨5503⟩ 106826

def event106828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10652⟩⟩) (.authority (.programFamilyFact))

def exact106829RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10652⟩⟩], []⟩, (1)⟩]

theorem exact106829RawTermsValid :
    exact106829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10652⟩⟩) exact106829RawTerms (.finite 3) 106828 .exactZero (none)

def event106830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9490⟩⟩) 0 ⟨5503⟩ 106826

def event106831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9490⟩⟩) (.authority (.programFamilyFact))

def exact106832RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9490⟩⟩], []⟩, (1)⟩]

theorem exact106832RawTermsValid :
    exact106832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106832 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9490⟩⟩) exact106832RawTerms (.finite 3) 106831 .exactZero (none)

def event106833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10653⟩⟩) 0 ⟨9490⟩ 106832

def event106834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10653⟩⟩) 1 ⟨10652⟩ 106829

def event106835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10653⟩⟩) (.product (.predecessor 0 106833 .coefficient) (.predecessor 1 106834 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10653⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], []⟩) [⟨.result 106832 .coefficient, true, some 1⟩, ⟨.result 106829 .coefficient, true, some 1⟩])

def event106837 : Event := .survivorFold (1) 106836

def exact106838RawTerms : List Term := []

theorem exact106838RawTermsValid :
    exact106838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106838 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10653⟩⟩) exact106838RawTerms (.finite 9) 106835 (.finite 9) (some (106836))

def event106839 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10654⟩⟩) 0 ⟨10653⟩ 106838

def event106840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10654⟩⟩) (.identity (.predecessor 0 106839 .coefficient))

def event106841 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10654⟩⟩) (.finite 9)

def event106842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14943⟩⟩) 0 ⟨10654⟩ 106841

def event106843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14943⟩⟩) (.authority (.programFamilyFact))

def exact106844RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], []⟩, (1)⟩]

theorem exact106844RawTermsValid :
    exact106844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14943⟩⟩) exact106844RawTerms (.finite 3) 106843 .exactZero (none)

def event106845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14944⟩⟩) 0 ⟨14943⟩ 106844

def event106846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14944⟩⟩) (.identity (.predecessor 0 106845 .coefficient))

def event106847 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14944⟩⟩) (.finite 3)

def event106848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20453⟩⟩) 0 ⟨14944⟩ 106847

def event106849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20453⟩⟩) (.authority (.relationPreimageSource ⟨29⟩))

def exact106850RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩, (1)⟩]

theorem exact106850RawTermsValid :
    exact106850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106850 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20453⟩⟩) exact106850RawTerms (.finite 136065468) 106849 .exactZero (none)

def event106851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact106852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact106852RawTermsValid :
    exact106852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact106852RawTerms .large 106851 .exactZero (none)

def event106853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20454⟩⟩) 0 ⟨6⟩ 106852

def event106854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20454⟩⟩) 1 ⟨20453⟩ 106850

def event106855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20454⟩⟩) (.product (.predecessor 0 106853 .coefficient) (.predecessor 1 106854 .coefficient) (⟨false, false, none, none, none⟩))

def event106856 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20454⟩⟩, .operator (⟨106852, 0⟩, ⟨106850, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩, (1)⟩)

def exact106857RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩, (1)⟩]

theorem exact106857RawTermsValid :
    exact106857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106857 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20454⟩⟩) exact106857RawTerms .large 106855 .exactZero (none)

def event106858 : Event := .preFoldPolynomial 106857 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩, (1)⟩] .exactZero none

def exact106859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩, (1)⟩]

def event106859 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20454⟩⟩) 106858 exact106859RawTerms .large 106855 .exactZero (none)

def event106860 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26528⟩⟩)

def event106861 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event106862 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event106863 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event106864 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event106865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 106864

def event106866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 106862

def event106867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 106865 .coefficient) (.value (.predecessor 1 106866 .coefficient)))

def event106868 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event106869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10652⟩⟩) 0 ⟨5503⟩ 106868

def event106870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10652⟩⟩) (.authority (.programFamilyFact))

def exact106871RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10652⟩⟩], []⟩, (1)⟩]

theorem exact106871RawTermsValid :
    exact106871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10652⟩⟩) exact106871RawTerms (.finite 3) 106870 .exactZero (none)

def event106872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9490⟩⟩) 0 ⟨5503⟩ 106868

def event106873 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9490⟩⟩) (.authority (.programFamilyFact))

def exact106874RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9490⟩⟩], []⟩, (1)⟩]

theorem exact106874RawTermsValid :
    exact106874RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106874 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9490⟩⟩) exact106874RawTerms (.finite 3) 106873 .exactZero (none)

def event106875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10653⟩⟩) 0 ⟨9490⟩ 106874

def event106876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10653⟩⟩) 1 ⟨10652⟩ 106871

def event106877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10653⟩⟩) (.product (.predecessor 0 106875 .coefficient) (.predecessor 1 106876 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106878 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10653⟩⟩, .operator (⟨106874, 0⟩, ⟨106871, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], []⟩, (1)⟩)

def exact106879RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], []⟩, (1)⟩]

theorem exact106879RawTermsValid :
    exact106879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106879 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10653⟩⟩) exact106879RawTerms (.finite 9) 106877 .exactZero (none)

def event106880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10654⟩⟩) 0 ⟨10653⟩ 106879

def event106881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10654⟩⟩) (.identity (.predecessor 0 106880 .coefficient))

def event106882 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10654⟩⟩) (.finite 9)

def event106883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14943⟩⟩) 0 ⟨10654⟩ 106882

def event106884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14943⟩⟩) (.authority (.programFamilyFact))

def exact106885RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], []⟩, (1)⟩]

theorem exact106885RawTermsValid :
    exact106885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106885 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14943⟩⟩) exact106885RawTerms (.finite 3) 106884 .exactZero (none)

def event106886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14944⟩⟩) 0 ⟨14943⟩ 106885

def event106887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14944⟩⟩) (.identity (.predecessor 0 106886 .coefficient))

def event106888 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14944⟩⟩) (.finite 3)

def event106889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23773⟩⟩) 0 ⟨14944⟩ 106888

def event106890 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23773⟩⟩) (.authority (.programFamilyFact))

def event106891 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23773⟩⟩) (.finite 3720)

def event106892 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event106893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23774⟩⟩) 0 ⟨6689⟩ 106892

def event106894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23774⟩⟩) 1 ⟨23773⟩ 106891

def event106895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23774⟩⟩) (.authority (.operator))

def exact106896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (1)⟩]

theorem exact106896RawTermsValid :
    exact106896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106896 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23774⟩⟩) exact106896RawTerms .large 106895 .exactZero (none)

def event106897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26522⟩⟩) 0 ⟨23774⟩ 106896

def event106898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26522⟩⟩) (.authority (.operator))

def exact106899RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (1)⟩]

theorem exact106899RawTermsValid :
    exact106899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106899 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26522⟩⟩) exact106899RawTerms (.finite 8192) 106898 .exactZero (none)

def event106900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event106901 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event106902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14985⟩⟩) 0 ⟨14944⟩ 106888

def event106903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14985⟩⟩) 1 ⟨110⟩ 106901

def event106904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14985⟩⟩) (.sum [.predecessor 0 106902 .coefficient, .predecessor 1 106903 .coefficient])

def event106905 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14985⟩⟩) (.finite 3)

def event106906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14986⟩⟩) 0 ⟨14985⟩ 106905

def event106907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14986⟩⟩) (.identity (.predecessor 0 106906 .coefficient))

def exact106908RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], []⟩, (1)⟩]

theorem exact106908RawTermsValid :
    exact106908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106908 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14986⟩⟩) exact106908RawTerms (.finite 3) 106907 .exactZero (none)

def event106909 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact106910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106910RawTermsValid :
    exact106910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106910 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact106910RawTerms .large 106909 .exactZero (none)

def event106911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14987⟩⟩) 0 ⟨6544⟩ 106910

def event106912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14987⟩⟩) 1 ⟨14986⟩ 106908

def event106913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14987⟩⟩) (.product (.predecessor 0 106911 .coefficient) (.predecessor 1 106912 .coefficient) (⟨false, false, none, none, none⟩))

def event106914 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14987⟩⟩, .operator (⟨106910, 0⟩, ⟨106908, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact106915RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106915RawTermsValid :
    exact106915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106915 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14987⟩⟩) exact106915RawTerms .large 106913 .exactZero (none)

def event106916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6691⟩⟩) 0 ⟨6689⟩ 106892

def event106917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6691⟩⟩) (.authority (.operator))

def exact106918RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩]

theorem exact106918RawTermsValid :
    exact106918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6691⟩⟩) exact106918RawTerms .large 106917 .exactZero (none)

def event106919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14988⟩⟩) 0 ⟨6691⟩ 106918

def event106920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14988⟩⟩) 1 ⟨14987⟩ 106915

def event106921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14988⟩⟩) (.sum [.predecessor 0 106919 .coefficient, .predecessor 1 106920 .coefficient])

def exact106922RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106922RawTermsValid :
    exact106922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106922 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14988⟩⟩) exact106922RawTerms .large 106921 .exactZero (none)

def event106923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26523⟩⟩) 0 ⟨14988⟩ 106922

def event106924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26523⟩⟩) 1 ⟨26522⟩ 106899

def event106925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26523⟩⟩) (.product (.predecessor 0 106923 .coefficient) (.predecessor 1 106924 .coefficient) (⟨false, false, none, none, none⟩))

def event106926 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26523⟩⟩, .operator (⟨106922, 0⟩, ⟨106899, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (1)⟩)

def event106927 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26523⟩⟩, .operator (⟨106922, 1⟩, ⟨106899, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (-1)⟩)

def event106928 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26523⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26522⟩⟩) ⟨23774⟩ 106896)

def event106929 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26523⟩⟩, .relation 106928 0, ⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (-1)⟩)

def exact106930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (-1)⟩]

theorem exact106930RawTermsValid :
    exact106930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26523⟩⟩) exact106930RawTerms .large 106925 .exactZero (none)

def event106931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15034⟩⟩) 0 ⟨14944⟩ 106888

def event106932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15034⟩⟩) (.authority (.programFamilyFact))

def exact106933RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩]

theorem exact106933RawTermsValid :
    exact106933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15034⟩⟩) exact106933RawTerms (.finite 3) 106932 .exactZero (none)

def event106934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15037⟩⟩) 0 ⟨6544⟩ 106910

def event106935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15037⟩⟩) 1 ⟨15034⟩ 106933

def event106936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15037⟩⟩) (.product (.predecessor 0 106934 .coefficient) (.predecessor 1 106935 .coefficient) (⟨false, true, none, none, some 1⟩))

def event106937 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15037⟩⟩, .operator (⟨106910, 0⟩, ⟨106933, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact106938RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106938RawTermsValid :
    exact106938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106938 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15037⟩⟩) exact106938RawTerms .large 106936 .exactZero (none)

def event106939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6710⟩⟩) 0 ⟨6689⟩ 106892

def event106940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6710⟩⟩) (.authority (.operator))

def exact106941RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩]

theorem exact106941RawTermsValid :
    exact106941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106941 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6710⟩⟩) exact106941RawTerms .large 106940 .exactZero (none)

def event106942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15038⟩⟩) 0 ⟨6710⟩ 106941

def event106943 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15038⟩⟩) 1 ⟨15037⟩ 106938

def event106944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15038⟩⟩) (.sum [.predecessor 0 106942 .coefficient, .predecessor 1 106943 .coefficient])

def exact106945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106945RawTermsValid :
    exact106945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106945 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15038⟩⟩) exact106945RawTerms .large 106944 .exactZero (none)

def event106946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26528⟩⟩) 0 ⟨15038⟩ 106945

def event106947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26528⟩⟩) 1 ⟨26523⟩ 106930

def event106948 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26528⟩⟩) (.sum [.predecessor 0 106946 .coefficient, .predecessor 1 106947 .coefficient])

def exact106949RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106949RawTermsValid :
    exact106949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106949 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26528⟩⟩) exact106949RawTerms .large 106948 .exactZero (none)

def event106950 : Event := .preFoldPolynomial 106949 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact106951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event106951 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26528⟩⟩) 106950 exact106951RawTerms .large 106948 .exactZero (none)

def event106952 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14944⟩⟩) ⟨⟨123⟩, ⟨29⟩, ⟨109⟩⟩ ⟨106818, 106952⟩

def event106953 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20456⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩) (1) 0 2 (.universal 106952 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩) (none) 106951)

def event106954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20456⟩⟩, .relation 106953 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩)

def event106955 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20456⟩⟩, .relation 106953 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (-1)⟩)

def event106956 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20456⟩⟩, .relation 106953 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (1)⟩)

def event106957 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20456⟩⟩, .relation 106953 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106958RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106958RawTermsValid :
    exact106958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106958 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20456⟩⟩) exact106958RawTerms .large 106814 (.finite 1811303510016) (some (106816))

def event106959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26525⟩⟩) 0 ⟨20456⟩ 106958

def event106960 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26525⟩⟩) 1 ⟨26524⟩ 106804

def event106961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26525⟩⟩) (.sum [.predecessor 0 106959 .coefficient, .predecessor 1 106960 .coefficient])

def event106962 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26525⟩⟩, .operator (⟨106958, 0⟩, ⟨106804, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩, (1)⟩)

def event106963 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26525⟩⟩, .operator (⟨106958, 2⟩, ⟨106804, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23774⟩⟩]⟩, (-1)⟩)

def event106964 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26525⟩⟩) (.sum [.result 106958 .summary, .result 106804 .summary])

def exact106965RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106965RawTermsValid :
    exact106965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106965 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26525⟩⟩) exact106965RawTerms .large 106961 (.finite 1291900380601931935744) (some (106964))

def event106966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26526⟩⟩) 0 ⟨26525⟩ 106965

def event106967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26526⟩⟩) 1 ⟨6672⟩ 5839

def event106968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26526⟩⟩) (.product (.predecessor 0 106966 .coefficient) (.predecessor 1 106967 .coefficient) (⟨false, false, none, none, none⟩))

def event106969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26526⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) [⟨.result 5835 .coefficient, false, none⟩])

def event106970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26526⟩⟩) (.product (.result 106965 .summary) (.transfer 106969) (⟨false, false, none, none, none⟩))

def event106971 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26526⟩⟩, .operator (⟨106965, 0⟩, ⟨5839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩)

def event106972 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26526⟩⟩, .operator (⟨106965, 1⟩, ⟨5839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (-1)⟩)

def event106973 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26526⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6671⟩⟩) ⟨6607⟩ 5832)

def event106974 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26526⟩⟩, .relation 106973 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106975RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106975RawTermsValid :
    exact106975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106975 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26526⟩⟩) exact106975RawTerms .large 106968 (.finite 4741295067215179835091451904) (some (106970))

def event106976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23711⟩⟩) 0 ⟨6689⟩ 5477

def event106977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23711⟩⟩) 1 ⟨23710⟩ 101742

def event106978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23711⟩⟩) (.authority (.operator))

def exact106979RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (1)⟩]

theorem exact106979RawTermsValid :
    exact106979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106979 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23711⟩⟩) exact106979RawTerms .large 106978 .exactZero (none)

def event106980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26319⟩⟩) 0 ⟨23711⟩ 106979

def event106981 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26319⟩⟩) (.authority (.operator))

def exact106982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (1)⟩]

theorem exact106982RawTermsValid :
    exact106982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106982 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26319⟩⟩) exact106982RawTerms (.finite 8192) 106981 .exactZero (none)

def event106983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26321⟩⟩) 0 ⟨24900⟩ 102002

def event106984 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26321⟩⟩) 1 ⟨26319⟩ 106982

def event106985 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26321⟩⟩) (.product (.predecessor 0 106983 .coefficient) (.predecessor 1 106984 .coefficient) (⟨false, false, none, none, none⟩))

def event106986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26321⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩) [⟨.result 106982 .coefficient, false, none⟩])

def event106987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26321⟩⟩) (.product (.result 102002 .summary) (.transfer 106986) (⟨false, false, none, none, none⟩))

def event106988 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26321⟩⟩, .operator (⟨102002, 0⟩, ⟨106982, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (1)⟩)

def event106989 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26321⟩⟩, .operator (⟨102002, 1⟩, ⟨106982, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (-1)⟩)

def event106990 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26321⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26319⟩⟩) ⟨23711⟩ 106979)

def event106991 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26321⟩⟩, .relation 106990 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (-1)⟩)

def exact106992RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (-1)⟩]

theorem exact106992RawTermsValid :
    exact106992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106992 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26321⟩⟩) exact106992RawTerms .large 106985 (.finite 1291889172568118132736) (some (106987))

def event106993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20309⟩⟩) 0 ⟨14783⟩ 4974

def event106994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20309⟩⟩) (.authority (.relationPreimageSource ⟨27⟩))

def exact106995RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩, (1)⟩]

theorem exact106995RawTermsValid :
    exact106995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106995 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20309⟩⟩) exact106995RawTerms (.finite 136065468) 106994 .exactZero (none)

def event106996 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20311⟩⟩) 0 ⟨20309⟩ 106995

def event106997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20311⟩⟩) 1 ⟨2348⟩ 4

def event106998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20311⟩⟩) (.scale (.predecessor 0 106996 .coefficient) (.value (.predecessor 1 106997 .coefficient)))

def exact106999RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩, (1)⟩]

theorem exact106999RawTermsValid :
    exact106999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106999 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20311⟩⟩) exact106999RawTerms (.finite 136065468) 106998 .exactZero (none)

def event107000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20312⟩⟩) 0 ⟨5509⟩ 94462

def event107001 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20312⟩⟩) 1 ⟨20311⟩ 106999

def event107002 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20312⟩⟩) (.product (.predecessor 0 107000 .coefficient) (.predecessor 1 107001 .coefficient) (⟨false, false, none, none, none⟩))

def event107003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20312⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩) [⟨.result 106995 .coefficient, false, none⟩])

def event107004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20312⟩⟩) (.product (.result 94462 .summary) (.transfer 107003) (⟨false, false, none, none, none⟩))

def event107005 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20312⟩⟩, .operator (⟨94462, 0⟩, ⟨106999, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩, (1)⟩)

def event107006 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20310⟩⟩)

def event107007 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def eventLeaf6672 : Array AnnotatedEvent := #[
  { event := event106752
    frameStart := 106672 },
  { event := event106753
    frameStart := 106672 },
  { event := event106754
    frameStart := 106672 },
  { event := event106755
    frameStart := 106672 },
  { event := event106756
    frameStart := 106672 },
  { event := event106757
    frameStart := 106672 },
  { event := event106758
    frameStart := 106672 },
  { event := event106759
    frameStart := 106672 },
  { event := event106760
    frameStart := 106672 },
  { event := event106761
    frameStart := 106672 },
  { event := event106762
    frameStart := 106672 },
  { event := event106763
    frameStart := 106672 },
  { event := event106764
    frameStart := 0 },
  { event := event106765
    frameStart := 0 },
  { event := event106766
    frameStart := 0 },
  { event := event106767
    frameStart := 0 }
]

def eventLeaf6673 : Array AnnotatedEvent := #[
  { event := event106768
    frameStart := 0 },
  { event := event106769
    frameStart := 0 },
  { event := event106770
    frameStart := 0 },
  { event := event106771
    frameStart := 0 },
  { event := event106772
    frameStart := 0 },
  { event := event106773
    frameStart := 0 },
  { event := event106774
    frameStart := 0 },
  { event := event106775
    frameStart := 0 },
  { event := event106776
    frameStart := 0 },
  { event := event106777
    frameStart := 0 },
  { event := event106778
    frameStart := 0 },
  { event := event106779
    frameStart := 0 },
  { event := event106780
    frameStart := 0 },
  { event := event106781
    frameStart := 0 },
  { event := event106782
    frameStart := 0 },
  { event := event106783
    frameStart := 0 }
]

def eventLeaf6674 : Array AnnotatedEvent := #[
  { event := event106784
    frameStart := 0 },
  { event := event106785
    frameStart := 0 },
  { event := event106786
    frameStart := 0 },
  { event := event106787
    frameStart := 0 },
  { event := event106788
    frameStart := 0 },
  { event := event106789
    frameStart := 0 },
  { event := event106790
    frameStart := 0 },
  { event := event106791
    frameStart := 0 },
  { event := event106792
    frameStart := 0 },
  { event := event106793
    frameStart := 0 },
  { event := event106794
    frameStart := 0 },
  { event := event106795
    frameStart := 0 },
  { event := event106796
    frameStart := 0 },
  { event := event106797
    frameStart := 0 },
  { event := event106798
    frameStart := 0 },
  { event := event106799
    frameStart := 0 }
]

def eventLeaf6675 : Array AnnotatedEvent := #[
  { event := event106800
    frameStart := 0 },
  { event := event106801
    frameStart := 0 },
  { event := event106802
    frameStart := 0 },
  { event := event106803
    frameStart := 0 },
  { event := event106804
    frameStart := 0 },
  { event := event106805
    frameStart := 0 },
  { event := event106806
    frameStart := 0 },
  { event := event106807
    frameStart := 0 },
  { event := event106808
    frameStart := 0 },
  { event := event106809
    frameStart := 0 },
  { event := event106810
    frameStart := 0 },
  { event := event106811
    frameStart := 0 },
  { event := event106812
    frameStart := 0 },
  { event := event106813
    frameStart := 0 },
  { event := event106814
    frameStart := 0 },
  { event := event106815
    frameStart := 0 }
]

def eventLeaf6676 : Array AnnotatedEvent := #[
  { event := event106816
    frameStart := 0 },
  { event := event106817
    frameStart := 0 },
  { event := event106818
    frameStart := 106818 },
  { event := event106819
    frameStart := 106818 },
  { event := event106820
    frameStart := 106818 },
  { event := event106821
    frameStart := 106818 },
  { event := event106822
    frameStart := 106818 },
  { event := event106823
    frameStart := 106818 },
  { event := event106824
    frameStart := 106818 },
  { event := event106825
    frameStart := 106818 },
  { event := event106826
    frameStart := 106818 },
  { event := event106827
    frameStart := 106818 },
  { event := event106828
    frameStart := 106818 },
  { event := event106829
    frameStart := 106818 },
  { event := event106830
    frameStart := 106818 },
  { event := event106831
    frameStart := 106818 }
]

def eventLeaf6677 : Array AnnotatedEvent := #[
  { event := event106832
    frameStart := 106818 },
  { event := event106833
    frameStart := 106818 },
  { event := event106834
    frameStart := 106818 },
  { event := event106835
    frameStart := 106818 },
  { event := event106836
    frameStart := 106818 },
  { event := event106837
    frameStart := 106818 },
  { event := event106838
    frameStart := 106818 },
  { event := event106839
    frameStart := 106818 },
  { event := event106840
    frameStart := 106818 },
  { event := event106841
    frameStart := 106818 },
  { event := event106842
    frameStart := 106818 },
  { event := event106843
    frameStart := 106818 },
  { event := event106844
    frameStart := 106818 },
  { event := event106845
    frameStart := 106818 },
  { event := event106846
    frameStart := 106818 },
  { event := event106847
    frameStart := 106818 }
]

def eventLeaf6678 : Array AnnotatedEvent := #[
  { event := event106848
    frameStart := 106818 },
  { event := event106849
    frameStart := 106818 },
  { event := event106850
    frameStart := 106818 },
  { event := event106851
    frameStart := 106818 },
  { event := event106852
    frameStart := 106818 },
  { event := event106853
    frameStart := 106818 },
  { event := event106854
    frameStart := 106818 },
  { event := event106855
    frameStart := 106818 },
  { event := event106856
    frameStart := 106818 },
  { event := event106857
    frameStart := 106818 },
  { event := event106858
    frameStart := 106818 },
  { event := event106859
    frameStart := 106818 },
  { event := event106860
    frameStart := 106860 },
  { event := event106861
    frameStart := 106860 },
  { event := event106862
    frameStart := 106860 },
  { event := event106863
    frameStart := 106860 }
]

def eventLeaf6679 : Array AnnotatedEvent := #[
  { event := event106864
    frameStart := 106860 },
  { event := event106865
    frameStart := 106860 },
  { event := event106866
    frameStart := 106860 },
  { event := event106867
    frameStart := 106860 },
  { event := event106868
    frameStart := 106860 },
  { event := event106869
    frameStart := 106860 },
  { event := event106870
    frameStart := 106860 },
  { event := event106871
    frameStart := 106860 },
  { event := event106872
    frameStart := 106860 },
  { event := event106873
    frameStart := 106860 },
  { event := event106874
    frameStart := 106860 },
  { event := event106875
    frameStart := 106860 },
  { event := event106876
    frameStart := 106860 },
  { event := event106877
    frameStart := 106860 },
  { event := event106878
    frameStart := 106860 },
  { event := event106879
    frameStart := 106860 }
]

def eventLeaf6680 : Array AnnotatedEvent := #[
  { event := event106880
    frameStart := 106860 },
  { event := event106881
    frameStart := 106860 },
  { event := event106882
    frameStart := 106860 },
  { event := event106883
    frameStart := 106860 },
  { event := event106884
    frameStart := 106860 },
  { event := event106885
    frameStart := 106860 },
  { event := event106886
    frameStart := 106860 },
  { event := event106887
    frameStart := 106860 },
  { event := event106888
    frameStart := 106860 },
  { event := event106889
    frameStart := 106860 },
  { event := event106890
    frameStart := 106860 },
  { event := event106891
    frameStart := 106860 },
  { event := event106892
    frameStart := 106860 },
  { event := event106893
    frameStart := 106860 },
  { event := event106894
    frameStart := 106860 },
  { event := event106895
    frameStart := 106860 }
]

def eventLeaf6681 : Array AnnotatedEvent := #[
  { event := event106896
    frameStart := 106860 },
  { event := event106897
    frameStart := 106860 },
  { event := event106898
    frameStart := 106860 },
  { event := event106899
    frameStart := 106860 },
  { event := event106900
    frameStart := 106860 },
  { event := event106901
    frameStart := 106860 },
  { event := event106902
    frameStart := 106860 },
  { event := event106903
    frameStart := 106860 },
  { event := event106904
    frameStart := 106860 },
  { event := event106905
    frameStart := 106860 },
  { event := event106906
    frameStart := 106860 },
  { event := event106907
    frameStart := 106860 },
  { event := event106908
    frameStart := 106860 },
  { event := event106909
    frameStart := 106860 },
  { event := event106910
    frameStart := 106860 },
  { event := event106911
    frameStart := 106860 }
]

def eventLeaf6682 : Array AnnotatedEvent := #[
  { event := event106912
    frameStart := 106860 },
  { event := event106913
    frameStart := 106860 },
  { event := event106914
    frameStart := 106860 },
  { event := event106915
    frameStart := 106860 },
  { event := event106916
    frameStart := 106860 },
  { event := event106917
    frameStart := 106860 },
  { event := event106918
    frameStart := 106860 },
  { event := event106919
    frameStart := 106860 },
  { event := event106920
    frameStart := 106860 },
  { event := event106921
    frameStart := 106860 },
  { event := event106922
    frameStart := 106860 },
  { event := event106923
    frameStart := 106860 },
  { event := event106924
    frameStart := 106860 },
  { event := event106925
    frameStart := 106860 },
  { event := event106926
    frameStart := 106860 },
  { event := event106927
    frameStart := 106860 }
]

def eventLeaf6683 : Array AnnotatedEvent := #[
  { event := event106928
    frameStart := 106860 },
  { event := event106929
    frameStart := 106860 },
  { event := event106930
    frameStart := 106860 },
  { event := event106931
    frameStart := 106860 },
  { event := event106932
    frameStart := 106860 },
  { event := event106933
    frameStart := 106860 },
  { event := event106934
    frameStart := 106860 },
  { event := event106935
    frameStart := 106860 },
  { event := event106936
    frameStart := 106860 },
  { event := event106937
    frameStart := 106860 },
  { event := event106938
    frameStart := 106860 },
  { event := event106939
    frameStart := 106860 },
  { event := event106940
    frameStart := 106860 },
  { event := event106941
    frameStart := 106860 },
  { event := event106942
    frameStart := 106860 },
  { event := event106943
    frameStart := 106860 }
]

def eventLeaf6684 : Array AnnotatedEvent := #[
  { event := event106944
    frameStart := 106860 },
  { event := event106945
    frameStart := 106860 },
  { event := event106946
    frameStart := 106860 },
  { event := event106947
    frameStart := 106860 },
  { event := event106948
    frameStart := 106860 },
  { event := event106949
    frameStart := 106860 },
  { event := event106950
    frameStart := 106860 },
  { event := event106951
    frameStart := 106860 },
  { event := event106952
    frameStart := 0 },
  { event := event106953
    frameStart := 0 },
  { event := event106954
    frameStart := 0 },
  { event := event106955
    frameStart := 0 },
  { event := event106956
    frameStart := 0 },
  { event := event106957
    frameStart := 0 },
  { event := event106958
    frameStart := 0 },
  { event := event106959
    frameStart := 0 }
]

def eventLeaf6685 : Array AnnotatedEvent := #[
  { event := event106960
    frameStart := 0 },
  { event := event106961
    frameStart := 0 },
  { event := event106962
    frameStart := 0 },
  { event := event106963
    frameStart := 0 },
  { event := event106964
    frameStart := 0 },
  { event := event106965
    frameStart := 0 },
  { event := event106966
    frameStart := 0 },
  { event := event106967
    frameStart := 0 },
  { event := event106968
    frameStart := 0 },
  { event := event106969
    frameStart := 0 },
  { event := event106970
    frameStart := 0 },
  { event := event106971
    frameStart := 0 },
  { event := event106972
    frameStart := 0 },
  { event := event106973
    frameStart := 0 },
  { event := event106974
    frameStart := 0 },
  { event := event106975
    frameStart := 0 }
]

def eventLeaf6686 : Array AnnotatedEvent := #[
  { event := event106976
    frameStart := 0 },
  { event := event106977
    frameStart := 0 },
  { event := event106978
    frameStart := 0 },
  { event := event106979
    frameStart := 0 },
  { event := event106980
    frameStart := 0 },
  { event := event106981
    frameStart := 0 },
  { event := event106982
    frameStart := 0 },
  { event := event106983
    frameStart := 0 },
  { event := event106984
    frameStart := 0 },
  { event := event106985
    frameStart := 0 },
  { event := event106986
    frameStart := 0 },
  { event := event106987
    frameStart := 0 },
  { event := event106988
    frameStart := 0 },
  { event := event106989
    frameStart := 0 },
  { event := event106990
    frameStart := 0 },
  { event := event106991
    frameStart := 0 }
]

def eventLeaf6687 : Array AnnotatedEvent := #[
  { event := event106992
    frameStart := 0 },
  { event := event106993
    frameStart := 0 },
  { event := event106994
    frameStart := 0 },
  { event := event106995
    frameStart := 0 },
  { event := event106996
    frameStart := 0 },
  { event := event106997
    frameStart := 0 },
  { event := event106998
    frameStart := 0 },
  { event := event106999
    frameStart := 0 },
  { event := event107000
    frameStart := 0 },
  { event := event107001
    frameStart := 0 },
  { event := event107002
    frameStart := 0 },
  { event := event107003
    frameStart := 0 },
  { event := event107004
    frameStart := 0 },
  { event := event107005
    frameStart := 0 },
  { event := event107006
    frameStart := 107006 },
  { event := event107007
    frameStart := 107006 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events417
