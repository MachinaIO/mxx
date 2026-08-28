import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events378

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event96768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56092⟩⟩) (.sum [.predecessor 0 96766 .coefficient, .predecessor 1 96767 .coefficient])

def exact96769RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56087⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53908⟩⟩], [⟨.program ⟨257⟩, ⟨55186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96769RawTermsValid :
    exact96769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96769 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56092⟩⟩) exact96769RawTerms .large 96768 .exactZero (none)

def event96770 : Event := .preFoldPolynomial 96769 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56087⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53908⟩⟩], [⟨.program ⟨257⟩, ⟨55186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact96771RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56087⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53908⟩⟩], [⟨.program ⟨257⟩, ⟨55186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event96771 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨56092⟩⟩) 96770 exact96771RawTerms .large 96768 .exactZero (none)

def event96772 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53909⟩⟩) ⟨⟨87⟩, ⟨68⟩, ⟨135⟩⟩ ⟨96614, 96772⟩

def event96773 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54839⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54836⟩⟩]⟩) (1) 0 2 (.universal 96772 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54836⟩⟩]⟩) (none) 96771)

def event96774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54839⟩⟩, .relation 96773 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩)

def event96775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54839⟩⟩, .relation 96773 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56087⟩⟩]⟩, (-1)⟩)

def event96776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54839⟩⟩, .relation 96773 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨53908⟩⟩], [⟨.program ⟨257⟩, ⟨55186⟩⟩]⟩, (1)⟩)

def event96777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54839⟩⟩, .relation 96773 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact96778RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56087⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨53908⟩⟩], [⟨.program ⟨257⟩, ⟨55186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96778RawTermsValid :
    exact96778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96778 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54839⟩⟩) exact96778RawTerms .large 96610 (.finite 202072841853861888) (some (96612))

def event96779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56090⟩⟩) 0 ⟨54839⟩ 96778

def event96780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56090⟩⟩) 1 ⟨56089⟩ 96600

def event96781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56090⟩⟩) (.sum [.predecessor 0 96779 .coefficient, .predecessor 1 96780 .coefficient])

def event96782 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56090⟩⟩, .operator (⟨96778, 0⟩, ⟨96600, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56087⟩⟩]⟩, (1)⟩)

def event96783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56090⟩⟩, .operator (⟨96778, 2⟩, ⟨96600, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨53908⟩⟩], [⟨.program ⟨257⟩, ⟨55186⟩⟩]⟩, (-1)⟩)

def event96784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56090⟩⟩) (.sum [.result 96778 .summary, .result 96600 .summary])

def exact96785RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96785RawTermsValid :
    exact96785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56090⟩⟩) exact96785RawTerms .large 96781 (.finite 32189789464712143775715074244608) (some (96784))

def event96786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52204⟩⟩) 0 ⟨50929⟩ 4150

def event96787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52204⟩⟩) (.authority (.programFamilyFact))

def event96788 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52204⟩⟩) (.finite 3720)

def event96789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52206⟩⟩) 0 ⟨7177⟩ 15500

def event96790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52206⟩⟩) 1 ⟨52204⟩ 96788

def event96791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52206⟩⟩) (.authority (.operator))

def exact96792RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (1)⟩]

theorem exact96792RawTermsValid :
    exact96792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96792 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52206⟩⟩) exact96792RawTerms .large 96791 .exactZero (none)

def event96793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53107⟩⟩) 0 ⟨52206⟩ 96792

def event96794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53107⟩⟩) (.authority (.operator))

def exact96795RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (1)⟩]

theorem exact96795RawTermsValid :
    exact96795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53107⟩⟩) exact96795RawTerms (.finite 8192) 96794 .exactZero (none)

def event96796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52038⟩⟩) 0 ⟨50682⟩ 4144

def event96797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52038⟩⟩) (.authority (.programFamilyFact))

def event96798 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52038⟩⟩) (.finite 3720)

def event96799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52039⟩⟩) 0 ⟨7177⟩ 15500

def event96800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52039⟩⟩) 1 ⟨52038⟩ 96798

def event96801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52039⟩⟩) (.authority (.operator))

def exact96802RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (1)⟩]

theorem exact96802RawTermsValid :
    exact96802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96802 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52039⟩⟩) exact96802RawTerms .large 96801 .exactZero (none)

def event96803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52574⟩⟩) 0 ⟨52039⟩ 96802

def event96804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52574⟩⟩) (.authority (.operator))

def exact96805RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (1)⟩]

theorem exact96805RawTermsValid :
    exact96805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96805 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52574⟩⟩) exact96805RawTerms (.finite 8192) 96804 .exactZero (none)

def event96806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24591⟩⟩) 0 ⟨24590⟩ 4133

def event96807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24591⟩⟩) 1 ⟨9904⟩ 90528

def event96808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24591⟩⟩) (.tensor (.predecessor 0 96806 .coefficient) (.predecessor 1 96807 .coefficient) true false)

def event96809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24591⟩⟩, .operator (⟨4133, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact96810RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact96810RawTermsValid :
    exact96810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96810 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24591⟩⟩) exact96810RawTerms .large 96808 .exactZero (none)

def event96811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9942⟩⟩) 0 ⟨9903⟩ 90398

def event96812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9942⟩⟩) 1 ⟨7308⟩ 23593

def event96813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9942⟩⟩) (.product (.predecessor 0 96811 .coefficient) (.predecessor 1 96812 .coefficient) (⟨false, false, none, none, none⟩))

def event96814 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9942⟩⟩, .operator (⟨90398, 0⟩, ⟨23593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def exact96815RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact96815RawTermsValid :
    exact96815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96815 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9942⟩⟩) exact96815RawTerms .large 96813 .exactZero (none)

def event96816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24592⟩⟩) 0 ⟨9942⟩ 96815

def event96817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24592⟩⟩) 1 ⟨24591⟩ 96810

def event96818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24592⟩⟩) (.sum [.predecessor 0 96816 .coefficient, .predecessor 1 96817 .coefficient])

def exact96819RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96819RawTermsValid :
    exact96819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96819 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24592⟩⟩) exact96819RawTerms .large 96818 .exactZero (none)

def event96820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24593⟩⟩) 0 ⟨24592⟩ 96819

def event96821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24593⟩⟩) 1 ⟨134⟩ 23585

def event96822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24593⟩⟩) (.sum [.predecessor 0 96820 .coefficient, .predecessor 1 96821 .coefficient])

def event96823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24593⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨134⟩⟩]⟩) [⟨.result 23585 .coefficient, false, none⟩])

def event96824 : Event := .survivorFold (1) 96823

def exact96825RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96825RawTermsValid :
    exact96825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24593⟩⟩) exact96825RawTerms .large 96822 (.finite 26) (some (96823))

def event96826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50683⟩⟩) 0 ⟨24593⟩ 96825

def event96827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50683⟩⟩) 1 ⟨50680⟩ 4136

def event96828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50683⟩⟩) (.product (.predecessor 0 96826 .coefficient) (.predecessor 1 96827 .coefficient) (⟨false, true, none, none, some 1⟩))

def event96829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50683⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩) [⟨.result 4136 .coefficient, true, some 1⟩])

def event96830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50683⟩⟩) (.product (.result 96825 .summary) (.transfer 96829) (⟨false, false, none, none, none⟩))

def event96831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50683⟩⟩, .operator (⟨96825, 1⟩, ⟨4136, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event96832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50683⟩⟩, .operator (⟨96825, 0⟩, ⟨4136, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def exact96833RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact96833RawTermsValid :
    exact96833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50683⟩⟩) exact96833RawTerms .large 96828 (.finite 8519680) (some (96830))

def event96834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50684⟩⟩) 0 ⟨50680⟩ 4136

def event96835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50684⟩⟩) 1 ⟨9904⟩ 90528

def event96836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50684⟩⟩) (.tensor (.predecessor 0 96834 .coefficient) (.predecessor 1 96835 .coefficient) true false)

def event96837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50684⟩⟩, .operator (⟨4136, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact96838RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact96838RawTermsValid :
    exact96838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96838 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50684⟩⟩) exact96838RawTerms .large 96836 .exactZero (none)

def event96839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9922⟩⟩) 0 ⟨9903⟩ 90398

def event96840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9922⟩⟩) 1 ⟨7288⟩ 23634

def event96841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9922⟩⟩) (.product (.predecessor 0 96839 .coefficient) (.predecessor 1 96840 .coefficient) (⟨false, false, none, none, none⟩))

def event96842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9922⟩⟩, .operator (⟨90398, 0⟩, ⟨23634, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩)

def exact96843RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩]

theorem exact96843RawTermsValid :
    exact96843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9922⟩⟩) exact96843RawTerms .large 96841 .exactZero (none)

def event96844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50685⟩⟩) 0 ⟨9922⟩ 96843

def event96845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50685⟩⟩) 1 ⟨50684⟩ 96838

def event96846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50685⟩⟩) (.sum [.predecessor 0 96844 .coefficient, .predecessor 1 96845 .coefficient])

def exact96847RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96847RawTermsValid :
    exact96847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96847 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50685⟩⟩) exact96847RawTerms .large 96846 .exactZero (none)

def event96848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50686⟩⟩) 0 ⟨50685⟩ 96847

def event96849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50686⟩⟩) 1 ⟨114⟩ 23626

def event96850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50686⟩⟩) (.sum [.predecessor 0 96848 .coefficient, .predecessor 1 96849 .coefficient])

def event96851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50686⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨114⟩⟩]⟩) [⟨.result 23626 .coefficient, false, none⟩])

def event96852 : Event := .survivorFold (1) 96851

def exact96853RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96853RawTermsValid :
    exact96853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96853 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50686⟩⟩) exact96853RawTerms .large 96850 (.finite 26) (some (96851))

def event96854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50687⟩⟩) 0 ⟨50686⟩ 96853

def event96855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50687⟩⟩) 1 ⟨9581⟩ 23623

def event96856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50687⟩⟩) (.product (.predecessor 0 96854 .coefficient) (.predecessor 1 96855 .coefficient) (⟨false, false, none, none, none⟩))

def event96857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50687⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) [⟨.result 23619 .coefficient, false, none⟩])

def event96858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50687⟩⟩) (.product (.result 96853 .summary) (.transfer 96857) (⟨false, false, none, none, none⟩))

def event96859 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50687⟩⟩, .operator (⟨96853, 1⟩, ⟨23623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (-1)⟩)

def event96860 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50687⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9580⟩⟩) ⟨7308⟩ 23593)

def event96861 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50687⟩⟩, .relation 96860 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (-1)⟩)

def event96862 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50687⟩⟩, .operator (⟨96853, 0⟩, ⟨23623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩)

def exact96863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (-1)⟩]

theorem exact96863RawTermsValid :
    exact96863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50687⟩⟩) exact96863RawTerms .large 96856 (.finite 279172874240) (some (96858))

def event96864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50688⟩⟩) 0 ⟨50687⟩ 96863

def event96865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50688⟩⟩) 1 ⟨50683⟩ 96833

def event96866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50688⟩⟩) (.sum [.predecessor 0 96864 .coefficient, .predecessor 1 96865 .coefficient])

def event96867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50688⟩⟩, .operator (⟨96863, 1⟩, ⟨96833, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def event96868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50688⟩⟩) (.sum [.result 96863 .summary, .result 96833 .summary])

def exact96869RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96869RawTermsValid :
    exact96869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50688⟩⟩) exact96869RawTerms .large 96866 (.finite 279181393920) (some (96868))

def event96870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52575⟩⟩) 0 ⟨50688⟩ 96869

def event96871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52575⟩⟩) 1 ⟨52574⟩ 96805

def event96872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52575⟩⟩) (.product (.predecessor 0 96870 .coefficient) (.predecessor 1 96871 .coefficient) (⟨false, false, none, none, none⟩))

def event96873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52575⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩) [⟨.result 96805 .coefficient, false, none⟩])

def event96874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52575⟩⟩) (.product (.result 96869 .summary) (.transfer 96873) (⟨false, false, none, none, none⟩))

def event96875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52575⟩⟩, .operator (⟨96869, 1⟩, ⟨96805, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (-1)⟩)

def event96876 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52575⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52574⟩⟩) ⟨52039⟩ 96802)

def event96877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52575⟩⟩, .relation 96876 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (-1)⟩)

def event96878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52575⟩⟩, .operator (⟨96869, 0⟩, ⟨96805, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (1)⟩)

def exact96879RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (-1)⟩]

theorem exact96879RawTermsValid :
    exact96879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52575⟩⟩) exact96879RawTerms .large 96872 (.finite 2997687391345233100800) (some (96874))

def event96880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51499⟩⟩) 0 ⟨50682⟩ 4144

def event96881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51499⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact96882RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩, (1)⟩]

theorem exact96882RawTermsValid :
    exact96882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96882 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51499⟩⟩) exact96882RawTerms (.finite 5647228698) 96881 .exactZero (none)

def event96883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51501⟩⟩) 0 ⟨51499⟩ 96882

def event96884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51501⟩⟩) 1 ⟨2370⟩ 4

def event96885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51501⟩⟩) (.scale (.predecessor 0 96883 .coefficient) (.value (.predecessor 1 96884 .coefficient)))

def exact96886RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩, (1)⟩]

theorem exact96886RawTermsValid :
    exact96886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51501⟩⟩) exact96886RawTerms (.finite 5647228698) 96885 .exactZero (none)

def event96887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51502⟩⟩) 0 ⟨9944⟩ 90620

def event96888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51502⟩⟩) 1 ⟨51501⟩ 96886

def event96889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51502⟩⟩) (.product (.predecessor 0 96887 .coefficient) (.predecessor 1 96888 .coefficient) (⟨false, false, none, none, none⟩))

def event96890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51502⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩) [⟨.result 96882 .coefficient, false, none⟩])

def event96891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51502⟩⟩) (.product (.result 90620 .summary) (.transfer 96890) (⟨false, false, none, none, none⟩))

def event96892 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51502⟩⟩, .operator (⟨90620, 0⟩, ⟨96886, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩, (1)⟩)

def event96893 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51500⟩⟩)

def event96894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event96895 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event96896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event96897 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event96898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event96899 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event96900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event96901 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event96902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 96901

def event96903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 96899

def event96904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 96902 .coefficient) (.value (.predecessor 1 96903 .coefficient)))

def event96905 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event96906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 96905

def event96907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 96897

def event96908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 96906 .coefficient, .predecessor 1 96907 .coefficient])

def event96909 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event96910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 96909

def event96911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 96895

def event96912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 96911 .coefficient))

def event96913 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event96914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24590⟩⟩) 0 ⟨9901⟩ 96913

def event96915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24590⟩⟩) (.authority (.programFamilyFact))

def exact96916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩], []⟩, (1)⟩]

theorem exact96916RawTermsValid :
    exact96916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24590⟩⟩) exact96916RawTerms (.finite 10) 96915 .exactZero (none)

def event96917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50680⟩⟩) 0 ⟨9901⟩ 96913

def event96918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50680⟩⟩) (.authority (.programFamilyFact))

def exact96919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩]

theorem exact96919RawTermsValid :
    exact96919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50680⟩⟩) exact96919RawTerms (.finite 10) 96918 .exactZero (none)

def event96920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50681⟩⟩) 0 ⟨50680⟩ 96919

def event96921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50681⟩⟩) 1 ⟨24590⟩ 96916

def event96922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50681⟩⟩) (.product (.predecessor 0 96920 .coefficient) (.predecessor 1 96921 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50681⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩) [⟨.result 96919 .coefficient, true, some 1⟩, ⟨.result 96916 .coefficient, true, some 1⟩])

def event96924 : Event := .survivorFold (1) 96923

def exact96925RawTerms : List Term := []

theorem exact96925RawTermsValid :
    exact96925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96925 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50681⟩⟩) exact96925RawTerms (.finite 100) 96922 (.finite 100) (some (96923))

def event96926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50682⟩⟩) 0 ⟨50681⟩ 96925

def event96927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50682⟩⟩) (.identity (.predecessor 0 96926 .coefficient))

def event96928 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50682⟩⟩) (.finite 100)

def event96929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51499⟩⟩) 0 ⟨50682⟩ 96928

def event96930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51499⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact96931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩, (1)⟩]

theorem exact96931RawTermsValid :
    exact96931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51499⟩⟩) exact96931RawTerms (.finite 5647228698) 96930 .exactZero (none)

def event96932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact96933RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact96933RawTermsValid :
    exact96933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96933 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact96933RawTerms .large 96932 .exactZero (none)

def event96934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51500⟩⟩) 0 ⟨35⟩ 96933

def event96935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51500⟩⟩) 1 ⟨51499⟩ 96931

def event96936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51500⟩⟩) (.product (.predecessor 0 96934 .coefficient) (.predecessor 1 96935 .coefficient) (⟨false, false, none, none, none⟩))

def event96937 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51500⟩⟩, .operator (⟨96933, 0⟩, ⟨96931, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩, (1)⟩)

def exact96938RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩, (1)⟩]

theorem exact96938RawTermsValid :
    exact96938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51500⟩⟩) exact96938RawTerms .large 96936 .exactZero (none)

def event96939 : Event := .preFoldPolynomial 96938 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩, (1)⟩] .exactZero none

def exact96940RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩, (1)⟩]

def event96940 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51500⟩⟩) 96939 exact96940RawTerms .large 96936 .exactZero (none)

def event96941 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨52578⟩⟩)

def event96942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event96943 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event96944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event96945 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event96946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event96947 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event96948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event96949 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event96950 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 96949

def event96951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 96947

def event96952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 96950 .coefficient) (.value (.predecessor 1 96951 .coefficient)))

def event96953 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event96954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 96953

def event96955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 96945

def event96956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 96954 .coefficient, .predecessor 1 96955 .coefficient])

def event96957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event96958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 96957

def event96959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 96943

def event96960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 96959 .coefficient))

def event96961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event96962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24590⟩⟩) 0 ⟨9901⟩ 96961

def event96963 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24590⟩⟩) (.authority (.programFamilyFact))

def exact96964RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩], []⟩, (1)⟩]

theorem exact96964RawTermsValid :
    exact96964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96964 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24590⟩⟩) exact96964RawTerms (.finite 10) 96963 .exactZero (none)

def event96965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50680⟩⟩) 0 ⟨9901⟩ 96961

def event96966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50680⟩⟩) (.authority (.programFamilyFact))

def exact96967RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩]

theorem exact96967RawTermsValid :
    exact96967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50680⟩⟩) exact96967RawTerms (.finite 10) 96966 .exactZero (none)

def event96968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50681⟩⟩) 0 ⟨50680⟩ 96967

def event96969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50681⟩⟩) 1 ⟨24590⟩ 96964

def event96970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50681⟩⟩) (.product (.predecessor 0 96968 .coefficient) (.predecessor 1 96969 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96971 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50681⟩⟩, .operator (⟨96967, 0⟩, ⟨96964, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩)

def exact96972RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩]

theorem exact96972RawTermsValid :
    exact96972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96972 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50681⟩⟩) exact96972RawTerms (.finite 100) 96970 .exactZero (none)

def event96973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50682⟩⟩) 0 ⟨50681⟩ 96972

def event96974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50682⟩⟩) (.identity (.predecessor 0 96973 .coefficient))

def event96975 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50682⟩⟩) (.finite 100)

def event96976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52038⟩⟩) 0 ⟨50682⟩ 96975

def event96977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52038⟩⟩) (.authority (.programFamilyFact))

def event96978 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52038⟩⟩) (.finite 3720)

def event96979 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event96980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52039⟩⟩) 0 ⟨7177⟩ 96979

def event96981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52039⟩⟩) 1 ⟨52038⟩ 96978

def event96982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52039⟩⟩) (.authority (.operator))

def exact96983RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (1)⟩]

theorem exact96983RawTermsValid :
    exact96983RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96983 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52039⟩⟩) exact96983RawTerms .large 96982 .exactZero (none)

def event96984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52574⟩⟩) 0 ⟨52039⟩ 96983

def event96985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52574⟩⟩) (.authority (.operator))

def exact96986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (1)⟩]

theorem exact96986RawTermsValid :
    exact96986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52574⟩⟩) exact96986RawTerms (.finite 8192) 96985 .exactZero (none)

def event96987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event96988 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event96989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52306⟩⟩) 0 ⟨50682⟩ 96975

def event96990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52306⟩⟩) 1 ⟨136⟩ 96988

def event96991 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52306⟩⟩) (.sum [.predecessor 0 96989 .coefficient, .predecessor 1 96990 .coefficient])

def event96992 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52306⟩⟩) (.finite 100)

def event96993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52307⟩⟩) 0 ⟨52306⟩ 96992

def event96994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52307⟩⟩) (.identity (.predecessor 0 96993 .coefficient))

def exact96995RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩]

theorem exact96995RawTermsValid :
    exact96995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96995 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52307⟩⟩) exact96995RawTerms (.finite 100) 96994 .exactZero (none)

def event96996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact96997RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact96997RawTermsValid :
    exact96997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96997 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact96997RawTerms .large 96996 .exactZero (none)

def event96998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52308⟩⟩) 0 ⟨6908⟩ 96997

def event96999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52308⟩⟩) 1 ⟨52307⟩ 96995

def event97000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52308⟩⟩) (.product (.predecessor 0 96998 .coefficient) (.predecessor 1 96999 .coefficient) (⟨false, false, none, none, none⟩))

def event97001 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52308⟩⟩, .operator (⟨96997, 0⟩, ⟨96995, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact97002RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97002RawTermsValid :
    exact97002RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97002 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52308⟩⟩) exact97002RawTerms .large 97000 .exactZero (none)

def event97003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event97004 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event97005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 96979

def event97006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact97007RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact97007RawTermsValid :
    exact97007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97007 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact97007RawTerms .large 97006 .exactZero (none)

def event97008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7308⟩⟩) 0 ⟨7178⟩ 97007

def event97009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7308⟩⟩) (.identity (.predecessor 0 97008 .coefficient))

def exact97010RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact97010RawTermsValid :
    exact97010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7308⟩⟩) exact97010RawTerms .large 97009 .exactZero (none)

def event97011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9580⟩⟩) 0 ⟨7308⟩ 97010

def event97012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9580⟩⟩) (.authority (.operator))

def exact97013RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact97013RawTermsValid :
    exact97013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97013 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9580⟩⟩) exact97013RawTerms (.finite 8192) 97012 .exactZero (none)

def event97014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9581⟩⟩) 0 ⟨9580⟩ 97013

def event97015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9581⟩⟩) 1 ⟨2370⟩ 97004

def event97016 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9581⟩⟩) (.scale (.predecessor 0 97014 .coefficient) (.value (.predecessor 1 97015 .coefficient)))

def exact97017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact97017RawTermsValid :
    exact97017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9581⟩⟩) exact97017RawTerms (.finite 8192) 97016 .exactZero (none)

def event97018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7288⟩⟩) 0 ⟨7178⟩ 97007

def event97019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7288⟩⟩) (.identity (.predecessor 0 97018 .coefficient))

def exact97020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩]

theorem exact97020RawTermsValid :
    exact97020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7288⟩⟩) exact97020RawTerms .large 97019 .exactZero (none)

def event97021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9582⟩⟩) 0 ⟨7288⟩ 97020

def event97022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9582⟩⟩) 1 ⟨9581⟩ 97017

def event97023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9582⟩⟩) (.product (.predecessor 0 97021 .coefficient) (.predecessor 1 97022 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf6048 : Array AnnotatedEvent := #[
  { event := event96768
    frameStart := 96668 },
  { event := event96769
    frameStart := 96668 },
  { event := event96770
    frameStart := 96668 },
  { event := event96771
    frameStart := 96668 },
  { event := event96772
    frameStart := 0 },
  { event := event96773
    frameStart := 0 },
  { event := event96774
    frameStart := 0 },
  { event := event96775
    frameStart := 0 },
  { event := event96776
    frameStart := 0 },
  { event := event96777
    frameStart := 0 },
  { event := event96778
    frameStart := 0 },
  { event := event96779
    frameStart := 0 },
  { event := event96780
    frameStart := 0 },
  { event := event96781
    frameStart := 0 },
  { event := event96782
    frameStart := 0 },
  { event := event96783
    frameStart := 0 }
]

def eventLeaf6049 : Array AnnotatedEvent := #[
  { event := event96784
    frameStart := 0 },
  { event := event96785
    frameStart := 0 },
  { event := event96786
    frameStart := 0 },
  { event := event96787
    frameStart := 0 },
  { event := event96788
    frameStart := 0 },
  { event := event96789
    frameStart := 0 },
  { event := event96790
    frameStart := 0 },
  { event := event96791
    frameStart := 0 },
  { event := event96792
    frameStart := 0 },
  { event := event96793
    frameStart := 0 },
  { event := event96794
    frameStart := 0 },
  { event := event96795
    frameStart := 0 },
  { event := event96796
    frameStart := 0 },
  { event := event96797
    frameStart := 0 },
  { event := event96798
    frameStart := 0 },
  { event := event96799
    frameStart := 0 }
]

def eventLeaf6050 : Array AnnotatedEvent := #[
  { event := event96800
    frameStart := 0 },
  { event := event96801
    frameStart := 0 },
  { event := event96802
    frameStart := 0 },
  { event := event96803
    frameStart := 0 },
  { event := event96804
    frameStart := 0 },
  { event := event96805
    frameStart := 0 },
  { event := event96806
    frameStart := 0 },
  { event := event96807
    frameStart := 0 },
  { event := event96808
    frameStart := 0 },
  { event := event96809
    frameStart := 0 },
  { event := event96810
    frameStart := 0 },
  { event := event96811
    frameStart := 0 },
  { event := event96812
    frameStart := 0 },
  { event := event96813
    frameStart := 0 },
  { event := event96814
    frameStart := 0 },
  { event := event96815
    frameStart := 0 }
]

def eventLeaf6051 : Array AnnotatedEvent := #[
  { event := event96816
    frameStart := 0 },
  { event := event96817
    frameStart := 0 },
  { event := event96818
    frameStart := 0 },
  { event := event96819
    frameStart := 0 },
  { event := event96820
    frameStart := 0 },
  { event := event96821
    frameStart := 0 },
  { event := event96822
    frameStart := 0 },
  { event := event96823
    frameStart := 0 },
  { event := event96824
    frameStart := 0 },
  { event := event96825
    frameStart := 0 },
  { event := event96826
    frameStart := 0 },
  { event := event96827
    frameStart := 0 },
  { event := event96828
    frameStart := 0 },
  { event := event96829
    frameStart := 0 },
  { event := event96830
    frameStart := 0 },
  { event := event96831
    frameStart := 0 }
]

def eventLeaf6052 : Array AnnotatedEvent := #[
  { event := event96832
    frameStart := 0 },
  { event := event96833
    frameStart := 0 },
  { event := event96834
    frameStart := 0 },
  { event := event96835
    frameStart := 0 },
  { event := event96836
    frameStart := 0 },
  { event := event96837
    frameStart := 0 },
  { event := event96838
    frameStart := 0 },
  { event := event96839
    frameStart := 0 },
  { event := event96840
    frameStart := 0 },
  { event := event96841
    frameStart := 0 },
  { event := event96842
    frameStart := 0 },
  { event := event96843
    frameStart := 0 },
  { event := event96844
    frameStart := 0 },
  { event := event96845
    frameStart := 0 },
  { event := event96846
    frameStart := 0 },
  { event := event96847
    frameStart := 0 }
]

def eventLeaf6053 : Array AnnotatedEvent := #[
  { event := event96848
    frameStart := 0 },
  { event := event96849
    frameStart := 0 },
  { event := event96850
    frameStart := 0 },
  { event := event96851
    frameStart := 0 },
  { event := event96852
    frameStart := 0 },
  { event := event96853
    frameStart := 0 },
  { event := event96854
    frameStart := 0 },
  { event := event96855
    frameStart := 0 },
  { event := event96856
    frameStart := 0 },
  { event := event96857
    frameStart := 0 },
  { event := event96858
    frameStart := 0 },
  { event := event96859
    frameStart := 0 },
  { event := event96860
    frameStart := 0 },
  { event := event96861
    frameStart := 0 },
  { event := event96862
    frameStart := 0 },
  { event := event96863
    frameStart := 0 }
]

def eventLeaf6054 : Array AnnotatedEvent := #[
  { event := event96864
    frameStart := 0 },
  { event := event96865
    frameStart := 0 },
  { event := event96866
    frameStart := 0 },
  { event := event96867
    frameStart := 0 },
  { event := event96868
    frameStart := 0 },
  { event := event96869
    frameStart := 0 },
  { event := event96870
    frameStart := 0 },
  { event := event96871
    frameStart := 0 },
  { event := event96872
    frameStart := 0 },
  { event := event96873
    frameStart := 0 },
  { event := event96874
    frameStart := 0 },
  { event := event96875
    frameStart := 0 },
  { event := event96876
    frameStart := 0 },
  { event := event96877
    frameStart := 0 },
  { event := event96878
    frameStart := 0 },
  { event := event96879
    frameStart := 0 }
]

def eventLeaf6055 : Array AnnotatedEvent := #[
  { event := event96880
    frameStart := 0 },
  { event := event96881
    frameStart := 0 },
  { event := event96882
    frameStart := 0 },
  { event := event96883
    frameStart := 0 },
  { event := event96884
    frameStart := 0 },
  { event := event96885
    frameStart := 0 },
  { event := event96886
    frameStart := 0 },
  { event := event96887
    frameStart := 0 },
  { event := event96888
    frameStart := 0 },
  { event := event96889
    frameStart := 0 },
  { event := event96890
    frameStart := 0 },
  { event := event96891
    frameStart := 0 },
  { event := event96892
    frameStart := 0 },
  { event := event96893
    frameStart := 96893 },
  { event := event96894
    frameStart := 96893 },
  { event := event96895
    frameStart := 96893 }
]

def eventLeaf6056 : Array AnnotatedEvent := #[
  { event := event96896
    frameStart := 96893 },
  { event := event96897
    frameStart := 96893 },
  { event := event96898
    frameStart := 96893 },
  { event := event96899
    frameStart := 96893 },
  { event := event96900
    frameStart := 96893 },
  { event := event96901
    frameStart := 96893 },
  { event := event96902
    frameStart := 96893 },
  { event := event96903
    frameStart := 96893 },
  { event := event96904
    frameStart := 96893 },
  { event := event96905
    frameStart := 96893 },
  { event := event96906
    frameStart := 96893 },
  { event := event96907
    frameStart := 96893 },
  { event := event96908
    frameStart := 96893 },
  { event := event96909
    frameStart := 96893 },
  { event := event96910
    frameStart := 96893 },
  { event := event96911
    frameStart := 96893 }
]

def eventLeaf6057 : Array AnnotatedEvent := #[
  { event := event96912
    frameStart := 96893 },
  { event := event96913
    frameStart := 96893 },
  { event := event96914
    frameStart := 96893 },
  { event := event96915
    frameStart := 96893 },
  { event := event96916
    frameStart := 96893 },
  { event := event96917
    frameStart := 96893 },
  { event := event96918
    frameStart := 96893 },
  { event := event96919
    frameStart := 96893 },
  { event := event96920
    frameStart := 96893 },
  { event := event96921
    frameStart := 96893 },
  { event := event96922
    frameStart := 96893 },
  { event := event96923
    frameStart := 96893 },
  { event := event96924
    frameStart := 96893 },
  { event := event96925
    frameStart := 96893 },
  { event := event96926
    frameStart := 96893 },
  { event := event96927
    frameStart := 96893 }
]

def eventLeaf6058 : Array AnnotatedEvent := #[
  { event := event96928
    frameStart := 96893 },
  { event := event96929
    frameStart := 96893 },
  { event := event96930
    frameStart := 96893 },
  { event := event96931
    frameStart := 96893 },
  { event := event96932
    frameStart := 96893 },
  { event := event96933
    frameStart := 96893 },
  { event := event96934
    frameStart := 96893 },
  { event := event96935
    frameStart := 96893 },
  { event := event96936
    frameStart := 96893 },
  { event := event96937
    frameStart := 96893 },
  { event := event96938
    frameStart := 96893 },
  { event := event96939
    frameStart := 96893 },
  { event := event96940
    frameStart := 96893 },
  { event := event96941
    frameStart := 96941 },
  { event := event96942
    frameStart := 96941 },
  { event := event96943
    frameStart := 96941 }
]

def eventLeaf6059 : Array AnnotatedEvent := #[
  { event := event96944
    frameStart := 96941 },
  { event := event96945
    frameStart := 96941 },
  { event := event96946
    frameStart := 96941 },
  { event := event96947
    frameStart := 96941 },
  { event := event96948
    frameStart := 96941 },
  { event := event96949
    frameStart := 96941 },
  { event := event96950
    frameStart := 96941 },
  { event := event96951
    frameStart := 96941 },
  { event := event96952
    frameStart := 96941 },
  { event := event96953
    frameStart := 96941 },
  { event := event96954
    frameStart := 96941 },
  { event := event96955
    frameStart := 96941 },
  { event := event96956
    frameStart := 96941 },
  { event := event96957
    frameStart := 96941 },
  { event := event96958
    frameStart := 96941 },
  { event := event96959
    frameStart := 96941 }
]

def eventLeaf6060 : Array AnnotatedEvent := #[
  { event := event96960
    frameStart := 96941 },
  { event := event96961
    frameStart := 96941 },
  { event := event96962
    frameStart := 96941 },
  { event := event96963
    frameStart := 96941 },
  { event := event96964
    frameStart := 96941 },
  { event := event96965
    frameStart := 96941 },
  { event := event96966
    frameStart := 96941 },
  { event := event96967
    frameStart := 96941 },
  { event := event96968
    frameStart := 96941 },
  { event := event96969
    frameStart := 96941 },
  { event := event96970
    frameStart := 96941 },
  { event := event96971
    frameStart := 96941 },
  { event := event96972
    frameStart := 96941 },
  { event := event96973
    frameStart := 96941 },
  { event := event96974
    frameStart := 96941 },
  { event := event96975
    frameStart := 96941 }
]

def eventLeaf6061 : Array AnnotatedEvent := #[
  { event := event96976
    frameStart := 96941 },
  { event := event96977
    frameStart := 96941 },
  { event := event96978
    frameStart := 96941 },
  { event := event96979
    frameStart := 96941 },
  { event := event96980
    frameStart := 96941 },
  { event := event96981
    frameStart := 96941 },
  { event := event96982
    frameStart := 96941 },
  { event := event96983
    frameStart := 96941 },
  { event := event96984
    frameStart := 96941 },
  { event := event96985
    frameStart := 96941 },
  { event := event96986
    frameStart := 96941 },
  { event := event96987
    frameStart := 96941 },
  { event := event96988
    frameStart := 96941 },
  { event := event96989
    frameStart := 96941 },
  { event := event96990
    frameStart := 96941 },
  { event := event96991
    frameStart := 96941 }
]

def eventLeaf6062 : Array AnnotatedEvent := #[
  { event := event96992
    frameStart := 96941 },
  { event := event96993
    frameStart := 96941 },
  { event := event96994
    frameStart := 96941 },
  { event := event96995
    frameStart := 96941 },
  { event := event96996
    frameStart := 96941 },
  { event := event96997
    frameStart := 96941 },
  { event := event96998
    frameStart := 96941 },
  { event := event96999
    frameStart := 96941 },
  { event := event97000
    frameStart := 96941 },
  { event := event97001
    frameStart := 96941 },
  { event := event97002
    frameStart := 96941 },
  { event := event97003
    frameStart := 96941 },
  { event := event97004
    frameStart := 96941 },
  { event := event97005
    frameStart := 96941 },
  { event := event97006
    frameStart := 96941 },
  { event := event97007
    frameStart := 96941 }
]

def eventLeaf6063 : Array AnnotatedEvent := #[
  { event := event97008
    frameStart := 96941 },
  { event := event97009
    frameStart := 96941 },
  { event := event97010
    frameStart := 96941 },
  { event := event97011
    frameStart := 96941 },
  { event := event97012
    frameStart := 96941 },
  { event := event97013
    frameStart := 96941 },
  { event := event97014
    frameStart := 96941 },
  { event := event97015
    frameStart := 96941 },
  { event := event97016
    frameStart := 96941 },
  { event := event97017
    frameStart := 96941 },
  { event := event97018
    frameStart := 96941 },
  { event := event97019
    frameStart := 96941 },
  { event := event97020
    frameStart := 96941 },
  { event := event97021
    frameStart := 96941 },
  { event := event97022
    frameStart := 96941 },
  { event := event97023
    frameStart := 96941 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events378
