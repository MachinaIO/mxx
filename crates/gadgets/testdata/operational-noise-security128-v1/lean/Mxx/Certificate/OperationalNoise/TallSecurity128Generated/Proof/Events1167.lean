import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1167

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact298752RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], []⟩, (1)⟩]

theorem exact298752RawTermsValid :
    exact298752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298752 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68888⟩⟩) exact298752RawTerms (.finite 784) 298751 .exactZero (none)

def event298753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact298754RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact298754RawTermsValid :
    exact298754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact298754RawTerms .large 298753 .exactZero (none)

def event298755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68889⟩⟩) 0 ⟨6908⟩ 298754

def event298756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68889⟩⟩) 1 ⟨68888⟩ 298752

def event298757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68889⟩⟩) (.product (.predecessor 0 298755 .coefficient) (.predecessor 1 298756 .coefficient) (⟨false, false, none, none, none⟩))

def event298758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68889⟩⟩, .operator (⟨298754, 0⟩, ⟨298752, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact298759RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact298759RawTermsValid :
    exact298759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68889⟩⟩) exact298759RawTerms .large 298757 .exactZero (none)

def event298760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event298761 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event298762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 298736

def event298763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact298764RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact298764RawTermsValid :
    exact298764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact298764RawTerms .large 298763 .exactZero (none)

def event298765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7276⟩⟩) 0 ⟨7178⟩ 298764

def event298766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7276⟩⟩) (.identity (.predecessor 0 298765 .coefficient))

def exact298767RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact298767RawTermsValid :
    exact298767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7276⟩⟩) exact298767RawTerms .large 298766 .exactZero (none)

def event298768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9541⟩⟩) 0 ⟨7276⟩ 298767

def event298769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9541⟩⟩) (.authority (.operator))

def exact298770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact298770RawTermsValid :
    exact298770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9541⟩⟩) exact298770RawTerms (.finite 8192) 298769 .exactZero (none)

def event298771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9542⟩⟩) 0 ⟨9541⟩ 298770

def event298772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9542⟩⟩) 1 ⟨2370⟩ 298761

def event298773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9542⟩⟩) (.scale (.predecessor 0 298771 .coefficient) (.value (.predecessor 1 298772 .coefficient)))

def exact298774RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact298774RawTermsValid :
    exact298774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9542⟩⟩) exact298774RawTerms (.finite 8192) 298773 .exactZero (none)

def event298775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7294⟩⟩) 0 ⟨7178⟩ 298764

def event298776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7294⟩⟩) (.identity (.predecessor 0 298775 .coefficient))

def exact298777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩]

theorem exact298777RawTermsValid :
    exact298777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7294⟩⟩) exact298777RawTerms .large 298776 .exactZero (none)

def event298778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9543⟩⟩) 0 ⟨7294⟩ 298777

def event298779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9543⟩⟩) 1 ⟨9542⟩ 298774

def event298780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9543⟩⟩) (.product (.predecessor 0 298778 .coefficient) (.predecessor 1 298779 .coefficient) (⟨false, false, none, none, none⟩))

def event298781 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9543⟩⟩, .operator (⟨298777, 0⟩, ⟨298774, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩)

def exact298782RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact298782RawTermsValid :
    exact298782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9543⟩⟩) exact298782RawTerms .large 298780 .exactZero (none)

def event298783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68890⟩⟩) 0 ⟨9543⟩ 298782

def event298784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68890⟩⟩) 1 ⟨68889⟩ 298759

def event298785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68890⟩⟩) (.sum [.predecessor 0 298783 .coefficient, .predecessor 1 298784 .coefficient])

def exact298786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298786RawTermsValid :
    exact298786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68890⟩⟩) exact298786RawTerms .large 298785 .exactZero (none)

def event298787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69133⟩⟩) 0 ⟨68890⟩ 298786

def event298788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69133⟩⟩) 1 ⟨69130⟩ 298743

def event298789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69133⟩⟩) (.product (.predecessor 0 298787 .coefficient) (.predecessor 1 298788 .coefficient) (⟨false, false, none, none, none⟩))

def event298790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69133⟩⟩, .operator (⟨298786, 0⟩, ⟨298743, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (1)⟩)

def event298791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69133⟩⟩, .operator (⟨298786, 1⟩, ⟨298743, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (-1)⟩)

def event298792 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69133⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69130⟩⟩) ⟨68470⟩ 298740)

def event298793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69133⟩⟩, .relation 298792 0, ⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩, (-1)⟩)

def exact298794RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩, (-1)⟩]

theorem exact298794RawTermsValid :
    exact298794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298794 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69133⟩⟩) exact298794RawTerms .large 298789 .exactZero (none)

def event298795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65708⟩⟩) 0 ⟨65177⟩ 298732

def event298796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65708⟩⟩) (.authority (.programFamilyFact))

def exact298797RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], []⟩, (1)⟩]

theorem exact298797RawTermsValid :
    exact298797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298797 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65708⟩⟩) exact298797RawTerms (.finite 28) 298796 .exactZero (none)

def event298798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65710⟩⟩) 0 ⟨6908⟩ 298754

def event298799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65710⟩⟩) 1 ⟨65708⟩ 298797

def event298800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65710⟩⟩) (.product (.predecessor 0 298798 .coefficient) (.predecessor 1 298799 .coefficient) (⟨false, true, none, none, some 1⟩))

def event298801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65710⟩⟩, .operator (⟨298754, 0⟩, ⟨298797, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact298802RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact298802RawTermsValid :
    exact298802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298802 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65710⟩⟩) exact298802RawTerms .large 298800 .exactZero (none)

def event298803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7188⟩⟩) 0 ⟨7177⟩ 298736

def event298804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7188⟩⟩) (.authority (.operator))

def exact298805RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩]

theorem exact298805RawTermsValid :
    exact298805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298805 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7188⟩⟩) exact298805RawTerms .large 298804 .exactZero (none)

def event298806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65711⟩⟩) 0 ⟨7188⟩ 298805

def event298807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65711⟩⟩) 1 ⟨65710⟩ 298802

def event298808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65711⟩⟩) (.sum [.predecessor 0 298806 .coefficient, .predecessor 1 298807 .coefficient])

def exact298809RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298809RawTermsValid :
    exact298809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298809 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65711⟩⟩) exact298809RawTerms .large 298808 .exactZero (none)

def event298810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69134⟩⟩) 0 ⟨65711⟩ 298809

def event298811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69134⟩⟩) 1 ⟨69133⟩ 298794

def event298812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69134⟩⟩) (.sum [.predecessor 0 298810 .coefficient, .predecessor 1 298811 .coefficient])

def exact298813RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298813RawTermsValid :
    exact298813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69134⟩⟩) exact298813RawTerms .large 298812 .exactZero (none)

def event298814 : Event := .preFoldPolynomial 298813 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact298815RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event298815 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨69134⟩⟩) 298814 exact298815RawTerms .large 298812 .exactZero (none)

def event298816 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨65177⟩⟩) ⟨⟨67⟩, ⟨46⟩, ⟨135⟩⟩ ⟨298674, 298816⟩

def event298817 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨67673⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩) (1) 0 2 (.universal 298816 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67670⟩⟩]⟩) (none) 298815)

def event298818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67673⟩⟩, .relation 298817 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩)

def event298819 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67673⟩⟩, .relation 298817 1, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (-1)⟩)

def event298820 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67673⟩⟩, .relation 298817 2, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩, (1)⟩)

def event298821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67673⟩⟩, .relation 298817 3, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact298822RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298822RawTermsValid :
    exact298822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298822 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67673⟩⟩) exact298822RawTerms .large 298670 (.finite 202072841853861888) (some (298672))

def event298823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69132⟩⟩) 0 ⟨67673⟩ 298822

def event298824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69132⟩⟩) 1 ⟨69131⟩ 298660

def event298825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69132⟩⟩) (.sum [.predecessor 0 298823 .coefficient, .predecessor 1 298824 .coefficient])

def event298826 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69132⟩⟩, .operator (⟨298822, 2⟩, ⟨298660, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], [⟨.program ⟨257⟩, ⟨68470⟩⟩]⟩, (-1)⟩)

def event298827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69132⟩⟩, .operator (⟨298822, 1⟩, ⟨298660, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69130⟩⟩]⟩, (1)⟩)

def event298828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69132⟩⟩) (.sum [.result 298822 .summary, .result 298660 .summary])

def exact298829RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298829RawTermsValid :
    exact298829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69132⟩⟩) exact298829RawTerms .large 298825 (.finite 2998054127048462696448) (some (298828))

def event298830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69389⟩⟩) 0 ⟨69132⟩ 298829

def event298831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69389⟩⟩) 1 ⟨69387⟩ 298576

def event298832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69389⟩⟩) (.product (.predecessor 0 298830 .coefficient) (.predecessor 1 298831 .coefficient) (⟨false, false, none, none, none⟩))

def event298833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69389⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩) [⟨.result 298576 .coefficient, false, none⟩])

def event298834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69389⟩⟩) (.product (.result 298829 .summary) (.transfer 298833) (⟨false, false, none, none, none⟩))

def event298835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69389⟩⟩, .operator (⟨298829, 0⟩, ⟨298576, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (1)⟩)

def event298836 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69389⟩⟩, .operator (⟨298829, 1⟩, ⟨298576, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (-1)⟩)

def event298837 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69389⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69387⟩⟩) ⟨68592⟩ 298573)

def event298838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69389⟩⟩, .relation 298837 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (-1)⟩)

def exact298839RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (-1)⟩]

theorem exact298839RawTermsValid :
    exact298839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69389⟩⟩) exact298839RawTerms .large 298832 (.finite 32191361068277440720800338411520) (some (298834))

def event298840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67877⟩⟩) 0 ⟨65709⟩ 14491

def event298841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67877⟩⟩) (.authority (.relationPreimageSource ⟨76⟩))

def exact298842RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩, (1)⟩]

theorem exact298842RawTermsValid :
    exact298842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67877⟩⟩) exact298842RawTerms (.finite 5647228698) 298841 .exactZero (none)

def event298843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67879⟩⟩) 0 ⟨67877⟩ 298842

def event298844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67879⟩⟩) 1 ⟨2370⟩ 4

def event298845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67879⟩⟩) (.scale (.predecessor 0 298843 .coefficient) (.value (.predecessor 1 298844 .coefficient)))

def exact298846RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩, (1)⟩]

theorem exact298846RawTermsValid :
    exact298846RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298846 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67879⟩⟩) exact298846RawTerms (.finite 5647228698) 298845 .exactZero (none)

def event298847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67880⟩⟩) 0 ⟨2380⟩ 295195

def event298848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67880⟩⟩) 1 ⟨67879⟩ 298846

def event298849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67880⟩⟩) (.product (.predecessor 0 298847 .coefficient) (.predecessor 1 298848 .coefficient) (⟨false, false, none, none, none⟩))

def event298850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67880⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩) [⟨.result 298842 .coefficient, false, none⟩])

def event298851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67880⟩⟩) (.product (.result 295195 .summary) (.transfer 298850) (⟨false, false, none, none, none⟩))

def event298852 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67880⟩⟩, .operator (⟨295195, 0⟩, ⟨298846, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩, (1)⟩)

def event298853 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨67878⟩⟩)

def event298854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event298855 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event298856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event298857 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event298858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 298857

def event298859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 298855

def event298860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 298858 .coefficient) (.value (.predecessor 1 298859 .coefficient)))

def event298861 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event298862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25610⟩⟩) 0 ⟨392⟩ 298861

def event298863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25610⟩⟩) (.authority (.programFamilyFact))

def exact298864RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩], []⟩, (1)⟩]

theorem exact298864RawTermsValid :
    exact298864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298864 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25610⟩⟩) exact298864RawTerms (.finite 28) 298863 .exactZero (none)

def event298865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65175⟩⟩) 0 ⟨392⟩ 298861

def event298866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65175⟩⟩) (.authority (.programFamilyFact))

def exact298867RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65175⟩⟩], []⟩, (1)⟩]

theorem exact298867RawTermsValid :
    exact298867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65175⟩⟩) exact298867RawTerms (.finite 28) 298866 .exactZero (none)

def event298868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65176⟩⟩) 0 ⟨65175⟩ 298867

def event298869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65176⟩⟩) 1 ⟨25610⟩ 298864

def event298870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65176⟩⟩) (.product (.predecessor 0 298868 .coefficient) (.predecessor 1 298869 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event298871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65176⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], []⟩) [⟨.result 298867 .coefficient, true, some 1⟩, ⟨.result 298864 .coefficient, true, some 1⟩])

def event298872 : Event := .survivorFold (1) 298871

def exact298873RawTerms : List Term := []

theorem exact298873RawTermsValid :
    exact298873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65176⟩⟩) exact298873RawTerms (.finite 784) 298870 (.finite 784) (some (298871))

def event298874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65177⟩⟩) 0 ⟨65176⟩ 298873

def event298875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65177⟩⟩) (.identity (.predecessor 0 298874 .coefficient))

def event298876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65177⟩⟩) (.finite 784)

def event298877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65708⟩⟩) 0 ⟨65177⟩ 298876

def event298878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65708⟩⟩) (.authority (.programFamilyFact))

def exact298879RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], []⟩, (1)⟩]

theorem exact298879RawTermsValid :
    exact298879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65708⟩⟩) exact298879RawTerms (.finite 28) 298878 .exactZero (none)

def event298880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65709⟩⟩) 0 ⟨65708⟩ 298879

def event298881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65709⟩⟩) (.identity (.predecessor 0 298880 .coefficient))

def event298882 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65709⟩⟩) (.finite 28)

def event298883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67877⟩⟩) 0 ⟨65709⟩ 298882

def event298884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67877⟩⟩) (.authority (.relationPreimageSource ⟨76⟩))

def exact298885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩, (1)⟩]

theorem exact298885RawTermsValid :
    exact298885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67877⟩⟩) exact298885RawTerms (.finite 5647228698) 298884 .exactZero (none)

def event298886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact298887RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact298887RawTermsValid :
    exact298887RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298887 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact298887RawTerms .large 298886 .exactZero (none)

def event298888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67878⟩⟩) 0 ⟨35⟩ 298887

def event298889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67878⟩⟩) 1 ⟨67877⟩ 298885

def event298890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67878⟩⟩) (.product (.predecessor 0 298888 .coefficient) (.predecessor 1 298889 .coefficient) (⟨false, false, none, none, none⟩))

def event298891 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67878⟩⟩, .operator (⟨298887, 0⟩, ⟨298885, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩, (1)⟩)

def exact298892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩, (1)⟩]

theorem exact298892RawTermsValid :
    exact298892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67878⟩⟩) exact298892RawTerms .large 298890 .exactZero (none)

def event298893 : Event := .preFoldPolynomial 298892 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩, (1)⟩] .exactZero none

def exact298894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩, (1)⟩]

def event298894 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨67878⟩⟩) 298893 exact298894RawTerms .large 298890 .exactZero (none)

def event298895 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨69400⟩⟩)

def event298896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event298897 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event298898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event298899 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event298900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 298899

def event298901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 298897

def event298902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 298900 .coefficient) (.value (.predecessor 1 298901 .coefficient)))

def event298903 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event298904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25610⟩⟩) 0 ⟨392⟩ 298903

def event298905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25610⟩⟩) (.authority (.programFamilyFact))

def exact298906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩], []⟩, (1)⟩]

theorem exact298906RawTermsValid :
    exact298906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25610⟩⟩) exact298906RawTerms (.finite 28) 298905 .exactZero (none)

def event298907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65175⟩⟩) 0 ⟨392⟩ 298903

def event298908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65175⟩⟩) (.authority (.programFamilyFact))

def exact298909RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65175⟩⟩], []⟩, (1)⟩]

theorem exact298909RawTermsValid :
    exact298909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65175⟩⟩) exact298909RawTerms (.finite 28) 298908 .exactZero (none)

def event298910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65176⟩⟩) 0 ⟨65175⟩ 298909

def event298911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65176⟩⟩) 1 ⟨25610⟩ 298906

def event298912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65176⟩⟩) (.product (.predecessor 0 298910 .coefficient) (.predecessor 1 298911 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event298913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65176⟩⟩, .operator (⟨298909, 0⟩, ⟨298906, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], []⟩, (1)⟩)

def exact298914RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25610⟩⟩, ⟨.program ⟨257⟩, ⟨65175⟩⟩], []⟩, (1)⟩]

theorem exact298914RawTermsValid :
    exact298914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298914 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65176⟩⟩) exact298914RawTerms (.finite 784) 298912 .exactZero (none)

def event298915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65177⟩⟩) 0 ⟨65176⟩ 298914

def event298916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65177⟩⟩) (.identity (.predecessor 0 298915 .coefficient))

def event298917 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65177⟩⟩) (.finite 784)

def event298918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65708⟩⟩) 0 ⟨65177⟩ 298917

def event298919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65708⟩⟩) (.authority (.programFamilyFact))

def exact298920RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], []⟩, (1)⟩]

theorem exact298920RawTermsValid :
    exact298920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65708⟩⟩) exact298920RawTerms (.finite 28) 298919 .exactZero (none)

def event298921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65709⟩⟩) 0 ⟨65708⟩ 298920

def event298922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65709⟩⟩) (.identity (.predecessor 0 298921 .coefficient))

def event298923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65709⟩⟩) (.finite 28)

def event298924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68590⟩⟩) 0 ⟨65709⟩ 298923

def event298925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68590⟩⟩) (.authority (.programFamilyFact))

def event298926 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68590⟩⟩) (.finite 3720)

def event298927 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event298928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68592⟩⟩) 0 ⟨7177⟩ 298927

def event298929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68592⟩⟩) 1 ⟨68590⟩ 298926

def event298930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68592⟩⟩) (.authority (.operator))

def exact298931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (1)⟩]

theorem exact298931RawTermsValid :
    exact298931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68592⟩⟩) exact298931RawTerms .large 298930 .exactZero (none)

def event298932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69387⟩⟩) 0 ⟨68592⟩ 298931

def event298933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69387⟩⟩) (.authority (.operator))

def exact298934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (1)⟩]

theorem exact298934RawTermsValid :
    exact298934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69387⟩⟩) exact298934RawTerms (.finite 8192) 298933 .exactZero (none)

def event298935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event298936 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event298937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68967⟩⟩) 0 ⟨65709⟩ 298923

def event298938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68967⟩⟩) 1 ⟨136⟩ 298936

def event298939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68967⟩⟩) (.sum [.predecessor 0 298937 .coefficient, .predecessor 1 298938 .coefficient])

def event298940 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68967⟩⟩) (.finite 28)

def event298941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68968⟩⟩) 0 ⟨68967⟩ 298940

def event298942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68968⟩⟩) (.identity (.predecessor 0 298941 .coefficient))

def exact298943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], []⟩, (1)⟩]

theorem exact298943RawTermsValid :
    exact298943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68968⟩⟩) exact298943RawTerms (.finite 28) 298942 .exactZero (none)

def event298944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact298945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact298945RawTermsValid :
    exact298945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact298945RawTerms .large 298944 .exactZero (none)

def event298946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68969⟩⟩) 0 ⟨6908⟩ 298945

def event298947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68969⟩⟩) 1 ⟨68968⟩ 298943

def event298948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68969⟩⟩) (.product (.predecessor 0 298946 .coefficient) (.predecessor 1 298947 .coefficient) (⟨false, false, none, none, none⟩))

def event298949 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68969⟩⟩, .operator (⟨298945, 0⟩, ⟨298943, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact298950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact298950RawTermsValid :
    exact298950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68969⟩⟩) exact298950RawTerms .large 298948 .exactZero (none)

def event298951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7188⟩⟩) 0 ⟨7177⟩ 298927

def event298952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7188⟩⟩) (.authority (.operator))

def exact298953RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩]

theorem exact298953RawTermsValid :
    exact298953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7188⟩⟩) exact298953RawTerms .large 298952 .exactZero (none)

def event298954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68970⟩⟩) 0 ⟨7188⟩ 298953

def event298955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68970⟩⟩) 1 ⟨68969⟩ 298950

def event298956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68970⟩⟩) (.sum [.predecessor 0 298954 .coefficient, .predecessor 1 298955 .coefficient])

def exact298957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298957RawTermsValid :
    exact298957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298957 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68970⟩⟩) exact298957RawTerms .large 298956 .exactZero (none)

def event298958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69388⟩⟩) 0 ⟨68970⟩ 298957

def event298959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69388⟩⟩) 1 ⟨69387⟩ 298934

def event298960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69388⟩⟩) (.product (.predecessor 0 298958 .coefficient) (.predecessor 1 298959 .coefficient) (⟨false, false, none, none, none⟩))

def event298961 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69388⟩⟩, .operator (⟨298957, 0⟩, ⟨298934, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (1)⟩)

def event298962 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69388⟩⟩, .operator (⟨298957, 1⟩, ⟨298934, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (-1)⟩)

def event298963 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69388⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69387⟩⟩) ⟨68592⟩ 298931)

def event298964 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69388⟩⟩, .relation 298963 0, ⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (-1)⟩)

def exact298965RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (-1)⟩]

theorem exact298965RawTermsValid :
    exact298965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69388⟩⟩) exact298965RawTerms .large 298960 .exactZero (none)

def event298966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65901⟩⟩) 0 ⟨65709⟩ 298923

def event298967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65901⟩⟩) (.authority (.programFamilyFact))

def exact298968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], []⟩, (1)⟩]

theorem exact298968RawTermsValid :
    exact298968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65901⟩⟩) exact298968RawTerms (.finite 62) 298967 .exactZero (none)

def event298969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65912⟩⟩) 0 ⟨6908⟩ 298945

def event298970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65912⟩⟩) 1 ⟨65901⟩ 298968

def event298971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65912⟩⟩) (.product (.predecessor 0 298969 .coefficient) (.predecessor 1 298970 .coefficient) (⟨false, true, none, none, some 1⟩))

def event298972 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65912⟩⟩, .operator (⟨298945, 0⟩, ⟨298968, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact298973RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact298973RawTermsValid :
    exact298973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298973 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65912⟩⟩) exact298973RawTerms .large 298971 .exactZero (none)

def event298974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 298927

def event298975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact298976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact298976RawTermsValid :
    exact298976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact298976RawTerms .large 298975 .exactZero (none)

def event298977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65913⟩⟩) 0 ⟨7216⟩ 298976

def event298978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65913⟩⟩) 1 ⟨65912⟩ 298973

def event298979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65913⟩⟩) (.sum [.predecessor 0 298977 .coefficient, .predecessor 1 298978 .coefficient])

def exact298980RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298980RawTermsValid :
    exact298980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65913⟩⟩) exact298980RawTerms .large 298979 .exactZero (none)

def event298981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69400⟩⟩) 0 ⟨65913⟩ 298980

def event298982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69400⟩⟩) 1 ⟨69388⟩ 298965

def event298983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69400⟩⟩) (.sum [.predecessor 0 298981 .coefficient, .predecessor 1 298982 .coefficient])

def exact298984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298984RawTermsValid :
    exact298984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69400⟩⟩) exact298984RawTerms .large 298983 .exactZero (none)

def event298985 : Event := .preFoldPolynomial 298984 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact298986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event298986 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨69400⟩⟩) 298985 exact298986RawTerms .large 298983 .exactZero (none)

def event298987 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨65709⟩⟩) ⟨⟨95⟩, ⟨76⟩, ⟨135⟩⟩ ⟨298853, 298987⟩

def event298988 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨67880⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩) (1) 0 2 (.universal 298987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩) (none) 298986)

def event298989 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67880⟩⟩, .relation 298988 1, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩)

def event298990 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67880⟩⟩, .relation 298988 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (-1)⟩)

def event298991 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67880⟩⟩, .relation 298988 2, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (1)⟩)

def event298992 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67880⟩⟩, .relation 298988 3, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact298993RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact298993RawTermsValid :
    exact298993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event298993 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67880⟩⟩) exact298993RawTerms .large 298849 (.finite 202072841853861888) (some (298851))

def event298994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69390⟩⟩) 0 ⟨67880⟩ 298993

def event298995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69390⟩⟩) 1 ⟨69389⟩ 298839

def event298996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69390⟩⟩) (.sum [.predecessor 0 298994 .coefficient, .predecessor 1 298995 .coefficient])

def event298997 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69390⟩⟩, .operator (⟨298993, 0⟩, ⟨298839, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩, (1)⟩)

def event298998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69390⟩⟩, .operator (⟨298993, 2⟩, ⟨298839, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65708⟩⟩], [⟨.program ⟨257⟩, ⟨68592⟩⟩]⟩, (-1)⟩)

def event298999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69390⟩⟩) (.sum [.result 298993 .summary, .result 298839 .summary])

def exact299000RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact299000RawTermsValid :
    exact299000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event299000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69390⟩⟩) exact299000RawTerms .large 298996 (.finite 32191361068277642793642192273408) (some (298999))

def event299001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63989⟩⟩) 0 ⟨62729⟩ 14514

def event299002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63989⟩⟩) (.authority (.programFamilyFact))

def event299003 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨63989⟩⟩) (.finite 3720)

def event299004 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63991⟩⟩) 0 ⟨7177⟩ 15500

def event299005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63991⟩⟩) 1 ⟨63989⟩ 299003

def event299006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63991⟩⟩) (.authority (.operator))

def exact299007RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63991⟩⟩]⟩, (1)⟩]

theorem exact299007RawTermsValid :
    exact299007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event299007 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63991⟩⟩) exact299007RawTerms .large 299006 .exactZero (none)

def eventLeaf18672 : Array AnnotatedEvent := #[
  { event := event298752
    frameStart := 298710 },
  { event := event298753
    frameStart := 298710 },
  { event := event298754
    frameStart := 298710 },
  { event := event298755
    frameStart := 298710 },
  { event := event298756
    frameStart := 298710 },
  { event := event298757
    frameStart := 298710 },
  { event := event298758
    frameStart := 298710 },
  { event := event298759
    frameStart := 298710 },
  { event := event298760
    frameStart := 298710 },
  { event := event298761
    frameStart := 298710 },
  { event := event298762
    frameStart := 298710 },
  { event := event298763
    frameStart := 298710 },
  { event := event298764
    frameStart := 298710 },
  { event := event298765
    frameStart := 298710 },
  { event := event298766
    frameStart := 298710 },
  { event := event298767
    frameStart := 298710 }
]

def eventLeaf18673 : Array AnnotatedEvent := #[
  { event := event298768
    frameStart := 298710 },
  { event := event298769
    frameStart := 298710 },
  { event := event298770
    frameStart := 298710 },
  { event := event298771
    frameStart := 298710 },
  { event := event298772
    frameStart := 298710 },
  { event := event298773
    frameStart := 298710 },
  { event := event298774
    frameStart := 298710 },
  { event := event298775
    frameStart := 298710 },
  { event := event298776
    frameStart := 298710 },
  { event := event298777
    frameStart := 298710 },
  { event := event298778
    frameStart := 298710 },
  { event := event298779
    frameStart := 298710 },
  { event := event298780
    frameStart := 298710 },
  { event := event298781
    frameStart := 298710 },
  { event := event298782
    frameStart := 298710 },
  { event := event298783
    frameStart := 298710 }
]

def eventLeaf18674 : Array AnnotatedEvent := #[
  { event := event298784
    frameStart := 298710 },
  { event := event298785
    frameStart := 298710 },
  { event := event298786
    frameStart := 298710 },
  { event := event298787
    frameStart := 298710 },
  { event := event298788
    frameStart := 298710 },
  { event := event298789
    frameStart := 298710 },
  { event := event298790
    frameStart := 298710 },
  { event := event298791
    frameStart := 298710 },
  { event := event298792
    frameStart := 298710 },
  { event := event298793
    frameStart := 298710 },
  { event := event298794
    frameStart := 298710 },
  { event := event298795
    frameStart := 298710 },
  { event := event298796
    frameStart := 298710 },
  { event := event298797
    frameStart := 298710 },
  { event := event298798
    frameStart := 298710 },
  { event := event298799
    frameStart := 298710 }
]

def eventLeaf18675 : Array AnnotatedEvent := #[
  { event := event298800
    frameStart := 298710 },
  { event := event298801
    frameStart := 298710 },
  { event := event298802
    frameStart := 298710 },
  { event := event298803
    frameStart := 298710 },
  { event := event298804
    frameStart := 298710 },
  { event := event298805
    frameStart := 298710 },
  { event := event298806
    frameStart := 298710 },
  { event := event298807
    frameStart := 298710 },
  { event := event298808
    frameStart := 298710 },
  { event := event298809
    frameStart := 298710 },
  { event := event298810
    frameStart := 298710 },
  { event := event298811
    frameStart := 298710 },
  { event := event298812
    frameStart := 298710 },
  { event := event298813
    frameStart := 298710 },
  { event := event298814
    frameStart := 298710 },
  { event := event298815
    frameStart := 298710 }
]

def eventLeaf18676 : Array AnnotatedEvent := #[
  { event := event298816
    frameStart := 0 },
  { event := event298817
    frameStart := 0 },
  { event := event298818
    frameStart := 0 },
  { event := event298819
    frameStart := 0 },
  { event := event298820
    frameStart := 0 },
  { event := event298821
    frameStart := 0 },
  { event := event298822
    frameStart := 0 },
  { event := event298823
    frameStart := 0 },
  { event := event298824
    frameStart := 0 },
  { event := event298825
    frameStart := 0 },
  { event := event298826
    frameStart := 0 },
  { event := event298827
    frameStart := 0 },
  { event := event298828
    frameStart := 0 },
  { event := event298829
    frameStart := 0 },
  { event := event298830
    frameStart := 0 },
  { event := event298831
    frameStart := 0 }
]

def eventLeaf18677 : Array AnnotatedEvent := #[
  { event := event298832
    frameStart := 0 },
  { event := event298833
    frameStart := 0 },
  { event := event298834
    frameStart := 0 },
  { event := event298835
    frameStart := 0 },
  { event := event298836
    frameStart := 0 },
  { event := event298837
    frameStart := 0 },
  { event := event298838
    frameStart := 0 },
  { event := event298839
    frameStart := 0 },
  { event := event298840
    frameStart := 0 },
  { event := event298841
    frameStart := 0 },
  { event := event298842
    frameStart := 0 },
  { event := event298843
    frameStart := 0 },
  { event := event298844
    frameStart := 0 },
  { event := event298845
    frameStart := 0 },
  { event := event298846
    frameStart := 0 },
  { event := event298847
    frameStart := 0 }
]

def eventLeaf18678 : Array AnnotatedEvent := #[
  { event := event298848
    frameStart := 0 },
  { event := event298849
    frameStart := 0 },
  { event := event298850
    frameStart := 0 },
  { event := event298851
    frameStart := 0 },
  { event := event298852
    frameStart := 0 },
  { event := event298853
    frameStart := 298853 },
  { event := event298854
    frameStart := 298853 },
  { event := event298855
    frameStart := 298853 },
  { event := event298856
    frameStart := 298853 },
  { event := event298857
    frameStart := 298853 },
  { event := event298858
    frameStart := 298853 },
  { event := event298859
    frameStart := 298853 },
  { event := event298860
    frameStart := 298853 },
  { event := event298861
    frameStart := 298853 },
  { event := event298862
    frameStart := 298853 },
  { event := event298863
    frameStart := 298853 }
]

def eventLeaf18679 : Array AnnotatedEvent := #[
  { event := event298864
    frameStart := 298853 },
  { event := event298865
    frameStart := 298853 },
  { event := event298866
    frameStart := 298853 },
  { event := event298867
    frameStart := 298853 },
  { event := event298868
    frameStart := 298853 },
  { event := event298869
    frameStart := 298853 },
  { event := event298870
    frameStart := 298853 },
  { event := event298871
    frameStart := 298853 },
  { event := event298872
    frameStart := 298853 },
  { event := event298873
    frameStart := 298853 },
  { event := event298874
    frameStart := 298853 },
  { event := event298875
    frameStart := 298853 },
  { event := event298876
    frameStart := 298853 },
  { event := event298877
    frameStart := 298853 },
  { event := event298878
    frameStart := 298853 },
  { event := event298879
    frameStart := 298853 }
]

def eventLeaf18680 : Array AnnotatedEvent := #[
  { event := event298880
    frameStart := 298853 },
  { event := event298881
    frameStart := 298853 },
  { event := event298882
    frameStart := 298853 },
  { event := event298883
    frameStart := 298853 },
  { event := event298884
    frameStart := 298853 },
  { event := event298885
    frameStart := 298853 },
  { event := event298886
    frameStart := 298853 },
  { event := event298887
    frameStart := 298853 },
  { event := event298888
    frameStart := 298853 },
  { event := event298889
    frameStart := 298853 },
  { event := event298890
    frameStart := 298853 },
  { event := event298891
    frameStart := 298853 },
  { event := event298892
    frameStart := 298853 },
  { event := event298893
    frameStart := 298853 },
  { event := event298894
    frameStart := 298853 },
  { event := event298895
    frameStart := 298895 }
]

def eventLeaf18681 : Array AnnotatedEvent := #[
  { event := event298896
    frameStart := 298895 },
  { event := event298897
    frameStart := 298895 },
  { event := event298898
    frameStart := 298895 },
  { event := event298899
    frameStart := 298895 },
  { event := event298900
    frameStart := 298895 },
  { event := event298901
    frameStart := 298895 },
  { event := event298902
    frameStart := 298895 },
  { event := event298903
    frameStart := 298895 },
  { event := event298904
    frameStart := 298895 },
  { event := event298905
    frameStart := 298895 },
  { event := event298906
    frameStart := 298895 },
  { event := event298907
    frameStart := 298895 },
  { event := event298908
    frameStart := 298895 },
  { event := event298909
    frameStart := 298895 },
  { event := event298910
    frameStart := 298895 },
  { event := event298911
    frameStart := 298895 }
]

def eventLeaf18682 : Array AnnotatedEvent := #[
  { event := event298912
    frameStart := 298895 },
  { event := event298913
    frameStart := 298895 },
  { event := event298914
    frameStart := 298895 },
  { event := event298915
    frameStart := 298895 },
  { event := event298916
    frameStart := 298895 },
  { event := event298917
    frameStart := 298895 },
  { event := event298918
    frameStart := 298895 },
  { event := event298919
    frameStart := 298895 },
  { event := event298920
    frameStart := 298895 },
  { event := event298921
    frameStart := 298895 },
  { event := event298922
    frameStart := 298895 },
  { event := event298923
    frameStart := 298895 },
  { event := event298924
    frameStart := 298895 },
  { event := event298925
    frameStart := 298895 },
  { event := event298926
    frameStart := 298895 },
  { event := event298927
    frameStart := 298895 }
]

def eventLeaf18683 : Array AnnotatedEvent := #[
  { event := event298928
    frameStart := 298895 },
  { event := event298929
    frameStart := 298895 },
  { event := event298930
    frameStart := 298895 },
  { event := event298931
    frameStart := 298895 },
  { event := event298932
    frameStart := 298895 },
  { event := event298933
    frameStart := 298895 },
  { event := event298934
    frameStart := 298895 },
  { event := event298935
    frameStart := 298895 },
  { event := event298936
    frameStart := 298895 },
  { event := event298937
    frameStart := 298895 },
  { event := event298938
    frameStart := 298895 },
  { event := event298939
    frameStart := 298895 },
  { event := event298940
    frameStart := 298895 },
  { event := event298941
    frameStart := 298895 },
  { event := event298942
    frameStart := 298895 },
  { event := event298943
    frameStart := 298895 }
]

def eventLeaf18684 : Array AnnotatedEvent := #[
  { event := event298944
    frameStart := 298895 },
  { event := event298945
    frameStart := 298895 },
  { event := event298946
    frameStart := 298895 },
  { event := event298947
    frameStart := 298895 },
  { event := event298948
    frameStart := 298895 },
  { event := event298949
    frameStart := 298895 },
  { event := event298950
    frameStart := 298895 },
  { event := event298951
    frameStart := 298895 },
  { event := event298952
    frameStart := 298895 },
  { event := event298953
    frameStart := 298895 },
  { event := event298954
    frameStart := 298895 },
  { event := event298955
    frameStart := 298895 },
  { event := event298956
    frameStart := 298895 },
  { event := event298957
    frameStart := 298895 },
  { event := event298958
    frameStart := 298895 },
  { event := event298959
    frameStart := 298895 }
]

def eventLeaf18685 : Array AnnotatedEvent := #[
  { event := event298960
    frameStart := 298895 },
  { event := event298961
    frameStart := 298895 },
  { event := event298962
    frameStart := 298895 },
  { event := event298963
    frameStart := 298895 },
  { event := event298964
    frameStart := 298895 },
  { event := event298965
    frameStart := 298895 },
  { event := event298966
    frameStart := 298895 },
  { event := event298967
    frameStart := 298895 },
  { event := event298968
    frameStart := 298895 },
  { event := event298969
    frameStart := 298895 },
  { event := event298970
    frameStart := 298895 },
  { event := event298971
    frameStart := 298895 },
  { event := event298972
    frameStart := 298895 },
  { event := event298973
    frameStart := 298895 },
  { event := event298974
    frameStart := 298895 },
  { event := event298975
    frameStart := 298895 }
]

def eventLeaf18686 : Array AnnotatedEvent := #[
  { event := event298976
    frameStart := 298895 },
  { event := event298977
    frameStart := 298895 },
  { event := event298978
    frameStart := 298895 },
  { event := event298979
    frameStart := 298895 },
  { event := event298980
    frameStart := 298895 },
  { event := event298981
    frameStart := 298895 },
  { event := event298982
    frameStart := 298895 },
  { event := event298983
    frameStart := 298895 },
  { event := event298984
    frameStart := 298895 },
  { event := event298985
    frameStart := 298895 },
  { event := event298986
    frameStart := 298895 },
  { event := event298987
    frameStart := 0 },
  { event := event298988
    frameStart := 0 },
  { event := event298989
    frameStart := 0 },
  { event := event298990
    frameStart := 0 },
  { event := event298991
    frameStart := 0 }
]

def eventLeaf18687 : Array AnnotatedEvent := #[
  { event := event298992
    frameStart := 0 },
  { event := event298993
    frameStart := 0 },
  { event := event298994
    frameStart := 0 },
  { event := event298995
    frameStart := 0 },
  { event := event298996
    frameStart := 0 },
  { event := event298997
    frameStart := 0 },
  { event := event298998
    frameStart := 0 },
  { event := event298999
    frameStart := 0 },
  { event := event299000
    frameStart := 0 },
  { event := event299001
    frameStart := 0 },
  { event := event299002
    frameStart := 0 },
  { event := event299003
    frameStart := 0 },
  { event := event299004
    frameStart := 0 },
  { event := event299005
    frameStart := 0 },
  { event := event299006
    frameStart := 0 },
  { event := event299007
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1167
