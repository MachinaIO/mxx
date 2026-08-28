import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events343

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact87808RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact87808RawTermsValid :
    exact87808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87808 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10774⟩⟩) exact87808RawTerms .large 87806 .exactZero (none)

def event87809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 87785

def event87810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact87811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact87811RawTermsValid :
    exact87811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact87811RawTerms .large 87810 .exactZero (none)

def event87812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6773⟩⟩) 0 ⟨6757⟩ 87811

def event87813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6773⟩⟩) (.identity (.predecessor 0 87812 .coefficient))

def exact87814RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩]

theorem exact87814RawTermsValid :
    exact87814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87814 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6773⟩⟩) exact87814RawTerms .large 87813 .exactZero (none)

def event87815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7834⟩⟩) 0 ⟨6773⟩ 87814

def event87816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7834⟩⟩) (.authority (.operator))

def exact87817RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩]

theorem exact87817RawTermsValid :
    exact87817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7834⟩⟩) exact87817RawTerms (.finite 8192) 87816 .exactZero (none)

def event87818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7835⟩⟩) 0 ⟨7834⟩ 87817

def event87819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7835⟩⟩) 1 ⟨2348⟩ 87751

def event87820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7835⟩⟩) (.scale (.predecessor 0 87818 .coefficient) (.value (.predecessor 1 87819 .coefficient)))

def exact87821RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩]

theorem exact87821RawTermsValid :
    exact87821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87821 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7835⟩⟩) exact87821RawTerms (.finite 8192) 87820 .exactZero (none)

def event87822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6782⟩⟩) 0 ⟨6757⟩ 87811

def event87823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6782⟩⟩) (.identity (.predecessor 0 87822 .coefficient))

def exact87824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩]⟩, (1)⟩]

theorem exact87824RawTermsValid :
    exact87824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87824 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6782⟩⟩) exact87824RawTerms .large 87823 .exactZero (none)

def event87825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7836⟩⟩) 0 ⟨6782⟩ 87824

def event87826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7836⟩⟩) 1 ⟨7835⟩ 87821

def event87827 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7836⟩⟩) (.product (.predecessor 0 87825 .coefficient) (.predecessor 1 87826 .coefficient) (⟨false, false, none, none, none⟩))

def event87828 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7836⟩⟩, .operator (⟨87824, 0⟩, ⟨87821, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩)

def exact87829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩]

theorem exact87829RawTermsValid :
    exact87829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7836⟩⟩) exact87829RawTerms .large 87827 .exactZero (none)

def event87830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10775⟩⟩) 0 ⟨7836⟩ 87829

def event87831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10775⟩⟩) 1 ⟨10774⟩ 87808

def event87832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10775⟩⟩) (.sum [.predecessor 0 87830 .coefficient, .predecessor 1 87831 .coefficient])

def exact87833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87833RawTermsValid :
    exact87833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87833 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10775⟩⟩) exact87833RawTerms .large 87832 .exactZero (none)

def event87834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24991⟩⟩) 0 ⟨10775⟩ 87833

def event87835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24991⟩⟩) 1 ⟨24988⟩ 87792

def event87836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24991⟩⟩) (.product (.predecessor 0 87834 .coefficient) (.predecessor 1 87835 .coefficient) (⟨false, false, none, none, none⟩))

def event87837 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24991⟩⟩, .operator (⟨87833, 0⟩, ⟨87792, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (1)⟩)

def event87838 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24991⟩⟩, .operator (⟨87833, 1⟩, ⟨87792, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (-1)⟩)

def event87839 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨24991⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24988⟩⟩) ⟨22996⟩ 87789)

def event87840 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24991⟩⟩, .relation 87839 0, ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩, (-1)⟩)

def exact87841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩, (-1)⟩]

theorem exact87841RawTermsValid :
    exact87841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87841 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24991⟩⟩) exact87841RawTerms .large 87836 .exactZero (none)

def event87842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14953⟩⟩) 0 ⟨10678⟩ 87781

def event87843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14953⟩⟩) (.authority (.programFamilyFact))

def exact87844RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], []⟩, (1)⟩]

theorem exact87844RawTermsValid :
    exact87844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14953⟩⟩) exact87844RawTerms (.finite 3) 87843 .exactZero (none)

def event87845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14955⟩⟩) 0 ⟨6544⟩ 87803

def event87846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14955⟩⟩) 1 ⟨14953⟩ 87844

def event87847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14955⟩⟩) (.product (.predecessor 0 87845 .coefficient) (.predecessor 1 87846 .coefficient) (⟨false, true, none, none, some 1⟩))

def event87848 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14955⟩⟩, .operator (⟨87803, 0⟩, ⟨87844, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact87849RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact87849RawTermsValid :
    exact87849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14955⟩⟩) exact87849RawTerms .large 87847 .exactZero (none)

def event87850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6691⟩⟩) 0 ⟨6689⟩ 87785

def event87851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6691⟩⟩) (.authority (.operator))

def exact87852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩]

theorem exact87852RawTermsValid :
    exact87852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6691⟩⟩) exact87852RawTerms .large 87851 .exactZero (none)

def event87853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14956⟩⟩) 0 ⟨6691⟩ 87852

def event87854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14956⟩⟩) 1 ⟨14955⟩ 87849

def event87855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14956⟩⟩) (.sum [.predecessor 0 87853 .coefficient, .predecessor 1 87854 .coefficient])

def exact87856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87856RawTermsValid :
    exact87856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87856 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14956⟩⟩) exact87856RawTerms .large 87855 .exactZero (none)

def event87857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24992⟩⟩) 0 ⟨14956⟩ 87856

def event87858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24992⟩⟩) 1 ⟨24991⟩ 87841

def event87859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24992⟩⟩) (.sum [.predecessor 0 87857 .coefficient, .predecessor 1 87858 .coefficient])

def exact87860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87860RawTermsValid :
    exact87860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87860 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24992⟩⟩) exact87860RawTerms .large 87859 .exactZero (none)

def event87861 : Event := .preFoldPolynomial 87860 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact87862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event87862 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨24992⟩⟩) 87861 exact87862RawTerms .large 87859 .exactZero (none)

def event87863 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨10678⟩⟩) ⟨⟨104⟩, ⟨8⟩, ⟨109⟩⟩ ⟨87699, 87863⟩

def event87864 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19099⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩) (1) 0 2 (.universal 87863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩) (none) 87862)

def event87865 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19099⟩⟩, .relation 87864 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩)

def event87866 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19099⟩⟩, .relation 87864 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (-1)⟩)

def event87867 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19099⟩⟩, .relation 87864 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩, (1)⟩)

def event87868 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19099⟩⟩, .relation 87864 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact87869RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87869RawTermsValid :
    exact87869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87869 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19099⟩⟩) exact87869RawTerms .large 87695 (.finite 1811303510016) (some (87697))

def event87870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24990⟩⟩) 0 ⟨19099⟩ 87869

def event87871 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24990⟩⟩) 1 ⟨24989⟩ 87685

def event87872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24990⟩⟩) (.sum [.predecessor 0 87870 .coefficient, .predecessor 1 87871 .coefficient])

def event87873 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24990⟩⟩, .operator (⟨87869, 2⟩, ⟨87685, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], [⟨.program ⟨214⟩, ⟨22996⟩⟩]⟩, (-1)⟩)

def event87874 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24990⟩⟩, .operator (⟨87869, 1⟩, ⟨87685, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩, (1)⟩)

def event87875 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24990⟩⟩) (.sum [.result 87869 .summary, .result 87685 .summary])

def exact87876RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87876RawTermsValid :
    exact87876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87876 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24990⟩⟩) exact87876RawTerms .large 87872 (.finite 352014917316608) (some (87875))

def event87877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26566⟩⟩) 0 ⟨24990⟩ 87876

def event87878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26566⟩⟩) 1 ⟨26564⟩ 87601

def event87879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26566⟩⟩) (.product (.predecessor 0 87877 .coefficient) (.predecessor 1 87878 .coefficient) (⟨false, false, none, none, none⟩))

def event87880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26566⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩) [⟨.result 87601 .coefficient, false, none⟩])

def event87881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26566⟩⟩) (.product (.result 87876 .summary) (.transfer 87880) (⟨false, false, none, none, none⟩))

def event87882 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26566⟩⟩, .operator (⟨87876, 0⟩, ⟨87601, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (1)⟩)

def event87883 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26566⟩⟩, .operator (⟨87876, 1⟩, ⟨87601, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (-1)⟩)

def event87884 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26566⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26564⟩⟩) ⟨23784⟩ 87598)

def event87885 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26566⟩⟩, .relation 87884 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (-1)⟩)

def exact87886RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (-1)⟩]

theorem exact87886RawTermsValid :
    exact87886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87886 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26566⟩⟩) exact87886RawTerms .large 87879 (.finite 1291900378790628425728) (some (87881))

def event87887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20536⟩⟩) 0 ⟨14954⟩ 4213

def event87888 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20536⟩⟩) (.authority (.relationPreimageSource ⟨30⟩))

def exact87889RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩, (1)⟩]

theorem exact87889RawTermsValid :
    exact87889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87889 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20536⟩⟩) exact87889RawTerms (.finite 136065468) 87888 .exactZero (none)

def event87890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20538⟩⟩) 0 ⟨20536⟩ 87889

def event87891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20538⟩⟩) 1 ⟨2348⟩ 4

def event87892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20538⟩⟩) (.scale (.predecessor 0 87890 .coefficient) (.value (.predecessor 1 87891 .coefficient)))

def exact87893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩, (1)⟩]

theorem exact87893RawTermsValid :
    exact87893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87893 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20538⟩⟩) exact87893RawTerms (.finite 136065468) 87892 .exactZero (none)

def event87894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20539⟩⟩) 0 ⟨5541⟩ 80012

def event87895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20539⟩⟩) 1 ⟨20538⟩ 87893

def event87896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20539⟩⟩) (.product (.predecessor 0 87894 .coefficient) (.predecessor 1 87895 .coefficient) (⟨false, false, none, none, none⟩))

def event87897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20539⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩) [⟨.result 87889 .coefficient, false, none⟩])

def event87898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20539⟩⟩) (.product (.result 80012 .summary) (.transfer 87897) (⟨false, false, none, none, none⟩))

def event87899 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20539⟩⟩, .operator (⟨80012, 0⟩, ⟨87893, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩, (1)⟩)

def event87900 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20537⟩⟩)

def event87901 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event87902 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event87903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event87904 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event87905 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event87906 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event87907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event87908 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event87909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 87908

def event87910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 87906

def event87911 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 87909 .coefficient) (.value (.predecessor 1 87910 .coefficient)))

def event87912 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event87913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 87912

def event87914 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 87904

def event87915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 87913 .coefficient, .predecessor 1 87914 .coefficient])

def event87916 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event87917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 87916

def event87918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 87902

def event87919 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 87918 .coefficient))

def event87920 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event87921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10676⟩⟩) 0 ⟨5536⟩ 87920

def event87922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10676⟩⟩) (.authority (.programFamilyFact))

def exact87923RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10676⟩⟩], []⟩, (1)⟩]

theorem exact87923RawTermsValid :
    exact87923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87923 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10676⟩⟩) exact87923RawTerms (.finite 3) 87922 .exactZero (none)

def event87924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9505⟩⟩) 0 ⟨5536⟩ 87920

def event87925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9505⟩⟩) (.authority (.programFamilyFact))

def exact87926RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩], []⟩, (1)⟩]

theorem exact87926RawTermsValid :
    exact87926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87926 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9505⟩⟩) exact87926RawTerms (.finite 3) 87925 .exactZero (none)

def event87927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10677⟩⟩) 0 ⟨9505⟩ 87926

def event87928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10677⟩⟩) 1 ⟨10676⟩ 87923

def event87929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10677⟩⟩) (.product (.predecessor 0 87927 .coefficient) (.predecessor 1 87928 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10677⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], []⟩) [⟨.result 87926 .coefficient, true, some 1⟩, ⟨.result 87923 .coefficient, true, some 1⟩])

def event87931 : Event := .survivorFold (1) 87930

def exact87932RawTerms : List Term := []

theorem exact87932RawTermsValid :
    exact87932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87932 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10677⟩⟩) exact87932RawTerms (.finite 9) 87929 (.finite 9) (some (87930))

def event87933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10678⟩⟩) 0 ⟨10677⟩ 87932

def event87934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10678⟩⟩) (.identity (.predecessor 0 87933 .coefficient))

def event87935 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10678⟩⟩) (.finite 9)

def event87936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14953⟩⟩) 0 ⟨10678⟩ 87935

def event87937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14953⟩⟩) (.authority (.programFamilyFact))

def exact87938RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], []⟩, (1)⟩]

theorem exact87938RawTermsValid :
    exact87938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87938 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14953⟩⟩) exact87938RawTerms (.finite 3) 87937 .exactZero (none)

def event87939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14954⟩⟩) 0 ⟨14953⟩ 87938

def event87940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14954⟩⟩) (.identity (.predecessor 0 87939 .coefficient))

def event87941 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14954⟩⟩) (.finite 3)

def event87942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20536⟩⟩) 0 ⟨14954⟩ 87941

def event87943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20536⟩⟩) (.authority (.relationPreimageSource ⟨30⟩))

def exact87944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩, (1)⟩]

theorem exact87944RawTermsValid :
    exact87944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20536⟩⟩) exact87944RawTerms (.finite 136065468) 87943 .exactZero (none)

def event87945 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact87946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact87946RawTermsValid :
    exact87946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87946 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact87946RawTerms .large 87945 .exactZero (none)

def event87947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20537⟩⟩) 0 ⟨6⟩ 87946

def event87948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20537⟩⟩) 1 ⟨20536⟩ 87944

def event87949 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20537⟩⟩) (.product (.predecessor 0 87947 .coefficient) (.predecessor 1 87948 .coefficient) (⟨false, false, none, none, none⟩))

def event87950 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20537⟩⟩, .operator (⟨87946, 0⟩, ⟨87944, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩, (1)⟩)

def exact87951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩, (1)⟩]

theorem exact87951RawTermsValid :
    exact87951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87951 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20537⟩⟩) exact87951RawTerms .large 87949 .exactZero (none)

def event87952 : Event := .preFoldPolynomial 87951 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩, (1)⟩] .exactZero none

def exact87953RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩, (1)⟩]

def event87953 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20537⟩⟩) 87952 exact87953RawTerms .large 87949 .exactZero (none)

def event87954 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26569⟩⟩)

def event87955 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event87956 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event87957 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event87958 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event87959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event87960 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event87961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event87962 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event87963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 87962

def event87964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 87960

def event87965 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 87963 .coefficient) (.value (.predecessor 1 87964 .coefficient)))

def event87966 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event87967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 87966

def event87968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 87958

def event87969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 87967 .coefficient, .predecessor 1 87968 .coefficient])

def event87970 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event87971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 87970

def event87972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 87956

def event87973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 87972 .coefficient))

def event87974 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event87975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10676⟩⟩) 0 ⟨5536⟩ 87974

def event87976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10676⟩⟩) (.authority (.programFamilyFact))

def exact87977RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10676⟩⟩], []⟩, (1)⟩]

theorem exact87977RawTermsValid :
    exact87977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87977 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10676⟩⟩) exact87977RawTerms (.finite 3) 87976 .exactZero (none)

def event87978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9505⟩⟩) 0 ⟨5536⟩ 87974

def event87979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9505⟩⟩) (.authority (.programFamilyFact))

def exact87980RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩], []⟩, (1)⟩]

theorem exact87980RawTermsValid :
    exact87980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87980 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9505⟩⟩) exact87980RawTerms (.finite 3) 87979 .exactZero (none)

def event87981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10677⟩⟩) 0 ⟨9505⟩ 87980

def event87982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10677⟩⟩) 1 ⟨10676⟩ 87977

def event87983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10677⟩⟩) (.product (.predecessor 0 87981 .coefficient) (.predecessor 1 87982 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87984 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10677⟩⟩, .operator (⟨87980, 0⟩, ⟨87977, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], []⟩, (1)⟩)

def exact87985RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], []⟩, (1)⟩]

theorem exact87985RawTermsValid :
    exact87985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87985 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10677⟩⟩) exact87985RawTerms (.finite 9) 87983 .exactZero (none)

def event87986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10678⟩⟩) 0 ⟨10677⟩ 87985

def event87987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10678⟩⟩) (.identity (.predecessor 0 87986 .coefficient))

def event87988 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10678⟩⟩) (.finite 9)

def event87989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14953⟩⟩) 0 ⟨10678⟩ 87988

def event87990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14953⟩⟩) (.authority (.programFamilyFact))

def exact87991RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], []⟩, (1)⟩]

theorem exact87991RawTermsValid :
    exact87991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14953⟩⟩) exact87991RawTerms (.finite 3) 87990 .exactZero (none)

def event87992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14954⟩⟩) 0 ⟨14953⟩ 87991

def event87993 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14954⟩⟩) (.identity (.predecessor 0 87992 .coefficient))

def event87994 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14954⟩⟩) (.finite 3)

def event87995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23782⟩⟩) 0 ⟨14954⟩ 87994

def event87996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23782⟩⟩) (.authority (.programFamilyFact))

def event87997 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23782⟩⟩) (.finite 3720)

def event87998 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event87999 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23784⟩⟩) 0 ⟨6689⟩ 87998

def event88000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23784⟩⟩) 1 ⟨23782⟩ 87997

def event88001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23784⟩⟩) (.authority (.operator))

def exact88002RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (1)⟩]

theorem exact88002RawTermsValid :
    exact88002RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88002 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23784⟩⟩) exact88002RawTerms .large 88001 .exactZero (none)

def event88003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26564⟩⟩) 0 ⟨23784⟩ 88002

def event88004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26564⟩⟩) (.authority (.operator))

def exact88005RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (1)⟩]

theorem exact88005RawTermsValid :
    exact88005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88005 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26564⟩⟩) exact88005RawTerms (.finite 8192) 88004 .exactZero (none)

def event88006 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event88007 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event88008 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14993⟩⟩) 0 ⟨14954⟩ 87994

def event88009 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14993⟩⟩) 1 ⟨110⟩ 88007

def event88010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14993⟩⟩) (.sum [.predecessor 0 88008 .coefficient, .predecessor 1 88009 .coefficient])

def event88011 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14993⟩⟩) (.finite 3)

def event88012 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14994⟩⟩) 0 ⟨14993⟩ 88011

def event88013 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14994⟩⟩) (.identity (.predecessor 0 88012 .coefficient))

def exact88014RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], []⟩, (1)⟩]

theorem exact88014RawTermsValid :
    exact88014RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88014 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14994⟩⟩) exact88014RawTerms (.finite 3) 88013 .exactZero (none)

def event88015 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact88016RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88016RawTermsValid :
    exact88016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88016 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact88016RawTerms .large 88015 .exactZero (none)

def event88017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14995⟩⟩) 0 ⟨6544⟩ 88016

def event88018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14995⟩⟩) 1 ⟨14994⟩ 88014

def event88019 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14995⟩⟩) (.product (.predecessor 0 88017 .coefficient) (.predecessor 1 88018 .coefficient) (⟨false, false, none, none, none⟩))

def event88020 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14995⟩⟩, .operator (⟨88016, 0⟩, ⟨88014, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact88021RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88021RawTermsValid :
    exact88021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88021 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14995⟩⟩) exact88021RawTerms .large 88019 .exactZero (none)

def event88022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6691⟩⟩) 0 ⟨6689⟩ 87998

def event88023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6691⟩⟩) (.authority (.operator))

def exact88024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩]

theorem exact88024RawTermsValid :
    exact88024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88024 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6691⟩⟩) exact88024RawTerms .large 88023 .exactZero (none)

def event88025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14996⟩⟩) 0 ⟨6691⟩ 88024

def event88026 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14996⟩⟩) 1 ⟨14995⟩ 88021

def event88027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14996⟩⟩) (.sum [.predecessor 0 88025 .coefficient, .predecessor 1 88026 .coefficient])

def exact88028RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88028RawTermsValid :
    exact88028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88028 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14996⟩⟩) exact88028RawTerms .large 88027 .exactZero (none)

def event88029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26565⟩⟩) 0 ⟨14996⟩ 88028

def event88030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26565⟩⟩) 1 ⟨26564⟩ 88005

def event88031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26565⟩⟩) (.product (.predecessor 0 88029 .coefficient) (.predecessor 1 88030 .coefficient) (⟨false, false, none, none, none⟩))

def event88032 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26565⟩⟩, .operator (⟨88028, 0⟩, ⟨88005, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (1)⟩)

def event88033 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26565⟩⟩, .operator (⟨88028, 1⟩, ⟨88005, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (-1)⟩)

def event88034 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26565⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26564⟩⟩) ⟨23784⟩ 88002)

def event88035 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26565⟩⟩, .relation 88034 0, ⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (-1)⟩)

def exact88036RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (-1)⟩]

theorem exact88036RawTermsValid :
    exact88036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88036 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26565⟩⟩) exact88036RawTerms .large 88031 .exactZero (none)

def event88037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15310⟩⟩) 0 ⟨14954⟩ 87994

def event88038 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15310⟩⟩) (.authority (.programFamilyFact))

def exact88039RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15310⟩⟩], []⟩, (1)⟩]

theorem exact88039RawTermsValid :
    exact88039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88039 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15310⟩⟩) exact88039RawTerms (.finite 48) 88038 .exactZero (none)

def event88040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15312⟩⟩) 0 ⟨6544⟩ 88016

def event88041 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15312⟩⟩) 1 ⟨15310⟩ 88039

def event88042 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15312⟩⟩) (.product (.predecessor 0 88040 .coefficient) (.predecessor 1 88041 .coefficient) (⟨false, true, none, none, some 1⟩))

def event88043 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15312⟩⟩, .operator (⟨88016, 0⟩, ⟨88039, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact88044RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88044RawTermsValid :
    exact88044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88044 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15312⟩⟩) exact88044RawTerms .large 88042 .exactZero (none)

def event88045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6711⟩⟩) 0 ⟨6689⟩ 87998

def event88046 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6711⟩⟩) (.authority (.operator))

def exact88047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩]

theorem exact88047RawTermsValid :
    exact88047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88047 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6711⟩⟩) exact88047RawTerms .large 88046 .exactZero (none)

def event88048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15313⟩⟩) 0 ⟨6711⟩ 88047

def event88049 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15313⟩⟩) 1 ⟨15312⟩ 88044

def event88050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15313⟩⟩) (.sum [.predecessor 0 88048 .coefficient, .predecessor 1 88049 .coefficient])

def exact88051RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88051RawTermsValid :
    exact88051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88051 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15313⟩⟩) exact88051RawTerms .large 88050 .exactZero (none)

def event88052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26569⟩⟩) 0 ⟨15313⟩ 88051

def event88053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26569⟩⟩) 1 ⟨26565⟩ 88036

def event88054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26569⟩⟩) (.sum [.predecessor 0 88052 .coefficient, .predecessor 1 88053 .coefficient])

def exact88055RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88055RawTermsValid :
    exact88055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88055 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26569⟩⟩) exact88055RawTerms .large 88054 .exactZero (none)

def event88056 : Event := .preFoldPolynomial 88055 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact88057RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event88057 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26569⟩⟩) 88056 exact88057RawTerms .large 88054 .exactZero (none)

def event88058 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14954⟩⟩) ⟨⟨124⟩, ⟨30⟩, ⟨109⟩⟩ ⟨87900, 88058⟩

def event88059 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20539⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩) (1) 0 2 (.universal 88058 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩) (none) 88057)

def event88060 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20539⟩⟩, .relation 88059 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩)

def event88061 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20539⟩⟩, .relation 88059 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (-1)⟩)

def event88062 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20539⟩⟩, .relation 88059 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (1)⟩)

def event88063 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20539⟩⟩, .relation 88059 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def eventLeaf5488 : Array AnnotatedEvent := #[
  { event := event87808
    frameStart := 87747 },
  { event := event87809
    frameStart := 87747 },
  { event := event87810
    frameStart := 87747 },
  { event := event87811
    frameStart := 87747 },
  { event := event87812
    frameStart := 87747 },
  { event := event87813
    frameStart := 87747 },
  { event := event87814
    frameStart := 87747 },
  { event := event87815
    frameStart := 87747 },
  { event := event87816
    frameStart := 87747 },
  { event := event87817
    frameStart := 87747 },
  { event := event87818
    frameStart := 87747 },
  { event := event87819
    frameStart := 87747 },
  { event := event87820
    frameStart := 87747 },
  { event := event87821
    frameStart := 87747 },
  { event := event87822
    frameStart := 87747 },
  { event := event87823
    frameStart := 87747 }
]

def eventLeaf5489 : Array AnnotatedEvent := #[
  { event := event87824
    frameStart := 87747 },
  { event := event87825
    frameStart := 87747 },
  { event := event87826
    frameStart := 87747 },
  { event := event87827
    frameStart := 87747 },
  { event := event87828
    frameStart := 87747 },
  { event := event87829
    frameStart := 87747 },
  { event := event87830
    frameStart := 87747 },
  { event := event87831
    frameStart := 87747 },
  { event := event87832
    frameStart := 87747 },
  { event := event87833
    frameStart := 87747 },
  { event := event87834
    frameStart := 87747 },
  { event := event87835
    frameStart := 87747 },
  { event := event87836
    frameStart := 87747 },
  { event := event87837
    frameStart := 87747 },
  { event := event87838
    frameStart := 87747 },
  { event := event87839
    frameStart := 87747 }
]

def eventLeaf5490 : Array AnnotatedEvent := #[
  { event := event87840
    frameStart := 87747 },
  { event := event87841
    frameStart := 87747 },
  { event := event87842
    frameStart := 87747 },
  { event := event87843
    frameStart := 87747 },
  { event := event87844
    frameStart := 87747 },
  { event := event87845
    frameStart := 87747 },
  { event := event87846
    frameStart := 87747 },
  { event := event87847
    frameStart := 87747 },
  { event := event87848
    frameStart := 87747 },
  { event := event87849
    frameStart := 87747 },
  { event := event87850
    frameStart := 87747 },
  { event := event87851
    frameStart := 87747 },
  { event := event87852
    frameStart := 87747 },
  { event := event87853
    frameStart := 87747 },
  { event := event87854
    frameStart := 87747 },
  { event := event87855
    frameStart := 87747 }
]

def eventLeaf5491 : Array AnnotatedEvent := #[
  { event := event87856
    frameStart := 87747 },
  { event := event87857
    frameStart := 87747 },
  { event := event87858
    frameStart := 87747 },
  { event := event87859
    frameStart := 87747 },
  { event := event87860
    frameStart := 87747 },
  { event := event87861
    frameStart := 87747 },
  { event := event87862
    frameStart := 87747 },
  { event := event87863
    frameStart := 0 },
  { event := event87864
    frameStart := 0 },
  { event := event87865
    frameStart := 0 },
  { event := event87866
    frameStart := 0 },
  { event := event87867
    frameStart := 0 },
  { event := event87868
    frameStart := 0 },
  { event := event87869
    frameStart := 0 },
  { event := event87870
    frameStart := 0 },
  { event := event87871
    frameStart := 0 }
]

def eventLeaf5492 : Array AnnotatedEvent := #[
  { event := event87872
    frameStart := 0 },
  { event := event87873
    frameStart := 0 },
  { event := event87874
    frameStart := 0 },
  { event := event87875
    frameStart := 0 },
  { event := event87876
    frameStart := 0 },
  { event := event87877
    frameStart := 0 },
  { event := event87878
    frameStart := 0 },
  { event := event87879
    frameStart := 0 },
  { event := event87880
    frameStart := 0 },
  { event := event87881
    frameStart := 0 },
  { event := event87882
    frameStart := 0 },
  { event := event87883
    frameStart := 0 },
  { event := event87884
    frameStart := 0 },
  { event := event87885
    frameStart := 0 },
  { event := event87886
    frameStart := 0 },
  { event := event87887
    frameStart := 0 }
]

def eventLeaf5493 : Array AnnotatedEvent := #[
  { event := event87888
    frameStart := 0 },
  { event := event87889
    frameStart := 0 },
  { event := event87890
    frameStart := 0 },
  { event := event87891
    frameStart := 0 },
  { event := event87892
    frameStart := 0 },
  { event := event87893
    frameStart := 0 },
  { event := event87894
    frameStart := 0 },
  { event := event87895
    frameStart := 0 },
  { event := event87896
    frameStart := 0 },
  { event := event87897
    frameStart := 0 },
  { event := event87898
    frameStart := 0 },
  { event := event87899
    frameStart := 0 },
  { event := event87900
    frameStart := 87900 },
  { event := event87901
    frameStart := 87900 },
  { event := event87902
    frameStart := 87900 },
  { event := event87903
    frameStart := 87900 }
]

def eventLeaf5494 : Array AnnotatedEvent := #[
  { event := event87904
    frameStart := 87900 },
  { event := event87905
    frameStart := 87900 },
  { event := event87906
    frameStart := 87900 },
  { event := event87907
    frameStart := 87900 },
  { event := event87908
    frameStart := 87900 },
  { event := event87909
    frameStart := 87900 },
  { event := event87910
    frameStart := 87900 },
  { event := event87911
    frameStart := 87900 },
  { event := event87912
    frameStart := 87900 },
  { event := event87913
    frameStart := 87900 },
  { event := event87914
    frameStart := 87900 },
  { event := event87915
    frameStart := 87900 },
  { event := event87916
    frameStart := 87900 },
  { event := event87917
    frameStart := 87900 },
  { event := event87918
    frameStart := 87900 },
  { event := event87919
    frameStart := 87900 }
]

def eventLeaf5495 : Array AnnotatedEvent := #[
  { event := event87920
    frameStart := 87900 },
  { event := event87921
    frameStart := 87900 },
  { event := event87922
    frameStart := 87900 },
  { event := event87923
    frameStart := 87900 },
  { event := event87924
    frameStart := 87900 },
  { event := event87925
    frameStart := 87900 },
  { event := event87926
    frameStart := 87900 },
  { event := event87927
    frameStart := 87900 },
  { event := event87928
    frameStart := 87900 },
  { event := event87929
    frameStart := 87900 },
  { event := event87930
    frameStart := 87900 },
  { event := event87931
    frameStart := 87900 },
  { event := event87932
    frameStart := 87900 },
  { event := event87933
    frameStart := 87900 },
  { event := event87934
    frameStart := 87900 },
  { event := event87935
    frameStart := 87900 }
]

def eventLeaf5496 : Array AnnotatedEvent := #[
  { event := event87936
    frameStart := 87900 },
  { event := event87937
    frameStart := 87900 },
  { event := event87938
    frameStart := 87900 },
  { event := event87939
    frameStart := 87900 },
  { event := event87940
    frameStart := 87900 },
  { event := event87941
    frameStart := 87900 },
  { event := event87942
    frameStart := 87900 },
  { event := event87943
    frameStart := 87900 },
  { event := event87944
    frameStart := 87900 },
  { event := event87945
    frameStart := 87900 },
  { event := event87946
    frameStart := 87900 },
  { event := event87947
    frameStart := 87900 },
  { event := event87948
    frameStart := 87900 },
  { event := event87949
    frameStart := 87900 },
  { event := event87950
    frameStart := 87900 },
  { event := event87951
    frameStart := 87900 }
]

def eventLeaf5497 : Array AnnotatedEvent := #[
  { event := event87952
    frameStart := 87900 },
  { event := event87953
    frameStart := 87900 },
  { event := event87954
    frameStart := 87954 },
  { event := event87955
    frameStart := 87954 },
  { event := event87956
    frameStart := 87954 },
  { event := event87957
    frameStart := 87954 },
  { event := event87958
    frameStart := 87954 },
  { event := event87959
    frameStart := 87954 },
  { event := event87960
    frameStart := 87954 },
  { event := event87961
    frameStart := 87954 },
  { event := event87962
    frameStart := 87954 },
  { event := event87963
    frameStart := 87954 },
  { event := event87964
    frameStart := 87954 },
  { event := event87965
    frameStart := 87954 },
  { event := event87966
    frameStart := 87954 },
  { event := event87967
    frameStart := 87954 }
]

def eventLeaf5498 : Array AnnotatedEvent := #[
  { event := event87968
    frameStart := 87954 },
  { event := event87969
    frameStart := 87954 },
  { event := event87970
    frameStart := 87954 },
  { event := event87971
    frameStart := 87954 },
  { event := event87972
    frameStart := 87954 },
  { event := event87973
    frameStart := 87954 },
  { event := event87974
    frameStart := 87954 },
  { event := event87975
    frameStart := 87954 },
  { event := event87976
    frameStart := 87954 },
  { event := event87977
    frameStart := 87954 },
  { event := event87978
    frameStart := 87954 },
  { event := event87979
    frameStart := 87954 },
  { event := event87980
    frameStart := 87954 },
  { event := event87981
    frameStart := 87954 },
  { event := event87982
    frameStart := 87954 },
  { event := event87983
    frameStart := 87954 }
]

def eventLeaf5499 : Array AnnotatedEvent := #[
  { event := event87984
    frameStart := 87954 },
  { event := event87985
    frameStart := 87954 },
  { event := event87986
    frameStart := 87954 },
  { event := event87987
    frameStart := 87954 },
  { event := event87988
    frameStart := 87954 },
  { event := event87989
    frameStart := 87954 },
  { event := event87990
    frameStart := 87954 },
  { event := event87991
    frameStart := 87954 },
  { event := event87992
    frameStart := 87954 },
  { event := event87993
    frameStart := 87954 },
  { event := event87994
    frameStart := 87954 },
  { event := event87995
    frameStart := 87954 },
  { event := event87996
    frameStart := 87954 },
  { event := event87997
    frameStart := 87954 },
  { event := event87998
    frameStart := 87954 },
  { event := event87999
    frameStart := 87954 }
]

def eventLeaf5500 : Array AnnotatedEvent := #[
  { event := event88000
    frameStart := 87954 },
  { event := event88001
    frameStart := 87954 },
  { event := event88002
    frameStart := 87954 },
  { event := event88003
    frameStart := 87954 },
  { event := event88004
    frameStart := 87954 },
  { event := event88005
    frameStart := 87954 },
  { event := event88006
    frameStart := 87954 },
  { event := event88007
    frameStart := 87954 },
  { event := event88008
    frameStart := 87954 },
  { event := event88009
    frameStart := 87954 },
  { event := event88010
    frameStart := 87954 },
  { event := event88011
    frameStart := 87954 },
  { event := event88012
    frameStart := 87954 },
  { event := event88013
    frameStart := 87954 },
  { event := event88014
    frameStart := 87954 },
  { event := event88015
    frameStart := 87954 }
]

def eventLeaf5501 : Array AnnotatedEvent := #[
  { event := event88016
    frameStart := 87954 },
  { event := event88017
    frameStart := 87954 },
  { event := event88018
    frameStart := 87954 },
  { event := event88019
    frameStart := 87954 },
  { event := event88020
    frameStart := 87954 },
  { event := event88021
    frameStart := 87954 },
  { event := event88022
    frameStart := 87954 },
  { event := event88023
    frameStart := 87954 },
  { event := event88024
    frameStart := 87954 },
  { event := event88025
    frameStart := 87954 },
  { event := event88026
    frameStart := 87954 },
  { event := event88027
    frameStart := 87954 },
  { event := event88028
    frameStart := 87954 },
  { event := event88029
    frameStart := 87954 },
  { event := event88030
    frameStart := 87954 },
  { event := event88031
    frameStart := 87954 }
]

def eventLeaf5502 : Array AnnotatedEvent := #[
  { event := event88032
    frameStart := 87954 },
  { event := event88033
    frameStart := 87954 },
  { event := event88034
    frameStart := 87954 },
  { event := event88035
    frameStart := 87954 },
  { event := event88036
    frameStart := 87954 },
  { event := event88037
    frameStart := 87954 },
  { event := event88038
    frameStart := 87954 },
  { event := event88039
    frameStart := 87954 },
  { event := event88040
    frameStart := 87954 },
  { event := event88041
    frameStart := 87954 },
  { event := event88042
    frameStart := 87954 },
  { event := event88043
    frameStart := 87954 },
  { event := event88044
    frameStart := 87954 },
  { event := event88045
    frameStart := 87954 },
  { event := event88046
    frameStart := 87954 },
  { event := event88047
    frameStart := 87954 }
]

def eventLeaf5503 : Array AnnotatedEvent := #[
  { event := event88048
    frameStart := 87954 },
  { event := event88049
    frameStart := 87954 },
  { event := event88050
    frameStart := 87954 },
  { event := event88051
    frameStart := 87954 },
  { event := event88052
    frameStart := 87954 },
  { event := event88053
    frameStart := 87954 },
  { event := event88054
    frameStart := 87954 },
  { event := event88055
    frameStart := 87954 },
  { event := event88056
    frameStart := 87954 },
  { event := event88057
    frameStart := 87954 },
  { event := event88058
    frameStart := 0 },
  { event := event88059
    frameStart := 0 },
  { event := event88060
    frameStart := 0 },
  { event := event88061
    frameStart := 0 },
  { event := event88062
    frameStart := 0 },
  { event := event88063
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events343
