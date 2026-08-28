import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events378

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event96768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6702⟩⟩) 0 ⟨6689⟩ 96701

def event96769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6702⟩⟩) (.authority (.operator))

def exact96770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩]

theorem exact96770RawTermsValid :
    exact96770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96770 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6702⟩⟩) exact96770RawTerms .large 96769 .exactZero (none)

def event96771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16458⟩⟩) 0 ⟨6702⟩ 96770

def event96772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16458⟩⟩) 1 ⟨16457⟩ 96767

def event96773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16458⟩⟩) (.sum [.predecessor 0 96771 .coefficient, .predecessor 1 96772 .coefficient])

def exact96774RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96774RawTermsValid :
    exact96774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96774 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16458⟩⟩) exact96774RawTerms .large 96773 .exactZero (none)

def event96775 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25364⟩⟩) 0 ⟨16458⟩ 96774

def event96776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25364⟩⟩) 1 ⟨25363⟩ 96759

def event96777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25364⟩⟩) (.sum [.predecessor 0 96775 .coefficient, .predecessor 1 96776 .coefficient])

def exact96778RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96778RawTermsValid :
    exact96778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96778 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25364⟩⟩) exact96778RawTerms .large 96777 .exactZero (none)

def event96779 : Event := .preFoldPolynomial 96778 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact96780RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event96780 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25364⟩⟩) 96779 exact96780RawTerms .large 96777 .exactZero (none)

def event96781 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12348⟩⟩) ⟨⟨115⟩, ⟨20⟩, ⟨109⟩⟩ ⟨96639, 96781⟩

def event96782 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19880⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩) (1) 0 2 (.universal 96781 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩) (none) 96780)

def event96783 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19880⟩⟩, .relation 96782 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩)

def event96784 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19880⟩⟩, .relation 96782 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩, (-1)⟩)

def event96785 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19880⟩⟩, .relation 96782 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩, (1)⟩)

def event96786 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19880⟩⟩, .relation 96782 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact96787RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96787RawTermsValid :
    exact96787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96787 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19880⟩⟩) exact96787RawTerms .large 96635 (.finite 1811303510016) (some (96637))

def event96788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25362⟩⟩) 0 ⟨19880⟩ 96787

def event96789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25362⟩⟩) 1 ⟨25361⟩ 96625

def event96790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25362⟩⟩) (.sum [.predecessor 0 96788 .coefficient, .predecessor 1 96789 .coefficient])

def event96791 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25362⟩⟩, .operator (⟨96787, 2⟩, ⟨96625, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩, (-1)⟩)

def event96792 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25362⟩⟩, .operator (⟨96787, 1⟩, ⟨96625, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩, (1)⟩)

def event96793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25362⟩⟩) (.sum [.result 96787 .summary, .result 96625 .summary])

def exact96794RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96794RawTermsValid :
    exact96794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96794 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25362⟩⟩) exact96794RawTerms .large 96790 (.finite 352127895089152) (some (96793))

def event96795 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28918⟩⟩) 0 ⟨25362⟩ 96794

def event96796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28918⟩⟩) 1 ⟨28916⟩ 96541

def event96797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28918⟩⟩) (.product (.predecessor 0 96795 .coefficient) (.predecessor 1 96796 .coefficient) (⟨false, false, none, none, none⟩))

def event96798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28918⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩) [⟨.result 96541 .coefficient, false, none⟩])

def event96799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28918⟩⟩) (.product (.result 96794 .summary) (.transfer 96798) (⟨false, false, none, none, none⟩))

def event96800 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28918⟩⟩, .operator (⟨96794, 0⟩, ⟨96541, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (1)⟩)

def event96801 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28918⟩⟩, .operator (⟨96794, 1⟩, ⟨96541, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (-1)⟩)

def event96802 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28918⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28916⟩⟩) ⟨24468⟩ 96538)

def event96803 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28918⟩⟩, .relation 96802 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (-1)⟩)

def exact96804RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (-1)⟩]

theorem exact96804RawTermsValid :
    exact96804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96804 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28918⟩⟩) exact96804RawTerms .large 96797 (.finite 1292315009023509266432) (some (96799))

def event96805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22109⟩⟩) 0 ⟨16456⟩ 4698

def event96806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22109⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact96807RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩, (1)⟩]

theorem exact96807RawTermsValid :
    exact96807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96807 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22109⟩⟩) exact96807RawTerms (.finite 136065468) 96806 .exactZero (none)

def event96808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22111⟩⟩) 0 ⟨22109⟩ 96807

def event96809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22111⟩⟩) 1 ⟨2348⟩ 4

def event96810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22111⟩⟩) (.scale (.predecessor 0 96808 .coefficient) (.value (.predecessor 1 96809 .coefficient)))

def exact96811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩, (1)⟩]

theorem exact96811RawTermsValid :
    exact96811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22111⟩⟩) exact96811RawTerms (.finite 136065468) 96810 .exactZero (none)

def event96812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22112⟩⟩) 0 ⟨5509⟩ 94462

def event96813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22112⟩⟩) 1 ⟨22111⟩ 96811

def event96814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22112⟩⟩) (.product (.predecessor 0 96812 .coefficient) (.predecessor 1 96813 .coefficient) (⟨false, false, none, none, none⟩))

def event96815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22112⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩) [⟨.result 96807 .coefficient, false, none⟩])

def event96816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22112⟩⟩) (.product (.result 94462 .summary) (.transfer 96815) (⟨false, false, none, none, none⟩))

def event96817 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22112⟩⟩, .operator (⟨94462, 0⟩, ⟨96811, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩, (1)⟩)

def event96818 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22110⟩⟩)

def event96819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event96820 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event96821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event96822 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event96823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 96822

def event96824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 96820

def event96825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 96823 .coefficient) (.value (.predecessor 1 96824 .coefficient)))

def event96826 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event96827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12346⟩⟩) 0 ⟨5503⟩ 96826

def event96828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12346⟩⟩) (.authority (.programFamilyFact))

def exact96829RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12346⟩⟩], []⟩, (1)⟩]

theorem exact96829RawTermsValid :
    exact96829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12346⟩⟩) exact96829RawTerms (.finite 40) 96828 .exactZero (none)

def event96830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9805⟩⟩) 0 ⟨5503⟩ 96826

def event96831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9805⟩⟩) (.authority (.programFamilyFact))

def exact96832RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩], []⟩, (1)⟩]

theorem exact96832RawTermsValid :
    exact96832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96832 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9805⟩⟩) exact96832RawTerms (.finite 40) 96831 .exactZero (none)

def event96833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12347⟩⟩) 0 ⟨9805⟩ 96832

def event96834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12347⟩⟩) 1 ⟨12346⟩ 96829

def event96835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12347⟩⟩) (.product (.predecessor 0 96833 .coefficient) (.predecessor 1 96834 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12347⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], []⟩) [⟨.result 96832 .coefficient, true, some 1⟩, ⟨.result 96829 .coefficient, true, some 1⟩])

def event96837 : Event := .survivorFold (1) 96836

def exact96838RawTerms : List Term := []

theorem exact96838RawTermsValid :
    exact96838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96838 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12347⟩⟩) exact96838RawTerms (.finite 1600) 96835 (.finite 1600) (some (96836))

def event96839 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12348⟩⟩) 0 ⟨12347⟩ 96838

def event96840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12348⟩⟩) (.identity (.predecessor 0 96839 .coefficient))

def event96841 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12348⟩⟩) (.finite 1600)

def event96842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16455⟩⟩) 0 ⟨12348⟩ 96841

def event96843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16455⟩⟩) (.authority (.programFamilyFact))

def exact96844RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], []⟩, (1)⟩]

theorem exact96844RawTermsValid :
    exact96844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16455⟩⟩) exact96844RawTerms (.finite 40) 96843 .exactZero (none)

def event96845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16456⟩⟩) 0 ⟨16455⟩ 96844

def event96846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16456⟩⟩) (.identity (.predecessor 0 96845 .coefficient))

def event96847 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16456⟩⟩) (.finite 40)

def event96848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22109⟩⟩) 0 ⟨16456⟩ 96847

def event96849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22109⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact96850RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩, (1)⟩]

theorem exact96850RawTermsValid :
    exact96850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96850 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22109⟩⟩) exact96850RawTerms (.finite 136065468) 96849 .exactZero (none)

def event96851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact96852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact96852RawTermsValid :
    exact96852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact96852RawTerms .large 96851 .exactZero (none)

def event96853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22110⟩⟩) 0 ⟨6⟩ 96852

def event96854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22110⟩⟩) 1 ⟨22109⟩ 96850

def event96855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22110⟩⟩) (.product (.predecessor 0 96853 .coefficient) (.predecessor 1 96854 .coefficient) (⟨false, false, none, none, none⟩))

def event96856 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22110⟩⟩, .operator (⟨96852, 0⟩, ⟨96850, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩, (1)⟩)

def exact96857RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩, (1)⟩]

theorem exact96857RawTermsValid :
    exact96857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96857 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22110⟩⟩) exact96857RawTerms .large 96855 .exactZero (none)

def event96858 : Event := .preFoldPolynomial 96857 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩, (1)⟩] .exactZero none

def exact96859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩, (1)⟩]

def event96859 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22110⟩⟩) 96858 exact96859RawTerms .large 96855 .exactZero (none)

def event96860 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28921⟩⟩)

def event96861 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event96862 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event96863 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event96864 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event96865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 96864

def event96866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 96862

def event96867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 96865 .coefficient) (.value (.predecessor 1 96866 .coefficient)))

def event96868 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event96869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12346⟩⟩) 0 ⟨5503⟩ 96868

def event96870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12346⟩⟩) (.authority (.programFamilyFact))

def exact96871RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12346⟩⟩], []⟩, (1)⟩]

theorem exact96871RawTermsValid :
    exact96871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12346⟩⟩) exact96871RawTerms (.finite 40) 96870 .exactZero (none)

def event96872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9805⟩⟩) 0 ⟨5503⟩ 96868

def event96873 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9805⟩⟩) (.authority (.programFamilyFact))

def exact96874RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩], []⟩, (1)⟩]

theorem exact96874RawTermsValid :
    exact96874RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96874 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9805⟩⟩) exact96874RawTerms (.finite 40) 96873 .exactZero (none)

def event96875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12347⟩⟩) 0 ⟨9805⟩ 96874

def event96876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12347⟩⟩) 1 ⟨12346⟩ 96871

def event96877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12347⟩⟩) (.product (.predecessor 0 96875 .coefficient) (.predecessor 1 96876 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96878 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12347⟩⟩, .operator (⟨96874, 0⟩, ⟨96871, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], []⟩, (1)⟩)

def exact96879RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], []⟩, (1)⟩]

theorem exact96879RawTermsValid :
    exact96879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96879 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12347⟩⟩) exact96879RawTerms (.finite 1600) 96877 .exactZero (none)

def event96880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12348⟩⟩) 0 ⟨12347⟩ 96879

def event96881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12348⟩⟩) (.identity (.predecessor 0 96880 .coefficient))

def event96882 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12348⟩⟩) (.finite 1600)

def event96883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16455⟩⟩) 0 ⟨12348⟩ 96882

def event96884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16455⟩⟩) (.authority (.programFamilyFact))

def exact96885RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], []⟩, (1)⟩]

theorem exact96885RawTermsValid :
    exact96885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96885 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16455⟩⟩) exact96885RawTerms (.finite 40) 96884 .exactZero (none)

def event96886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16456⟩⟩) 0 ⟨16455⟩ 96885

def event96887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16456⟩⟩) (.identity (.predecessor 0 96886 .coefficient))

def event96888 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16456⟩⟩) (.finite 40)

def event96889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24466⟩⟩) 0 ⟨16456⟩ 96888

def event96890 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24466⟩⟩) (.authority (.programFamilyFact))

def event96891 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24466⟩⟩) (.finite 3720)

def event96892 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event96893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24468⟩⟩) 0 ⟨6689⟩ 96892

def event96894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24468⟩⟩) 1 ⟨24466⟩ 96891

def event96895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24468⟩⟩) (.authority (.operator))

def exact96896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (1)⟩]

theorem exact96896RawTermsValid :
    exact96896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96896 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24468⟩⟩) exact96896RawTerms .large 96895 .exactZero (none)

def event96897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28916⟩⟩) 0 ⟨24468⟩ 96896

def event96898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28916⟩⟩) (.authority (.operator))

def exact96899RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (1)⟩]

theorem exact96899RawTermsValid :
    exact96899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96899 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28916⟩⟩) exact96899RawTerms (.finite 8192) 96898 .exactZero (none)

def event96900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event96901 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event96902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16497⟩⟩) 0 ⟨16456⟩ 96888

def event96903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16497⟩⟩) 1 ⟨110⟩ 96901

def event96904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16497⟩⟩) (.sum [.predecessor 0 96902 .coefficient, .predecessor 1 96903 .coefficient])

def event96905 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16497⟩⟩) (.finite 40)

def event96906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16498⟩⟩) 0 ⟨16497⟩ 96905

def event96907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16498⟩⟩) (.identity (.predecessor 0 96906 .coefficient))

def exact96908RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], []⟩, (1)⟩]

theorem exact96908RawTermsValid :
    exact96908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96908 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16498⟩⟩) exact96908RawTerms (.finite 40) 96907 .exactZero (none)

def event96909 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact96910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96910RawTermsValid :
    exact96910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96910 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact96910RawTerms .large 96909 .exactZero (none)

def event96911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16499⟩⟩) 0 ⟨6544⟩ 96910

def event96912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16499⟩⟩) 1 ⟨16498⟩ 96908

def event96913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16499⟩⟩) (.product (.predecessor 0 96911 .coefficient) (.predecessor 1 96912 .coefficient) (⟨false, false, none, none, none⟩))

def event96914 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16499⟩⟩, .operator (⟨96910, 0⟩, ⟨96908, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact96915RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96915RawTermsValid :
    exact96915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96915 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16499⟩⟩) exact96915RawTerms .large 96913 .exactZero (none)

def event96916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6702⟩⟩) 0 ⟨6689⟩ 96892

def event96917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6702⟩⟩) (.authority (.operator))

def exact96918RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩]

theorem exact96918RawTermsValid :
    exact96918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6702⟩⟩) exact96918RawTerms .large 96917 .exactZero (none)

def event96919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16500⟩⟩) 0 ⟨6702⟩ 96918

def event96920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16500⟩⟩) 1 ⟨16499⟩ 96915

def event96921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16500⟩⟩) (.sum [.predecessor 0 96919 .coefficient, .predecessor 1 96920 .coefficient])

def exact96922RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96922RawTermsValid :
    exact96922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96922 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16500⟩⟩) exact96922RawTerms .large 96921 .exactZero (none)

def event96923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28917⟩⟩) 0 ⟨16500⟩ 96922

def event96924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28917⟩⟩) 1 ⟨28916⟩ 96899

def event96925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28917⟩⟩) (.product (.predecessor 0 96923 .coefficient) (.predecessor 1 96924 .coefficient) (⟨false, false, none, none, none⟩))

def event96926 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28917⟩⟩, .operator (⟨96922, 0⟩, ⟨96899, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (1)⟩)

def event96927 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28917⟩⟩, .operator (⟨96922, 1⟩, ⟨96899, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (-1)⟩)

def event96928 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28917⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28916⟩⟩) ⟨24468⟩ 96896)

def event96929 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28917⟩⟩, .relation 96928 0, ⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (-1)⟩)

def exact96930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (-1)⟩]

theorem exact96930RawTermsValid :
    exact96930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28917⟩⟩) exact96930RawTerms .large 96925 .exactZero (none)

def event96931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17897⟩⟩) 0 ⟨16456⟩ 96888

def event96932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17897⟩⟩) (.authority (.programFamilyFact))

def exact96933RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], []⟩, (1)⟩]

theorem exact96933RawTermsValid :
    exact96933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17897⟩⟩) exact96933RawTerms (.finite 62) 96932 .exactZero (none)

def event96934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17898⟩⟩) 0 ⟨6544⟩ 96910

def event96935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17898⟩⟩) 1 ⟨17897⟩ 96933

def event96936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17898⟩⟩) (.product (.predecessor 0 96934 .coefficient) (.predecessor 1 96935 .coefficient) (⟨false, true, none, none, some 1⟩))

def event96937 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17898⟩⟩, .operator (⟨96910, 0⟩, ⟨96933, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact96938RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96938RawTermsValid :
    exact96938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96938 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17898⟩⟩) exact96938RawTerms .large 96936 .exactZero (none)

def event96939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6733⟩⟩) 0 ⟨6689⟩ 96892

def event96940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6733⟩⟩) (.authority (.operator))

def exact96941RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩]

theorem exact96941RawTermsValid :
    exact96941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96941 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6733⟩⟩) exact96941RawTerms .large 96940 .exactZero (none)

def event96942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17899⟩⟩) 0 ⟨6733⟩ 96941

def event96943 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17899⟩⟩) 1 ⟨17898⟩ 96938

def event96944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17899⟩⟩) (.sum [.predecessor 0 96942 .coefficient, .predecessor 1 96943 .coefficient])

def exact96945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96945RawTermsValid :
    exact96945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96945 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17899⟩⟩) exact96945RawTerms .large 96944 .exactZero (none)

def event96946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28921⟩⟩) 0 ⟨17899⟩ 96945

def event96947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28921⟩⟩) 1 ⟨28917⟩ 96930

def event96948 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28921⟩⟩) (.sum [.predecessor 0 96946 .coefficient, .predecessor 1 96947 .coefficient])

def exact96949RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96949RawTermsValid :
    exact96949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96949 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28921⟩⟩) exact96949RawTerms .large 96948 .exactZero (none)

def event96950 : Event := .preFoldPolynomial 96949 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact96951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event96951 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28921⟩⟩) 96950 exact96951RawTerms .large 96948 .exactZero (none)

def event96952 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16456⟩⟩) ⟨⟨146⟩, ⟨54⟩, ⟨109⟩⟩ ⟨96818, 96952⟩

def event96953 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22112⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩) (1) 0 2 (.universal 96952 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩) (none) 96951)

def event96954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22112⟩⟩, .relation 96953 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩)

def event96955 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22112⟩⟩, .relation 96953 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (-1)⟩)

def event96956 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22112⟩⟩, .relation 96953 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (1)⟩)

def event96957 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22112⟩⟩, .relation 96953 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact96958RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96958RawTermsValid :
    exact96958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96958 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22112⟩⟩) exact96958RawTerms .large 96814 (.finite 1811303510016) (some (96816))

def event96959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28919⟩⟩) 0 ⟨22112⟩ 96958

def event96960 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28919⟩⟩) 1 ⟨28918⟩ 96804

def event96961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28919⟩⟩) (.sum [.predecessor 0 96959 .coefficient, .predecessor 1 96960 .coefficient])

def event96962 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28919⟩⟩, .operator (⟨96958, 0⟩, ⟨96804, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩, (1)⟩)

def event96963 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28919⟩⟩, .operator (⟨96958, 2⟩, ⟨96804, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩, (-1)⟩)

def event96964 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28919⟩⟩) (.sum [.result 96958 .summary, .result 96804 .summary])

def exact96965RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96965RawTermsValid :
    exact96965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96965 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28919⟩⟩) exact96965RawTerms .large 96961 (.finite 1292315010834812776448) (some (96964))

def event96966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24403⟩⟩) 0 ⟨16372⟩ 4721

def event96967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24403⟩⟩) (.authority (.programFamilyFact))

def event96968 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24403⟩⟩) (.finite 3720)

def event96969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24405⟩⟩) 0 ⟨6689⟩ 5477

def event96970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24405⟩⟩) 1 ⟨24403⟩ 96968

def event96971 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24405⟩⟩) (.authority (.operator))

def exact96972RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (1)⟩]

theorem exact96972RawTermsValid :
    exact96972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24405⟩⟩) exact96972RawTerms .large 96971 .exactZero (none)

def event96973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28699⟩⟩) 0 ⟨24405⟩ 96972

def event96974 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28699⟩⟩) (.authority (.operator))

def exact96975RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (1)⟩]

theorem exact96975RawTermsValid :
    exact96975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96975 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28699⟩⟩) exact96975RawTerms (.finite 8192) 96974 .exactZero (none)

def event96976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23115⟩⟩) 0 ⟨11935⟩ 4715

def event96977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23115⟩⟩) (.authority (.programFamilyFact))

def event96978 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23115⟩⟩) (.finite 3720)

def event96979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23116⟩⟩) 0 ⟨6689⟩ 5477

def event96980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23116⟩⟩) 1 ⟨23115⟩ 96978

def event96981 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23116⟩⟩) (.authority (.operator))

def exact96982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23116⟩⟩]⟩, (1)⟩]

theorem exact96982RawTermsValid :
    exact96982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96982 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23116⟩⟩) exact96982RawTerms .large 96981 .exactZero (none)

def event96983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25206⟩⟩) 0 ⟨23116⟩ 96982

def event96984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25206⟩⟩) (.authority (.operator))

def exact96985RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25206⟩⟩]⟩, (1)⟩]

theorem exact96985RawTermsValid :
    exact96985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96985 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25206⟩⟩) exact96985RawTerms (.finite 8192) 96984 .exactZero (none)

def event96986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11936⟩⟩) 0 ⟨11933⟩ 4704

def event96987 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11936⟩⟩) 1 ⟨6564⟩ 32

def event96988 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11936⟩⟩) (.tensor (.predecessor 0 96986 .coefficient) (.predecessor 1 96987 .coefficient) true false)

def event96989 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11936⟩⟩, .operator (⟨4704, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact96990RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96990RawTermsValid :
    exact96990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96990 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11936⟩⟩) exact96990RawTerms .large 96988 .exactZero (none)

def event96991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7121⟩⟩) 0 ⟨5506⟩ 27

def event96992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7121⟩⟩) 1 ⟨6784⟩ 9478

def event96993 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7121⟩⟩) (.product (.predecessor 0 96991 .coefficient) (.predecessor 1 96992 .coefficient) (⟨false, false, none, none, none⟩))

def event96994 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7121⟩⟩, .operator (⟨27, 0⟩, ⟨9478, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩)

def exact96995RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩]

theorem exact96995RawTermsValid :
    exact96995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96995 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7121⟩⟩) exact96995RawTerms .large 96993 .exactZero (none)

def event96996 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11937⟩⟩) 0 ⟨7121⟩ 96995

def event96997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11937⟩⟩) 1 ⟨11936⟩ 96990

def event96998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11937⟩⟩) (.sum [.predecessor 0 96996 .coefficient, .predecessor 1 96997 .coefficient])

def exact96999RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96999RawTermsValid :
    exact96999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96999 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11937⟩⟩) exact96999RawTerms .large 96998 .exactZero (none)

def event97000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11938⟩⟩) 0 ⟨11937⟩ 96999

def event97001 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11938⟩⟩) 1 ⟨98⟩ 9470

def event97002 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11938⟩⟩) (.sum [.predecessor 0 97000 .coefficient, .predecessor 1 97001 .coefficient])

def event97003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11938⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩) [⟨.result 9470 .coefficient, false, none⟩])

def event97004 : Event := .survivorFold (1) 97003

def exact97005RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97005RawTermsValid :
    exact97005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97005 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11938⟩⟩) exact97005RawTerms .large 97002 (.finite 26) (some (97003))

def event97006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11939⟩⟩) 0 ⟨11938⟩ 97005

def event97007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11939⟩⟩) 1 ⟨9700⟩ 4707

def event97008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11939⟩⟩) (.product (.predecessor 0 97006 .coefficient) (.predecessor 1 97007 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11939⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩], []⟩) [⟨.result 4707 .coefficient, true, some 1⟩])

def event97010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11939⟩⟩) (.product (.result 97005 .summary) (.transfer 97009) (⟨false, false, none, none, none⟩))

def event97011 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11939⟩⟩, .operator (⟨97005, 1⟩, ⟨4707, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event97012 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11939⟩⟩, .operator (⟨97005, 0⟩, ⟨4707, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩)

def exact97013RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97013RawTermsValid :
    exact97013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97013 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11939⟩⟩) exact97013RawTerms .large 97008 (.finite 29952) (some (97010))

def event97014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9701⟩⟩) 0 ⟨9700⟩ 4707

def event97015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9701⟩⟩) 1 ⟨6564⟩ 32

def event97016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9701⟩⟩) (.tensor (.predecessor 0 97014 .coefficient) (.predecessor 1 97015 .coefficient) true false)

def event97017 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9701⟩⟩, .operator (⟨4707, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97018RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97018RawTermsValid :
    exact97018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97018 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9701⟩⟩) exact97018RawTerms .large 97016 .exactZero (none)

def event97019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7101⟩⟩) 0 ⟨5506⟩ 27

def event97020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7101⟩⟩) 1 ⟨6764⟩ 9519

def event97021 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7101⟩⟩) (.product (.predecessor 0 97019 .coefficient) (.predecessor 1 97020 .coefficient) (⟨false, false, none, none, none⟩))

def event97022 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7101⟩⟩, .operator (⟨27, 0⟩, ⟨9519, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩)

def exact97023RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩]

theorem exact97023RawTermsValid :
    exact97023RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97023 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7101⟩⟩) exact97023RawTerms .large 97021 .exactZero (none)

def eventLeaf6048 : Array AnnotatedEvent := #[
  { event := event96768
    frameStart := 96675 },
  { event := event96769
    frameStart := 96675 },
  { event := event96770
    frameStart := 96675 },
  { event := event96771
    frameStart := 96675 },
  { event := event96772
    frameStart := 96675 },
  { event := event96773
    frameStart := 96675 },
  { event := event96774
    frameStart := 96675 },
  { event := event96775
    frameStart := 96675 },
  { event := event96776
    frameStart := 96675 },
  { event := event96777
    frameStart := 96675 },
  { event := event96778
    frameStart := 96675 },
  { event := event96779
    frameStart := 96675 },
  { event := event96780
    frameStart := 96675 },
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
    frameStart := 96818 },
  { event := event96819
    frameStart := 96818 },
  { event := event96820
    frameStart := 96818 },
  { event := event96821
    frameStart := 96818 },
  { event := event96822
    frameStart := 96818 },
  { event := event96823
    frameStart := 96818 },
  { event := event96824
    frameStart := 96818 },
  { event := event96825
    frameStart := 96818 },
  { event := event96826
    frameStart := 96818 },
  { event := event96827
    frameStart := 96818 },
  { event := event96828
    frameStart := 96818 },
  { event := event96829
    frameStart := 96818 },
  { event := event96830
    frameStart := 96818 },
  { event := event96831
    frameStart := 96818 }
]

def eventLeaf6052 : Array AnnotatedEvent := #[
  { event := event96832
    frameStart := 96818 },
  { event := event96833
    frameStart := 96818 },
  { event := event96834
    frameStart := 96818 },
  { event := event96835
    frameStart := 96818 },
  { event := event96836
    frameStart := 96818 },
  { event := event96837
    frameStart := 96818 },
  { event := event96838
    frameStart := 96818 },
  { event := event96839
    frameStart := 96818 },
  { event := event96840
    frameStart := 96818 },
  { event := event96841
    frameStart := 96818 },
  { event := event96842
    frameStart := 96818 },
  { event := event96843
    frameStart := 96818 },
  { event := event96844
    frameStart := 96818 },
  { event := event96845
    frameStart := 96818 },
  { event := event96846
    frameStart := 96818 },
  { event := event96847
    frameStart := 96818 }
]

def eventLeaf6053 : Array AnnotatedEvent := #[
  { event := event96848
    frameStart := 96818 },
  { event := event96849
    frameStart := 96818 },
  { event := event96850
    frameStart := 96818 },
  { event := event96851
    frameStart := 96818 },
  { event := event96852
    frameStart := 96818 },
  { event := event96853
    frameStart := 96818 },
  { event := event96854
    frameStart := 96818 },
  { event := event96855
    frameStart := 96818 },
  { event := event96856
    frameStart := 96818 },
  { event := event96857
    frameStart := 96818 },
  { event := event96858
    frameStart := 96818 },
  { event := event96859
    frameStart := 96818 },
  { event := event96860
    frameStart := 96860 },
  { event := event96861
    frameStart := 96860 },
  { event := event96862
    frameStart := 96860 },
  { event := event96863
    frameStart := 96860 }
]

def eventLeaf6054 : Array AnnotatedEvent := #[
  { event := event96864
    frameStart := 96860 },
  { event := event96865
    frameStart := 96860 },
  { event := event96866
    frameStart := 96860 },
  { event := event96867
    frameStart := 96860 },
  { event := event96868
    frameStart := 96860 },
  { event := event96869
    frameStart := 96860 },
  { event := event96870
    frameStart := 96860 },
  { event := event96871
    frameStart := 96860 },
  { event := event96872
    frameStart := 96860 },
  { event := event96873
    frameStart := 96860 },
  { event := event96874
    frameStart := 96860 },
  { event := event96875
    frameStart := 96860 },
  { event := event96876
    frameStart := 96860 },
  { event := event96877
    frameStart := 96860 },
  { event := event96878
    frameStart := 96860 },
  { event := event96879
    frameStart := 96860 }
]

def eventLeaf6055 : Array AnnotatedEvent := #[
  { event := event96880
    frameStart := 96860 },
  { event := event96881
    frameStart := 96860 },
  { event := event96882
    frameStart := 96860 },
  { event := event96883
    frameStart := 96860 },
  { event := event96884
    frameStart := 96860 },
  { event := event96885
    frameStart := 96860 },
  { event := event96886
    frameStart := 96860 },
  { event := event96887
    frameStart := 96860 },
  { event := event96888
    frameStart := 96860 },
  { event := event96889
    frameStart := 96860 },
  { event := event96890
    frameStart := 96860 },
  { event := event96891
    frameStart := 96860 },
  { event := event96892
    frameStart := 96860 },
  { event := event96893
    frameStart := 96860 },
  { event := event96894
    frameStart := 96860 },
  { event := event96895
    frameStart := 96860 }
]

def eventLeaf6056 : Array AnnotatedEvent := #[
  { event := event96896
    frameStart := 96860 },
  { event := event96897
    frameStart := 96860 },
  { event := event96898
    frameStart := 96860 },
  { event := event96899
    frameStart := 96860 },
  { event := event96900
    frameStart := 96860 },
  { event := event96901
    frameStart := 96860 },
  { event := event96902
    frameStart := 96860 },
  { event := event96903
    frameStart := 96860 },
  { event := event96904
    frameStart := 96860 },
  { event := event96905
    frameStart := 96860 },
  { event := event96906
    frameStart := 96860 },
  { event := event96907
    frameStart := 96860 },
  { event := event96908
    frameStart := 96860 },
  { event := event96909
    frameStart := 96860 },
  { event := event96910
    frameStart := 96860 },
  { event := event96911
    frameStart := 96860 }
]

def eventLeaf6057 : Array AnnotatedEvent := #[
  { event := event96912
    frameStart := 96860 },
  { event := event96913
    frameStart := 96860 },
  { event := event96914
    frameStart := 96860 },
  { event := event96915
    frameStart := 96860 },
  { event := event96916
    frameStart := 96860 },
  { event := event96917
    frameStart := 96860 },
  { event := event96918
    frameStart := 96860 },
  { event := event96919
    frameStart := 96860 },
  { event := event96920
    frameStart := 96860 },
  { event := event96921
    frameStart := 96860 },
  { event := event96922
    frameStart := 96860 },
  { event := event96923
    frameStart := 96860 },
  { event := event96924
    frameStart := 96860 },
  { event := event96925
    frameStart := 96860 },
  { event := event96926
    frameStart := 96860 },
  { event := event96927
    frameStart := 96860 }
]

def eventLeaf6058 : Array AnnotatedEvent := #[
  { event := event96928
    frameStart := 96860 },
  { event := event96929
    frameStart := 96860 },
  { event := event96930
    frameStart := 96860 },
  { event := event96931
    frameStart := 96860 },
  { event := event96932
    frameStart := 96860 },
  { event := event96933
    frameStart := 96860 },
  { event := event96934
    frameStart := 96860 },
  { event := event96935
    frameStart := 96860 },
  { event := event96936
    frameStart := 96860 },
  { event := event96937
    frameStart := 96860 },
  { event := event96938
    frameStart := 96860 },
  { event := event96939
    frameStart := 96860 },
  { event := event96940
    frameStart := 96860 },
  { event := event96941
    frameStart := 96860 },
  { event := event96942
    frameStart := 96860 },
  { event := event96943
    frameStart := 96860 }
]

def eventLeaf6059 : Array AnnotatedEvent := #[
  { event := event96944
    frameStart := 96860 },
  { event := event96945
    frameStart := 96860 },
  { event := event96946
    frameStart := 96860 },
  { event := event96947
    frameStart := 96860 },
  { event := event96948
    frameStart := 96860 },
  { event := event96949
    frameStart := 96860 },
  { event := event96950
    frameStart := 96860 },
  { event := event96951
    frameStart := 96860 },
  { event := event96952
    frameStart := 0 },
  { event := event96953
    frameStart := 0 },
  { event := event96954
    frameStart := 0 },
  { event := event96955
    frameStart := 0 },
  { event := event96956
    frameStart := 0 },
  { event := event96957
    frameStart := 0 },
  { event := event96958
    frameStart := 0 },
  { event := event96959
    frameStart := 0 }
]

def eventLeaf6060 : Array AnnotatedEvent := #[
  { event := event96960
    frameStart := 0 },
  { event := event96961
    frameStart := 0 },
  { event := event96962
    frameStart := 0 },
  { event := event96963
    frameStart := 0 },
  { event := event96964
    frameStart := 0 },
  { event := event96965
    frameStart := 0 },
  { event := event96966
    frameStart := 0 },
  { event := event96967
    frameStart := 0 },
  { event := event96968
    frameStart := 0 },
  { event := event96969
    frameStart := 0 },
  { event := event96970
    frameStart := 0 },
  { event := event96971
    frameStart := 0 },
  { event := event96972
    frameStart := 0 },
  { event := event96973
    frameStart := 0 },
  { event := event96974
    frameStart := 0 },
  { event := event96975
    frameStart := 0 }
]

def eventLeaf6061 : Array AnnotatedEvent := #[
  { event := event96976
    frameStart := 0 },
  { event := event96977
    frameStart := 0 },
  { event := event96978
    frameStart := 0 },
  { event := event96979
    frameStart := 0 },
  { event := event96980
    frameStart := 0 },
  { event := event96981
    frameStart := 0 },
  { event := event96982
    frameStart := 0 },
  { event := event96983
    frameStart := 0 },
  { event := event96984
    frameStart := 0 },
  { event := event96985
    frameStart := 0 },
  { event := event96986
    frameStart := 0 },
  { event := event96987
    frameStart := 0 },
  { event := event96988
    frameStart := 0 },
  { event := event96989
    frameStart := 0 },
  { event := event96990
    frameStart := 0 },
  { event := event96991
    frameStart := 0 }
]

def eventLeaf6062 : Array AnnotatedEvent := #[
  { event := event96992
    frameStart := 0 },
  { event := event96993
    frameStart := 0 },
  { event := event96994
    frameStart := 0 },
  { event := event96995
    frameStart := 0 },
  { event := event96996
    frameStart := 0 },
  { event := event96997
    frameStart := 0 },
  { event := event96998
    frameStart := 0 },
  { event := event96999
    frameStart := 0 },
  { event := event97000
    frameStart := 0 },
  { event := event97001
    frameStart := 0 },
  { event := event97002
    frameStart := 0 },
  { event := event97003
    frameStart := 0 },
  { event := event97004
    frameStart := 0 },
  { event := event97005
    frameStart := 0 },
  { event := event97006
    frameStart := 0 },
  { event := event97007
    frameStart := 0 }
]

def eventLeaf6063 : Array AnnotatedEvent := #[
  { event := event97008
    frameStart := 0 },
  { event := event97009
    frameStart := 0 },
  { event := event97010
    frameStart := 0 },
  { event := event97011
    frameStart := 0 },
  { event := event97012
    frameStart := 0 },
  { event := event97013
    frameStart := 0 },
  { event := event97014
    frameStart := 0 },
  { event := event97015
    frameStart := 0 },
  { event := event97016
    frameStart := 0 },
  { event := event97017
    frameStart := 0 },
  { event := event97018
    frameStart := 0 },
  { event := event97019
    frameStart := 0 },
  { event := event97020
    frameStart := 0 },
  { event := event97021
    frameStart := 0 },
  { event := event97022
    frameStart := 0 },
  { event := event97023
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events378
