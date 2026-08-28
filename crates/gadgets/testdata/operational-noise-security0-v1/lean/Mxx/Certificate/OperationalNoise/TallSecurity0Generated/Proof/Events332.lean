import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events332

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event84992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26068⟩⟩) (.sum [.predecessor 0 84990 .coefficient, .predecessor 1 84991 .coefficient])

def event84993 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26068⟩⟩, .operator (⟨84989, 2⟩, ⟨84805, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (-1)⟩)

def event84994 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26068⟩⟩, .operator (⟨84989, 1⟩, ⟨84805, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (1)⟩)

def event84995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26068⟩⟩) (.sum [.result 84989 .summary, .result 84805 .summary])

def exact84996RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84996RawTermsValid :
    exact84996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84996 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26068⟩⟩) exact84996RawTerms .large 84992 (.finite 352060719116288) (some (84995))

def event84997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27868⟩⟩) 0 ⟨26068⟩ 84996

def event84998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27868⟩⟩) 1 ⟨27866⟩ 84721

def event84999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27868⟩⟩) (.product (.predecessor 0 84997 .coefficient) (.predecessor 1 84998 .coefficient) (⟨false, false, none, none, none⟩))

def event85000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27868⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩) [⟨.result 84721 .coefficient, false, none⟩])

def event85001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27868⟩⟩) (.product (.result 84996 .summary) (.transfer 85000) (⟨false, false, none, none, none⟩))

def event85002 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27868⟩⟩, .operator (⟨84996, 0⟩, ⟨84721, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (1)⟩)

def event85003 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27868⟩⟩, .operator (⟨84996, 1⟩, ⟨84721, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (-1)⟩)

def event85004 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27868⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27866⟩⟩) ⟨24162⟩ 84718)

def event85005 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27868⟩⟩, .relation 85004 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (-1)⟩)

def exact85006RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (-1)⟩]

theorem exact85006RawTermsValid :
    exact85006RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85006 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27868⟩⟩) exact85006RawTerms .large 84999 (.finite 1292068472128282820608) (some (85001))

def event85007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21400⟩⟩) 0 ⟨15941⟩ 4075

def event85008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21400⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact85009RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩, (1)⟩]

theorem exact85009RawTermsValid :
    exact85009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21400⟩⟩) exact85009RawTerms (.finite 136065468) 85008 .exactZero (none)

def event85010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21402⟩⟩) 0 ⟨21400⟩ 85009

def event85011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21402⟩⟩) 1 ⟨2348⟩ 4

def event85012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21402⟩⟩) (.scale (.predecessor 0 85010 .coefficient) (.value (.predecessor 1 85011 .coefficient)))

def exact85013RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩, (1)⟩]

theorem exact85013RawTermsValid :
    exact85013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85013 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21402⟩⟩) exact85013RawTerms (.finite 136065468) 85012 .exactZero (none)

def event85014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21403⟩⟩) 0 ⟨5541⟩ 80012

def event85015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21403⟩⟩) 1 ⟨21402⟩ 85013

def event85016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21403⟩⟩) (.product (.predecessor 0 85014 .coefficient) (.predecessor 1 85015 .coefficient) (⟨false, false, none, none, none⟩))

def event85017 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21403⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩) [⟨.result 85009 .coefficient, false, none⟩])

def event85018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21403⟩⟩) (.product (.result 80012 .summary) (.transfer 85017) (⟨false, false, none, none, none⟩))

def event85019 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21403⟩⟩, .operator (⟨80012, 0⟩, ⟨85013, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩, (1)⟩)

def event85020 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21401⟩⟩)

def event85021 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event85022 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event85023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event85024 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event85025 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event85026 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event85027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event85028 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event85029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 85028

def event85030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 85026

def event85031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 85029 .coefficient) (.value (.predecessor 1 85030 .coefficient)))

def event85032 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event85033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 85032

def event85034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 85024

def event85035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 85033 .coefficient, .predecessor 1 85034 .coefficient])

def event85036 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event85037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 85036

def event85038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 85022

def event85039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 85038 .coefficient))

def event85040 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event85041 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11469⟩⟩) 0 ⟨5536⟩ 85040

def event85042 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11469⟩⟩) (.authority (.programFamilyFact))

def exact85043RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩], []⟩, (1)⟩]

theorem exact85043RawTermsValid :
    exact85043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85043 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11469⟩⟩) exact85043RawTerms (.finite 18) 85042 .exactZero (none)

def event85044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14207⟩⟩) 0 ⟨5536⟩ 85040

def event85045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14207⟩⟩) (.authority (.programFamilyFact))

def exact85046RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩]

theorem exact85046RawTermsValid :
    exact85046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85046 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14207⟩⟩) exact85046RawTerms (.finite 18) 85045 .exactZero (none)

def event85047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14208⟩⟩) 0 ⟨14207⟩ 85046

def event85048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14208⟩⟩) 1 ⟨11469⟩ 85043

def event85049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14208⟩⟩) (.product (.predecessor 0 85047 .coefficient) (.predecessor 1 85048 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event85050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14208⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩) [⟨.result 85046 .coefficient, true, some 1⟩, ⟨.result 85043 .coefficient, true, some 1⟩])

def event85051 : Event := .survivorFold (1) 85050

def exact85052RawTerms : List Term := []

theorem exact85052RawTermsValid :
    exact85052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85052 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14208⟩⟩) exact85052RawTerms (.finite 324) 85049 (.finite 324) (some (85050))

def event85053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14209⟩⟩) 0 ⟨14208⟩ 85052

def event85054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14209⟩⟩) (.identity (.predecessor 0 85053 .coefficient))

def event85055 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14209⟩⟩) (.finite 324)

def event85056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15940⟩⟩) 0 ⟨14209⟩ 85055

def event85057 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15940⟩⟩) (.authority (.programFamilyFact))

def exact85058RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], []⟩, (1)⟩]

theorem exact85058RawTermsValid :
    exact85058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85058 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15940⟩⟩) exact85058RawTerms (.finite 18) 85057 .exactZero (none)

def event85059 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15941⟩⟩) 0 ⟨15940⟩ 85058

def event85060 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15941⟩⟩) (.identity (.predecessor 0 85059 .coefficient))

def event85061 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15941⟩⟩) (.finite 18)

def event85062 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21400⟩⟩) 0 ⟨15941⟩ 85061

def event85063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21400⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact85064RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩, (1)⟩]

theorem exact85064RawTermsValid :
    exact85064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85064 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21400⟩⟩) exact85064RawTerms (.finite 136065468) 85063 .exactZero (none)

def event85065 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact85066RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact85066RawTermsValid :
    exact85066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85066 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact85066RawTerms .large 85065 .exactZero (none)

def event85067 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21401⟩⟩) 0 ⟨6⟩ 85066

def event85068 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21401⟩⟩) 1 ⟨21400⟩ 85064

def event85069 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21401⟩⟩) (.product (.predecessor 0 85067 .coefficient) (.predecessor 1 85068 .coefficient) (⟨false, false, none, none, none⟩))

def event85070 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21401⟩⟩, .operator (⟨85066, 0⟩, ⟨85064, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩, (1)⟩)

def exact85071RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩, (1)⟩]

theorem exact85071RawTermsValid :
    exact85071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85071 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21401⟩⟩) exact85071RawTerms .large 85069 .exactZero (none)

def event85072 : Event := .preFoldPolynomial 85071 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩, (1)⟩] .exactZero none

def exact85073RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩, (1)⟩]

def event85073 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21401⟩⟩) 85072 exact85073RawTerms .large 85069 .exactZero (none)

def event85074 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27871⟩⟩)

def event85075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event85076 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event85077 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event85078 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event85079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event85080 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event85081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event85082 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event85083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 85082

def event85084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 85080

def event85085 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 85083 .coefficient) (.value (.predecessor 1 85084 .coefficient)))

def event85086 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event85087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 85086

def event85088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 85078

def event85089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 85087 .coefficient, .predecessor 1 85088 .coefficient])

def event85090 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event85091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 85090

def event85092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 85076

def event85093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 85092 .coefficient))

def event85094 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event85095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11469⟩⟩) 0 ⟨5536⟩ 85094

def event85096 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11469⟩⟩) (.authority (.programFamilyFact))

def exact85097RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩], []⟩, (1)⟩]

theorem exact85097RawTermsValid :
    exact85097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85097 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11469⟩⟩) exact85097RawTerms (.finite 18) 85096 .exactZero (none)

def event85098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14207⟩⟩) 0 ⟨5536⟩ 85094

def event85099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14207⟩⟩) (.authority (.programFamilyFact))

def exact85100RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩]

theorem exact85100RawTermsValid :
    exact85100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85100 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14207⟩⟩) exact85100RawTerms (.finite 18) 85099 .exactZero (none)

def event85101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14208⟩⟩) 0 ⟨14207⟩ 85100

def event85102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14208⟩⟩) 1 ⟨11469⟩ 85097

def event85103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14208⟩⟩) (.product (.predecessor 0 85101 .coefficient) (.predecessor 1 85102 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event85104 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14208⟩⟩, .operator (⟨85100, 0⟩, ⟨85097, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩)

def exact85105RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩]

theorem exact85105RawTermsValid :
    exact85105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85105 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14208⟩⟩) exact85105RawTerms (.finite 324) 85103 .exactZero (none)

def event85106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14209⟩⟩) 0 ⟨14208⟩ 85105

def event85107 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14209⟩⟩) (.identity (.predecessor 0 85106 .coefficient))

def event85108 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14209⟩⟩) (.finite 324)

def event85109 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15940⟩⟩) 0 ⟨14209⟩ 85108

def event85110 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15940⟩⟩) (.authority (.programFamilyFact))

def exact85111RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], []⟩, (1)⟩]

theorem exact85111RawTermsValid :
    exact85111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85111 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15940⟩⟩) exact85111RawTerms (.finite 18) 85110 .exactZero (none)

def event85112 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15941⟩⟩) 0 ⟨15940⟩ 85111

def event85113 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15941⟩⟩) (.identity (.predecessor 0 85112 .coefficient))

def event85114 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15941⟩⟩) (.finite 18)

def event85115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24160⟩⟩) 0 ⟨15941⟩ 85114

def event85116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24160⟩⟩) (.authority (.programFamilyFact))

def event85117 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24160⟩⟩) (.finite 3720)

def event85118 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event85119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24162⟩⟩) 0 ⟨6689⟩ 85118

def event85120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24162⟩⟩) 1 ⟨24160⟩ 85117

def event85121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24162⟩⟩) (.authority (.operator))

def exact85122RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (1)⟩]

theorem exact85122RawTermsValid :
    exact85122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85122 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24162⟩⟩) exact85122RawTerms .large 85121 .exactZero (none)

def event85123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27866⟩⟩) 0 ⟨24162⟩ 85122

def event85124 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27866⟩⟩) (.authority (.operator))

def exact85125RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (1)⟩]

theorem exact85125RawTermsValid :
    exact85125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85125 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27866⟩⟩) exact85125RawTerms (.finite 8192) 85124 .exactZero (none)

def event85126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event85127 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event85128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16015⟩⟩) 0 ⟨15941⟩ 85114

def event85129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16015⟩⟩) 1 ⟨110⟩ 85127

def event85130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16015⟩⟩) (.sum [.predecessor 0 85128 .coefficient, .predecessor 1 85129 .coefficient])

def event85131 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16015⟩⟩) (.finite 18)

def event85132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16016⟩⟩) 0 ⟨16015⟩ 85131

def event85133 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16016⟩⟩) (.identity (.predecessor 0 85132 .coefficient))

def exact85134RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], []⟩, (1)⟩]

theorem exact85134RawTermsValid :
    exact85134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85134 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16016⟩⟩) exact85134RawTerms (.finite 18) 85133 .exactZero (none)

def event85135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact85136RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact85136RawTermsValid :
    exact85136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85136 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact85136RawTerms .large 85135 .exactZero (none)

def event85137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16017⟩⟩) 0 ⟨6544⟩ 85136

def event85138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16017⟩⟩) 1 ⟨16016⟩ 85134

def event85139 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16017⟩⟩) (.product (.predecessor 0 85137 .coefficient) (.predecessor 1 85138 .coefficient) (⟨false, false, none, none, none⟩))

def event85140 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16017⟩⟩, .operator (⟨85136, 0⟩, ⟨85134, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact85141RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact85141RawTermsValid :
    exact85141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85141 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16017⟩⟩) exact85141RawTerms .large 85139 .exactZero (none)

def event85142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 85118

def event85143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact85144RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact85144RawTermsValid :
    exact85144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact85144RawTerms .large 85143 .exactZero (none)

def event85145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16018⟩⟩) 0 ⟨6697⟩ 85144

def event85146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16018⟩⟩) 1 ⟨16017⟩ 85141

def event85147 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16018⟩⟩) (.sum [.predecessor 0 85145 .coefficient, .predecessor 1 85146 .coefficient])

def exact85148RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85148RawTermsValid :
    exact85148RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85148 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16018⟩⟩) exact85148RawTerms .large 85147 .exactZero (none)

def event85149 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27867⟩⟩) 0 ⟨16018⟩ 85148

def event85150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27867⟩⟩) 1 ⟨27866⟩ 85125

def event85151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27867⟩⟩) (.product (.predecessor 0 85149 .coefficient) (.predecessor 1 85150 .coefficient) (⟨false, false, none, none, none⟩))

def event85152 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27867⟩⟩, .operator (⟨85148, 0⟩, ⟨85125, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (1)⟩)

def event85153 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27867⟩⟩, .operator (⟨85148, 1⟩, ⟨85125, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (-1)⟩)

def event85154 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27867⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27866⟩⟩) ⟨24162⟩ 85122)

def event85155 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27867⟩⟩, .relation 85154 0, ⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (-1)⟩)

def exact85156RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (-1)⟩]

theorem exact85156RawTermsValid :
    exact85156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85156 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27867⟩⟩) exact85156RawTerms .large 85151 .exactZero (none)

def event85157 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15986⟩⟩) 0 ⟨15941⟩ 85114

def event85158 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15986⟩⟩) (.authority (.programFamilyFact))

def exact85159RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], []⟩, (1)⟩]

theorem exact85159RawTermsValid :
    exact85159RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85159 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15986⟩⟩) exact85159RawTerms (.finite 61) 85158 .exactZero (none)

def event85160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15987⟩⟩) 0 ⟨6544⟩ 85136

def event85161 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15987⟩⟩) 1 ⟨15986⟩ 85159

def event85162 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15987⟩⟩) (.product (.predecessor 0 85160 .coefficient) (.predecessor 1 85161 .coefficient) (⟨false, true, none, none, some 1⟩))

def event85163 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15987⟩⟩, .operator (⟨85136, 0⟩, ⟨85159, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact85164RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact85164RawTermsValid :
    exact85164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85164 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15987⟩⟩) exact85164RawTerms .large 85162 .exactZero (none)

def event85165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6723⟩⟩) 0 ⟨6689⟩ 85118

def event85166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6723⟩⟩) (.authority (.operator))

def exact85167RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩]

theorem exact85167RawTermsValid :
    exact85167RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85167 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6723⟩⟩) exact85167RawTerms .large 85166 .exactZero (none)

def event85168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15988⟩⟩) 0 ⟨6723⟩ 85167

def event85169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15988⟩⟩) 1 ⟨15987⟩ 85164

def event85170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15988⟩⟩) (.sum [.predecessor 0 85168 .coefficient, .predecessor 1 85169 .coefficient])

def exact85171RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85171RawTermsValid :
    exact85171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85171 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15988⟩⟩) exact85171RawTerms .large 85170 .exactZero (none)

def event85172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27871⟩⟩) 0 ⟨15988⟩ 85171

def event85173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27871⟩⟩) 1 ⟨27867⟩ 85156

def event85174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27871⟩⟩) (.sum [.predecessor 0 85172 .coefficient, .predecessor 1 85173 .coefficient])

def exact85175RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85175RawTermsValid :
    exact85175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85175 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27871⟩⟩) exact85175RawTerms .large 85174 .exactZero (none)

def event85176 : Event := .preFoldPolynomial 85175 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact85177RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event85177 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27871⟩⟩) 85176 exact85177RawTerms .large 85174 .exactZero (none)

def event85178 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15941⟩⟩) ⟨⟨136⟩, ⟨43⟩, ⟨109⟩⟩ ⟨85020, 85178⟩

def event85179 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21403⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩) (1) 0 2 (.universal 85178 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩) (none) 85177)

def event85180 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21403⟩⟩, .relation 85179 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩)

def event85181 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21403⟩⟩, .relation 85179 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (-1)⟩)

def event85182 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21403⟩⟩, .relation 85179 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (1)⟩)

def event85183 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21403⟩⟩, .relation 85179 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact85184RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85184RawTermsValid :
    exact85184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85184 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21403⟩⟩) exact85184RawTerms .large 85016 (.finite 1811303510016) (some (85018))

def event85185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27869⟩⟩) 0 ⟨21403⟩ 85184

def event85186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27869⟩⟩) 1 ⟨27868⟩ 85006

def event85187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27869⟩⟩) (.sum [.predecessor 0 85185 .coefficient, .predecessor 1 85186 .coefficient])

def event85188 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27869⟩⟩, .operator (⟨85184, 0⟩, ⟨85006, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩, (1)⟩)

def event85189 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27869⟩⟩, .operator (⟨85184, 2⟩, ⟨85006, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨24162⟩⟩]⟩, (-1)⟩)

def event85190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27869⟩⟩) (.sum [.result 85184 .summary, .result 85006 .summary])

def exact85191RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85191RawTermsValid :
    exact85191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85191 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27869⟩⟩) exact85191RawTerms .large 85187 (.finite 1292068473939586330624) (some (85190))

def event85192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24097⟩⟩) 0 ⟨15822⟩ 4098

def event85193 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24097⟩⟩) (.authority (.programFamilyFact))

def event85194 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24097⟩⟩) (.finite 3720)

def event85195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24099⟩⟩) 0 ⟨6689⟩ 5477

def event85196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24099⟩⟩) 1 ⟨24097⟩ 85194

def event85197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24099⟩⟩) (.authority (.operator))

def exact85198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24099⟩⟩]⟩, (1)⟩]

theorem exact85198RawTermsValid :
    exact85198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85198 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24099⟩⟩) exact85198RawTerms .large 85197 .exactZero (none)

def event85199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27649⟩⟩) 0 ⟨24099⟩ 85198

def event85200 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27649⟩⟩) (.authority (.operator))

def exact85201RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩, (1)⟩]

theorem exact85201RawTermsValid :
    exact85201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85201 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27649⟩⟩) exact85201RawTerms (.finite 8192) 85200 .exactZero (none)

def event85202 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23541⟩⟩) 0 ⟨13992⟩ 4092

def event85203 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23541⟩⟩) (.authority (.programFamilyFact))

def event85204 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23541⟩⟩) (.finite 3720)

def event85205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23542⟩⟩) 0 ⟨6689⟩ 5477

def event85206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23542⟩⟩) 1 ⟨23541⟩ 85204

def event85207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23542⟩⟩) (.authority (.operator))

def exact85208RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23542⟩⟩]⟩, (1)⟩]

theorem exact85208RawTermsValid :
    exact85208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85208 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23542⟩⟩) exact85208RawTerms .large 85207 .exactZero (none)

def event85209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25989⟩⟩) 0 ⟨23542⟩ 85208

def event85210 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25989⟩⟩) (.authority (.operator))

def exact85211RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩, (1)⟩]

theorem exact85211RawTermsValid :
    exact85211RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85211 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25989⟩⟩) exact85211RawTerms (.finite 8192) 85210 .exactZero (none)

def event85212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11386⟩⟩) 0 ⟨11385⟩ 4081

def event85213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11386⟩⟩) 1 ⟨6567⟩ 79920

def event85214 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11386⟩⟩) (.tensor (.predecessor 0 85212 .coefficient) (.predecessor 1 85213 .coefficient) true false)

def event85215 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11386⟩⟩, .operator (⟨4081, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact85216RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact85216RawTermsValid :
    exact85216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85216 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11386⟩⟩) exact85216RawTerms .large 85214 .exactZero (none)

def event85217 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7234⟩⟩) 0 ⟨5539⟩ 79790

def event85218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7234⟩⟩) 1 ⟨6778⟩ 11983

def event85219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7234⟩⟩) (.product (.predecessor 0 85217 .coefficient) (.predecessor 1 85218 .coefficient) (⟨false, false, none, none, none⟩))

def event85220 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7234⟩⟩, .operator (⟨79790, 0⟩, ⟨11983, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩)

def exact85221RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact85221RawTermsValid :
    exact85221RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85221 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7234⟩⟩) exact85221RawTerms .large 85219 .exactZero (none)

def event85222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11387⟩⟩) 0 ⟨7234⟩ 85221

def event85223 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11387⟩⟩) 1 ⟨11386⟩ 85216

def event85224 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11387⟩⟩) (.sum [.predecessor 0 85222 .coefficient, .predecessor 1 85223 .coefficient])

def exact85225RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85225RawTermsValid :
    exact85225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85225 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11387⟩⟩) exact85225RawTerms .large 85224 .exactZero (none)

def event85226 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11388⟩⟩) 0 ⟨11387⟩ 85225

def event85227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11388⟩⟩) 1 ⟨92⟩ 11975

def event85228 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11388⟩⟩) (.sum [.predecessor 0 85226 .coefficient, .predecessor 1 85227 .coefficient])

def event85229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11388⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩) [⟨.result 11975 .coefficient, false, none⟩])

def event85230 : Event := .survivorFold (1) 85229

def exact85231RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85231RawTermsValid :
    exact85231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85231 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11388⟩⟩) exact85231RawTerms .large 85228 (.finite 26) (some (85229))

def event85232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13993⟩⟩) 0 ⟨11388⟩ 85231

def event85233 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13993⟩⟩) 1 ⟨13990⟩ 4084

def event85234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13993⟩⟩) (.product (.predecessor 0 85232 .coefficient) (.predecessor 1 85233 .coefficient) (⟨false, true, none, none, some 1⟩))

def event85235 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13993⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨13990⟩⟩], []⟩) [⟨.result 4084 .coefficient, true, some 1⟩])

def event85236 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13993⟩⟩) (.product (.result 85231 .summary) (.transfer 85235) (⟨false, false, none, none, none⟩))

def event85237 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13993⟩⟩, .operator (⟨85231, 1⟩, ⟨4084, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event85238 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13993⟩⟩, .operator (⟨85231, 0⟩, ⟨4084, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩)

def exact85239RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact85239RawTermsValid :
    exact85239RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85239 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13993⟩⟩) exact85239RawTerms .large 85234 (.finite 13312) (some (85236))

def event85240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13994⟩⟩) 0 ⟨13990⟩ 4084

def event85241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13994⟩⟩) 1 ⟨6567⟩ 79920

def event85242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13994⟩⟩) (.tensor (.predecessor 0 85240 .coefficient) (.predecessor 1 85241 .coefficient) true false)

def event85243 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13994⟩⟩, .operator (⟨4084, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact85244RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact85244RawTermsValid :
    exact85244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85244 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13994⟩⟩) exact85244RawTerms .large 85242 .exactZero (none)

def event85245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7214⟩⟩) 0 ⟨5539⟩ 79790

def event85246 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7214⟩⟩) 1 ⟨6758⟩ 12024

def event85247 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7214⟩⟩) (.product (.predecessor 0 85245 .coefficient) (.predecessor 1 85246 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf5312 : Array AnnotatedEvent := #[
  { event := event84992
    frameStart := 0 },
  { event := event84993
    frameStart := 0 },
  { event := event84994
    frameStart := 0 },
  { event := event84995
    frameStart := 0 },
  { event := event84996
    frameStart := 0 },
  { event := event84997
    frameStart := 0 },
  { event := event84998
    frameStart := 0 },
  { event := event84999
    frameStart := 0 },
  { event := event85000
    frameStart := 0 },
  { event := event85001
    frameStart := 0 },
  { event := event85002
    frameStart := 0 },
  { event := event85003
    frameStart := 0 },
  { event := event85004
    frameStart := 0 },
  { event := event85005
    frameStart := 0 },
  { event := event85006
    frameStart := 0 },
  { event := event85007
    frameStart := 0 }
]

def eventLeaf5313 : Array AnnotatedEvent := #[
  { event := event85008
    frameStart := 0 },
  { event := event85009
    frameStart := 0 },
  { event := event85010
    frameStart := 0 },
  { event := event85011
    frameStart := 0 },
  { event := event85012
    frameStart := 0 },
  { event := event85013
    frameStart := 0 },
  { event := event85014
    frameStart := 0 },
  { event := event85015
    frameStart := 0 },
  { event := event85016
    frameStart := 0 },
  { event := event85017
    frameStart := 0 },
  { event := event85018
    frameStart := 0 },
  { event := event85019
    frameStart := 0 },
  { event := event85020
    frameStart := 85020 },
  { event := event85021
    frameStart := 85020 },
  { event := event85022
    frameStart := 85020 },
  { event := event85023
    frameStart := 85020 }
]

def eventLeaf5314 : Array AnnotatedEvent := #[
  { event := event85024
    frameStart := 85020 },
  { event := event85025
    frameStart := 85020 },
  { event := event85026
    frameStart := 85020 },
  { event := event85027
    frameStart := 85020 },
  { event := event85028
    frameStart := 85020 },
  { event := event85029
    frameStart := 85020 },
  { event := event85030
    frameStart := 85020 },
  { event := event85031
    frameStart := 85020 },
  { event := event85032
    frameStart := 85020 },
  { event := event85033
    frameStart := 85020 },
  { event := event85034
    frameStart := 85020 },
  { event := event85035
    frameStart := 85020 },
  { event := event85036
    frameStart := 85020 },
  { event := event85037
    frameStart := 85020 },
  { event := event85038
    frameStart := 85020 },
  { event := event85039
    frameStart := 85020 }
]

def eventLeaf5315 : Array AnnotatedEvent := #[
  { event := event85040
    frameStart := 85020 },
  { event := event85041
    frameStart := 85020 },
  { event := event85042
    frameStart := 85020 },
  { event := event85043
    frameStart := 85020 },
  { event := event85044
    frameStart := 85020 },
  { event := event85045
    frameStart := 85020 },
  { event := event85046
    frameStart := 85020 },
  { event := event85047
    frameStart := 85020 },
  { event := event85048
    frameStart := 85020 },
  { event := event85049
    frameStart := 85020 },
  { event := event85050
    frameStart := 85020 },
  { event := event85051
    frameStart := 85020 },
  { event := event85052
    frameStart := 85020 },
  { event := event85053
    frameStart := 85020 },
  { event := event85054
    frameStart := 85020 },
  { event := event85055
    frameStart := 85020 }
]

def eventLeaf5316 : Array AnnotatedEvent := #[
  { event := event85056
    frameStart := 85020 },
  { event := event85057
    frameStart := 85020 },
  { event := event85058
    frameStart := 85020 },
  { event := event85059
    frameStart := 85020 },
  { event := event85060
    frameStart := 85020 },
  { event := event85061
    frameStart := 85020 },
  { event := event85062
    frameStart := 85020 },
  { event := event85063
    frameStart := 85020 },
  { event := event85064
    frameStart := 85020 },
  { event := event85065
    frameStart := 85020 },
  { event := event85066
    frameStart := 85020 },
  { event := event85067
    frameStart := 85020 },
  { event := event85068
    frameStart := 85020 },
  { event := event85069
    frameStart := 85020 },
  { event := event85070
    frameStart := 85020 },
  { event := event85071
    frameStart := 85020 }
]

def eventLeaf5317 : Array AnnotatedEvent := #[
  { event := event85072
    frameStart := 85020 },
  { event := event85073
    frameStart := 85020 },
  { event := event85074
    frameStart := 85074 },
  { event := event85075
    frameStart := 85074 },
  { event := event85076
    frameStart := 85074 },
  { event := event85077
    frameStart := 85074 },
  { event := event85078
    frameStart := 85074 },
  { event := event85079
    frameStart := 85074 },
  { event := event85080
    frameStart := 85074 },
  { event := event85081
    frameStart := 85074 },
  { event := event85082
    frameStart := 85074 },
  { event := event85083
    frameStart := 85074 },
  { event := event85084
    frameStart := 85074 },
  { event := event85085
    frameStart := 85074 },
  { event := event85086
    frameStart := 85074 },
  { event := event85087
    frameStart := 85074 }
]

def eventLeaf5318 : Array AnnotatedEvent := #[
  { event := event85088
    frameStart := 85074 },
  { event := event85089
    frameStart := 85074 },
  { event := event85090
    frameStart := 85074 },
  { event := event85091
    frameStart := 85074 },
  { event := event85092
    frameStart := 85074 },
  { event := event85093
    frameStart := 85074 },
  { event := event85094
    frameStart := 85074 },
  { event := event85095
    frameStart := 85074 },
  { event := event85096
    frameStart := 85074 },
  { event := event85097
    frameStart := 85074 },
  { event := event85098
    frameStart := 85074 },
  { event := event85099
    frameStart := 85074 },
  { event := event85100
    frameStart := 85074 },
  { event := event85101
    frameStart := 85074 },
  { event := event85102
    frameStart := 85074 },
  { event := event85103
    frameStart := 85074 }
]

def eventLeaf5319 : Array AnnotatedEvent := #[
  { event := event85104
    frameStart := 85074 },
  { event := event85105
    frameStart := 85074 },
  { event := event85106
    frameStart := 85074 },
  { event := event85107
    frameStart := 85074 },
  { event := event85108
    frameStart := 85074 },
  { event := event85109
    frameStart := 85074 },
  { event := event85110
    frameStart := 85074 },
  { event := event85111
    frameStart := 85074 },
  { event := event85112
    frameStart := 85074 },
  { event := event85113
    frameStart := 85074 },
  { event := event85114
    frameStart := 85074 },
  { event := event85115
    frameStart := 85074 },
  { event := event85116
    frameStart := 85074 },
  { event := event85117
    frameStart := 85074 },
  { event := event85118
    frameStart := 85074 },
  { event := event85119
    frameStart := 85074 }
]

def eventLeaf5320 : Array AnnotatedEvent := #[
  { event := event85120
    frameStart := 85074 },
  { event := event85121
    frameStart := 85074 },
  { event := event85122
    frameStart := 85074 },
  { event := event85123
    frameStart := 85074 },
  { event := event85124
    frameStart := 85074 },
  { event := event85125
    frameStart := 85074 },
  { event := event85126
    frameStart := 85074 },
  { event := event85127
    frameStart := 85074 },
  { event := event85128
    frameStart := 85074 },
  { event := event85129
    frameStart := 85074 },
  { event := event85130
    frameStart := 85074 },
  { event := event85131
    frameStart := 85074 },
  { event := event85132
    frameStart := 85074 },
  { event := event85133
    frameStart := 85074 },
  { event := event85134
    frameStart := 85074 },
  { event := event85135
    frameStart := 85074 }
]

def eventLeaf5321 : Array AnnotatedEvent := #[
  { event := event85136
    frameStart := 85074 },
  { event := event85137
    frameStart := 85074 },
  { event := event85138
    frameStart := 85074 },
  { event := event85139
    frameStart := 85074 },
  { event := event85140
    frameStart := 85074 },
  { event := event85141
    frameStart := 85074 },
  { event := event85142
    frameStart := 85074 },
  { event := event85143
    frameStart := 85074 },
  { event := event85144
    frameStart := 85074 },
  { event := event85145
    frameStart := 85074 },
  { event := event85146
    frameStart := 85074 },
  { event := event85147
    frameStart := 85074 },
  { event := event85148
    frameStart := 85074 },
  { event := event85149
    frameStart := 85074 },
  { event := event85150
    frameStart := 85074 },
  { event := event85151
    frameStart := 85074 }
]

def eventLeaf5322 : Array AnnotatedEvent := #[
  { event := event85152
    frameStart := 85074 },
  { event := event85153
    frameStart := 85074 },
  { event := event85154
    frameStart := 85074 },
  { event := event85155
    frameStart := 85074 },
  { event := event85156
    frameStart := 85074 },
  { event := event85157
    frameStart := 85074 },
  { event := event85158
    frameStart := 85074 },
  { event := event85159
    frameStart := 85074 },
  { event := event85160
    frameStart := 85074 },
  { event := event85161
    frameStart := 85074 },
  { event := event85162
    frameStart := 85074 },
  { event := event85163
    frameStart := 85074 },
  { event := event85164
    frameStart := 85074 },
  { event := event85165
    frameStart := 85074 },
  { event := event85166
    frameStart := 85074 },
  { event := event85167
    frameStart := 85074 }
]

def eventLeaf5323 : Array AnnotatedEvent := #[
  { event := event85168
    frameStart := 85074 },
  { event := event85169
    frameStart := 85074 },
  { event := event85170
    frameStart := 85074 },
  { event := event85171
    frameStart := 85074 },
  { event := event85172
    frameStart := 85074 },
  { event := event85173
    frameStart := 85074 },
  { event := event85174
    frameStart := 85074 },
  { event := event85175
    frameStart := 85074 },
  { event := event85176
    frameStart := 85074 },
  { event := event85177
    frameStart := 85074 },
  { event := event85178
    frameStart := 0 },
  { event := event85179
    frameStart := 0 },
  { event := event85180
    frameStart := 0 },
  { event := event85181
    frameStart := 0 },
  { event := event85182
    frameStart := 0 },
  { event := event85183
    frameStart := 0 }
]

def eventLeaf5324 : Array AnnotatedEvent := #[
  { event := event85184
    frameStart := 0 },
  { event := event85185
    frameStart := 0 },
  { event := event85186
    frameStart := 0 },
  { event := event85187
    frameStart := 0 },
  { event := event85188
    frameStart := 0 },
  { event := event85189
    frameStart := 0 },
  { event := event85190
    frameStart := 0 },
  { event := event85191
    frameStart := 0 },
  { event := event85192
    frameStart := 0 },
  { event := event85193
    frameStart := 0 },
  { event := event85194
    frameStart := 0 },
  { event := event85195
    frameStart := 0 },
  { event := event85196
    frameStart := 0 },
  { event := event85197
    frameStart := 0 },
  { event := event85198
    frameStart := 0 },
  { event := event85199
    frameStart := 0 }
]

def eventLeaf5325 : Array AnnotatedEvent := #[
  { event := event85200
    frameStart := 0 },
  { event := event85201
    frameStart := 0 },
  { event := event85202
    frameStart := 0 },
  { event := event85203
    frameStart := 0 },
  { event := event85204
    frameStart := 0 },
  { event := event85205
    frameStart := 0 },
  { event := event85206
    frameStart := 0 },
  { event := event85207
    frameStart := 0 },
  { event := event85208
    frameStart := 0 },
  { event := event85209
    frameStart := 0 },
  { event := event85210
    frameStart := 0 },
  { event := event85211
    frameStart := 0 },
  { event := event85212
    frameStart := 0 },
  { event := event85213
    frameStart := 0 },
  { event := event85214
    frameStart := 0 },
  { event := event85215
    frameStart := 0 }
]

def eventLeaf5326 : Array AnnotatedEvent := #[
  { event := event85216
    frameStart := 0 },
  { event := event85217
    frameStart := 0 },
  { event := event85218
    frameStart := 0 },
  { event := event85219
    frameStart := 0 },
  { event := event85220
    frameStart := 0 },
  { event := event85221
    frameStart := 0 },
  { event := event85222
    frameStart := 0 },
  { event := event85223
    frameStart := 0 },
  { event := event85224
    frameStart := 0 },
  { event := event85225
    frameStart := 0 },
  { event := event85226
    frameStart := 0 },
  { event := event85227
    frameStart := 0 },
  { event := event85228
    frameStart := 0 },
  { event := event85229
    frameStart := 0 },
  { event := event85230
    frameStart := 0 },
  { event := event85231
    frameStart := 0 }
]

def eventLeaf5327 : Array AnnotatedEvent := #[
  { event := event85232
    frameStart := 0 },
  { event := event85233
    frameStart := 0 },
  { event := event85234
    frameStart := 0 },
  { event := event85235
    frameStart := 0 },
  { event := event85236
    frameStart := 0 },
  { event := event85237
    frameStart := 0 },
  { event := event85238
    frameStart := 0 },
  { event := event85239
    frameStart := 0 },
  { event := event85240
    frameStart := 0 },
  { event := event85241
    frameStart := 0 },
  { event := event85242
    frameStart := 0 },
  { event := event85243
    frameStart := 0 },
  { event := event85244
    frameStart := 0 },
  { event := event85245
    frameStart := 0 },
  { event := event85246
    frameStart := 0 },
  { event := event85247
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events332
