import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events176

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event45056 : Event := .preFoldPolynomial 45055 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨55221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact45057RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨55221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event45057 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨56210⟩⟩) 45056 exact45057RawTerms .large 45054 .exactZero (none)

def event45058 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53941⟩⟩) ⟨⟨86⟩, ⟨67⟩, ⟨135⟩⟩ ⟨44900, 45058⟩

def event45059 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54915⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54912⟩⟩]⟩) (1) 0 2 (.universal 45058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54912⟩⟩]⟩) (none) 45057)

def event45060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54915⟩⟩, .relation 45059 1, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩, (1)⟩)

def event45061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54915⟩⟩, .relation 45059 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩]⟩, (-1)⟩)

def event45062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54915⟩⟩, .relation 45059 2, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨55221⟩⟩]⟩, (1)⟩)

def event45063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54915⟩⟩, .relation 45059 3, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact45064RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨55221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact45064RawTermsValid :
    exact45064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45064 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54915⟩⟩) exact45064RawTerms .large 44896 (.finite 202072841853861888) (some (44898))

def event45065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56207⟩⟩) 0 ⟨54915⟩ 45064

def event45066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56207⟩⟩) 1 ⟨56206⟩ 44886

def event45067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56207⟩⟩) (.sum [.predecessor 0 45065 .coefficient, .predecessor 1 45066 .coefficient])

def event45068 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56207⟩⟩, .operator (⟨45064, 0⟩, ⟨44886, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩]⟩, (1)⟩)

def event45069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56207⟩⟩, .operator (⟨45064, 2⟩, ⟨44886, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨55221⟩⟩]⟩, (-1)⟩)

def event45070 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56207⟩⟩) (.sum [.result 45064 .summary, .result 44886 .summary])

def exact45071RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact45071RawTermsValid :
    exact45071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45071 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56207⟩⟩) exact45071RawTerms .large 45067 (.finite 32189789464712143775715074244608) (some (45070))

def event45072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56208⟩⟩) 0 ⟨56207⟩ 45071

def event45073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56208⟩⟩) 1 ⟨7126⟩ 15782

def event45074 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56208⟩⟩) (.product (.predecessor 0 45072 .coefficient) (.predecessor 1 45073 .coefficient) (⟨false, false, none, none, none⟩))

def event45075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56208⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) [⟨.result 15778 .coefficient, false, none⟩])

def event45076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56208⟩⟩) (.product (.result 45071 .summary) (.transfer 45075) (⟨false, false, none, none, none⟩))

def event45077 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56208⟩⟩, .operator (⟨45071, 0⟩, ⟨15782, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩)

def event45078 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56208⟩⟩, .operator (⟨45071, 1⟩, ⟨15782, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩)

def event45079 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨56208⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7125⟩⟩) ⟨7028⟩ 15775)

def event45080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56208⟩⟩, .relation 45079 0, ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact45081RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩]

theorem exact45081RawTermsValid :
    exact45081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56208⟩⟩) exact45081RawTerms .large 45074 (.finite 345635232540160008926865507237008160849920) (some (45076))

def event45082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52241⟩⟩) 0 ⟨7177⟩ 15500

def event45083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52241⟩⟩) 1 ⟨52240⟩ 38288

def event45084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52241⟩⟩) (.authority (.operator))

def exact45085RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (1)⟩]

theorem exact45085RawTermsValid :
    exact45085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52241⟩⟩) exact45085RawTerms .large 45084 .exactZero (none)

def event45086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53224⟩⟩) 0 ⟨52241⟩ 45085

def event45087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53224⟩⟩) (.authority (.operator))

def exact45088RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (1)⟩]

theorem exact45088RawTermsValid :
    exact45088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53224⟩⟩) exact45088RawTerms (.finite 8192) 45087 .exactZero (none)

def event45089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53226⟩⟩) 0 ⟨52620⟩ 38572

def event45090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53226⟩⟩) 1 ⟨53224⟩ 45088

def event45091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53226⟩⟩) (.product (.predecessor 0 45089 .coefficient) (.predecessor 1 45090 .coefficient) (⟨false, false, none, none, none⟩))

def event45092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53226⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩) [⟨.result 45088 .coefficient, false, none⟩])

def event45093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53226⟩⟩) (.product (.result 38572 .summary) (.transfer 45092) (⟨false, false, none, none, none⟩))

def event45094 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53226⟩⟩, .operator (⟨38572, 0⟩, ⟨45088, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (1)⟩)

def event45095 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53226⟩⟩, .operator (⟨38572, 1⟩, ⟨45088, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (-1)⟩)

def event45096 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53226⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53224⟩⟩) ⟨52241⟩ 45085)

def event45097 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53226⟩⟩, .relation 45096 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (-1)⟩)

def exact45098RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (-1)⟩]

theorem exact45098RawTermsValid :
    exact45098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53226⟩⟩) exact45098RawTerms .large 45091 (.finite 32189593014266254325632330629120) (some (45093))

def event45099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51932⟩⟩) 0 ⟨50961⟩ 1158

def event45100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51932⟩⟩) (.authority (.relationPreimageSource ⟨64⟩))

def exact45101RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩, (1)⟩]

theorem exact45101RawTermsValid :
    exact45101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45101 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51932⟩⟩) exact45101RawTerms (.finite 5647228698) 45100 .exactZero (none)

def event45102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51934⟩⟩) 0 ⟨51932⟩ 45101

def event45103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51934⟩⟩) 1 ⟨2370⟩ 4

def event45104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51934⟩⟩) (.scale (.predecessor 0 45102 .coefficient) (.value (.predecessor 1 45103 .coefficient)))

def exact45105RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩, (1)⟩]

theorem exact45105RawTermsValid :
    exact45105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45105 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51934⟩⟩) exact45105RawTerms (.finite 5647228698) 45104 .exactZero (none)

def event45106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51935⟩⟩) 0 ⟨11643⟩ 32120

def event45107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51935⟩⟩) 1 ⟨51934⟩ 45105

def event45108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51935⟩⟩) (.product (.predecessor 0 45106 .coefficient) (.predecessor 1 45107 .coefficient) (⟨false, false, none, none, none⟩))

def event45109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51935⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩) [⟨.result 45101 .coefficient, false, none⟩])

def event45110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51935⟩⟩) (.product (.result 32120 .summary) (.transfer 45109) (⟨false, false, none, none, none⟩))

def event45111 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51935⟩⟩, .operator (⟨32120, 0⟩, ⟨45105, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩, (1)⟩)

def event45112 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51933⟩⟩)

def event45113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event45114 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event45115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event45116 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event45117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event45118 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event45119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event45120 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event45121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 45120

def event45122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 45118

def event45123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 45121 .coefficient) (.value (.predecessor 1 45122 .coefficient)))

def event45124 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event45125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 45124

def event45126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 45116

def event45127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 45125 .coefficient, .predecessor 1 45126 .coefficient])

def event45128 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event45129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 45128

def event45130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 45114

def event45131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 45130 .coefficient))

def event45132 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event45133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24638⟩⟩) 0 ⟨11600⟩ 45132

def event45134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24638⟩⟩) (.authority (.programFamilyFact))

def exact45135RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24638⟩⟩], []⟩, (1)⟩]

theorem exact45135RawTermsValid :
    exact45135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45135 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24638⟩⟩) exact45135RawTerms (.finite 10) 45134 .exactZero (none)

def event45136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50788⟩⟩) 0 ⟨11600⟩ 45132

def event45137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50788⟩⟩) (.authority (.programFamilyFact))

def exact45138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50788⟩⟩], []⟩, (1)⟩]

theorem exact45138RawTermsValid :
    exact45138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50788⟩⟩) exact45138RawTerms (.finite 10) 45137 .exactZero (none)

def event45139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50789⟩⟩) 0 ⟨50788⟩ 45138

def event45140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50789⟩⟩) 1 ⟨24638⟩ 45135

def event45141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50789⟩⟩) (.product (.predecessor 0 45139 .coefficient) (.predecessor 1 45140 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event45142 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50789⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], []⟩) [⟨.result 45138 .coefficient, true, some 1⟩, ⟨.result 45135 .coefficient, true, some 1⟩])

def event45143 : Event := .survivorFold (1) 45142

def exact45144RawTerms : List Term := []

theorem exact45144RawTermsValid :
    exact45144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45144 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50789⟩⟩) exact45144RawTerms (.finite 100) 45141 (.finite 100) (some (45142))

def event45145 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50790⟩⟩) 0 ⟨50789⟩ 45144

def event45146 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50790⟩⟩) (.identity (.predecessor 0 45145 .coefficient))

def event45147 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50790⟩⟩) (.finite 100)

def event45148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50960⟩⟩) 0 ⟨50790⟩ 45147

def event45149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50960⟩⟩) (.authority (.programFamilyFact))

def exact45150RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], []⟩, (1)⟩]

theorem exact45150RawTermsValid :
    exact45150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45150 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50960⟩⟩) exact45150RawTerms (.finite 10) 45149 .exactZero (none)

def event45151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50961⟩⟩) 0 ⟨50960⟩ 45150

def event45152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50961⟩⟩) (.identity (.predecessor 0 45151 .coefficient))

def event45153 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50961⟩⟩) (.finite 10)

def event45154 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51932⟩⟩) 0 ⟨50961⟩ 45153

def event45155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51932⟩⟩) (.authority (.relationPreimageSource ⟨64⟩))

def exact45156RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩, (1)⟩]

theorem exact45156RawTermsValid :
    exact45156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45156 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51932⟩⟩) exact45156RawTerms (.finite 5647228698) 45155 .exactZero (none)

def event45157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact45158RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact45158RawTermsValid :
    exact45158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45158 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact45158RawTerms .large 45157 .exactZero (none)

def event45159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51933⟩⟩) 0 ⟨35⟩ 45158

def event45160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51933⟩⟩) 1 ⟨51932⟩ 45156

def event45161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51933⟩⟩) (.product (.predecessor 0 45159 .coefficient) (.predecessor 1 45160 .coefficient) (⟨false, false, none, none, none⟩))

def event45162 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51933⟩⟩, .operator (⟨45158, 0⟩, ⟨45156, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩, (1)⟩)

def exact45163RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩, (1)⟩]

theorem exact45163RawTermsValid :
    exact45163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51933⟩⟩) exact45163RawTerms .large 45161 .exactZero (none)

def event45164 : Event := .preFoldPolynomial 45163 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩, (1)⟩] .exactZero none

def exact45165RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩, (1)⟩]

def event45165 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51933⟩⟩) 45164 exact45165RawTerms .large 45161 .exactZero (none)

def event45166 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨53230⟩⟩)

def event45167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event45168 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event45169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event45170 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event45171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event45172 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event45173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event45174 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event45175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 45174

def event45176 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 45172

def event45177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 45175 .coefficient) (.value (.predecessor 1 45176 .coefficient)))

def event45178 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event45179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 45178

def event45180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 45170

def event45181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 45179 .coefficient, .predecessor 1 45180 .coefficient])

def event45182 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event45183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 45182

def event45184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 45168

def event45185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 45184 .coefficient))

def event45186 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event45187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24638⟩⟩) 0 ⟨11600⟩ 45186

def event45188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24638⟩⟩) (.authority (.programFamilyFact))

def exact45189RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24638⟩⟩], []⟩, (1)⟩]

theorem exact45189RawTermsValid :
    exact45189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45189 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24638⟩⟩) exact45189RawTerms (.finite 10) 45188 .exactZero (none)

def event45190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50788⟩⟩) 0 ⟨11600⟩ 45186

def event45191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50788⟩⟩) (.authority (.programFamilyFact))

def exact45192RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50788⟩⟩], []⟩, (1)⟩]

theorem exact45192RawTermsValid :
    exact45192RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45192 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50788⟩⟩) exact45192RawTerms (.finite 10) 45191 .exactZero (none)

def event45193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50789⟩⟩) 0 ⟨50788⟩ 45192

def event45194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50789⟩⟩) 1 ⟨24638⟩ 45189

def event45195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50789⟩⟩) (.product (.predecessor 0 45193 .coefficient) (.predecessor 1 45194 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event45196 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50789⟩⟩, .operator (⟨45192, 0⟩, ⟨45189, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], []⟩, (1)⟩)

def exact45197RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24638⟩⟩, ⟨.program ⟨257⟩, ⟨50788⟩⟩], []⟩, (1)⟩]

theorem exact45197RawTermsValid :
    exact45197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45197 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50789⟩⟩) exact45197RawTerms (.finite 100) 45195 .exactZero (none)

def event45198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50790⟩⟩) 0 ⟨50789⟩ 45197

def event45199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50790⟩⟩) (.identity (.predecessor 0 45198 .coefficient))

def event45200 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50790⟩⟩) (.finite 100)

def event45201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50960⟩⟩) 0 ⟨50790⟩ 45200

def event45202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50960⟩⟩) (.authority (.programFamilyFact))

def exact45203RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], []⟩, (1)⟩]

theorem exact45203RawTermsValid :
    exact45203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45203 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50960⟩⟩) exact45203RawTerms (.finite 10) 45202 .exactZero (none)

def event45204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50961⟩⟩) 0 ⟨50960⟩ 45203

def event45205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50961⟩⟩) (.identity (.predecessor 0 45204 .coefficient))

def event45206 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50961⟩⟩) (.finite 10)

def event45207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52240⟩⟩) 0 ⟨50961⟩ 45206

def event45208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52240⟩⟩) (.authority (.programFamilyFact))

def event45209 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52240⟩⟩) (.finite 3720)

def event45210 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event45211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52241⟩⟩) 0 ⟨7177⟩ 45210

def event45212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52241⟩⟩) 1 ⟨52240⟩ 45209

def event45213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52241⟩⟩) (.authority (.operator))

def exact45214RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (1)⟩]

theorem exact45214RawTermsValid :
    exact45214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45214 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52241⟩⟩) exact45214RawTerms .large 45213 .exactZero (none)

def event45215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53224⟩⟩) 0 ⟨52241⟩ 45214

def event45216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53224⟩⟩) (.authority (.operator))

def exact45217RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (1)⟩]

theorem exact45217RawTermsValid :
    exact45217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45217 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53224⟩⟩) exact45217RawTerms (.finite 8192) 45216 .exactZero (none)

def event45218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event45219 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event45220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52402⟩⟩) 0 ⟨50961⟩ 45206

def event45221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52402⟩⟩) 1 ⟨136⟩ 45219

def event45222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52402⟩⟩) (.sum [.predecessor 0 45220 .coefficient, .predecessor 1 45221 .coefficient])

def event45223 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52402⟩⟩) (.finite 10)

def event45224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52403⟩⟩) 0 ⟨52402⟩ 45223

def event45225 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52403⟩⟩) (.identity (.predecessor 0 45224 .coefficient))

def exact45226RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], []⟩, (1)⟩]

theorem exact45226RawTermsValid :
    exact45226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45226 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52403⟩⟩) exact45226RawTerms (.finite 10) 45225 .exactZero (none)

def event45227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact45228RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact45228RawTermsValid :
    exact45228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45228 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact45228RawTerms .large 45227 .exactZero (none)

def event45229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52404⟩⟩) 0 ⟨6908⟩ 45228

def event45230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52404⟩⟩) 1 ⟨52403⟩ 45226

def event45231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52404⟩⟩) (.product (.predecessor 0 45229 .coefficient) (.predecessor 1 45230 .coefficient) (⟨false, false, none, none, none⟩))

def event45232 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52404⟩⟩, .operator (⟨45228, 0⟩, ⟨45226, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact45233RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact45233RawTermsValid :
    exact45233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45233 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52404⟩⟩) exact45233RawTerms .large 45231 .exactZero (none)

def event45234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 45210

def event45235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact45236RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact45236RawTermsValid :
    exact45236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact45236RawTerms .large 45235 .exactZero (none)

def event45237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52405⟩⟩) 0 ⟨7183⟩ 45236

def event45238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52405⟩⟩) 1 ⟨52404⟩ 45233

def event45239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52405⟩⟩) (.sum [.predecessor 0 45237 .coefficient, .predecessor 1 45238 .coefficient])

def exact45240RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact45240RawTermsValid :
    exact45240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45240 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52405⟩⟩) exact45240RawTerms .large 45239 .exactZero (none)

def event45241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53225⟩⟩) 0 ⟨52405⟩ 45240

def event45242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53225⟩⟩) 1 ⟨53224⟩ 45217

def event45243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53225⟩⟩) (.product (.predecessor 0 45241 .coefficient) (.predecessor 1 45242 .coefficient) (⟨false, false, none, none, none⟩))

def event45244 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53225⟩⟩, .operator (⟨45240, 0⟩, ⟨45217, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (1)⟩)

def event45245 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53225⟩⟩, .operator (⟨45240, 1⟩, ⟨45217, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (-1)⟩)

def event45246 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53225⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53224⟩⟩) ⟨52241⟩ 45214)

def event45247 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53225⟩⟩, .relation 45246 0, ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (-1)⟩)

def exact45248RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (-1)⟩]

theorem exact45248RawTermsValid :
    exact45248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53225⟩⟩) exact45248RawTerms .large 45243 .exactZero (none)

def event45249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51336⟩⟩) 0 ⟨50961⟩ 45206

def event45250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51336⟩⟩) (.authority (.programFamilyFact))

def exact45251RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51336⟩⟩], []⟩, (1)⟩]

theorem exact45251RawTermsValid :
    exact45251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45251 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51336⟩⟩) exact45251RawTerms (.finite 10) 45250 .exactZero (none)

def event45252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51339⟩⟩) 0 ⟨6908⟩ 45228

def event45253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51339⟩⟩) 1 ⟨51336⟩ 45251

def event45254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51339⟩⟩) (.product (.predecessor 0 45252 .coefficient) (.predecessor 1 45253 .coefficient) (⟨false, true, none, none, some 1⟩))

def event45255 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51339⟩⟩, .operator (⟨45228, 0⟩, ⟨45251, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact45256RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact45256RawTermsValid :
    exact45256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45256 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51339⟩⟩) exact45256RawTerms .large 45254 .exactZero (none)

def event45257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7205⟩⟩) 0 ⟨7177⟩ 45210

def event45258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7205⟩⟩) (.authority (.operator))

def exact45259RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩, (1)⟩]

theorem exact45259RawTermsValid :
    exact45259RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45259 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7205⟩⟩) exact45259RawTerms .large 45258 .exactZero (none)

def event45260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51340⟩⟩) 0 ⟨7205⟩ 45259

def event45261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51340⟩⟩) 1 ⟨51339⟩ 45256

def event45262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51340⟩⟩) (.sum [.predecessor 0 45260 .coefficient, .predecessor 1 45261 .coefficient])

def exact45263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact45263RawTermsValid :
    exact45263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51340⟩⟩) exact45263RawTerms .large 45262 .exactZero (none)

def event45264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53230⟩⟩) 0 ⟨51340⟩ 45263

def event45265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53230⟩⟩) 1 ⟨53225⟩ 45248

def event45266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53230⟩⟩) (.sum [.predecessor 0 45264 .coefficient, .predecessor 1 45265 .coefficient])

def exact45267RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact45267RawTermsValid :
    exact45267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45267 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53230⟩⟩) exact45267RawTerms .large 45266 .exactZero (none)

def event45268 : Event := .preFoldPolynomial 45267 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact45269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event45269 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨53230⟩⟩) 45268 exact45269RawTerms .large 45266 .exactZero (none)

def event45270 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50961⟩⟩) ⟨⟨84⟩, ⟨64⟩, ⟨135⟩⟩ ⟨45112, 45270⟩

def event45271 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51935⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩) (1) 0 2 (.universal 45270 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51932⟩⟩]⟩) (none) 45269)

def event45272 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51935⟩⟩, .relation 45271 1, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩, (1)⟩)

def event45273 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51935⟩⟩, .relation 45271 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (-1)⟩)

def event45274 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51935⟩⟩, .relation 45271 2, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (1)⟩)

def event45275 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51935⟩⟩, .relation 45271 3, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact45276RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact45276RawTermsValid :
    exact45276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45276 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51935⟩⟩) exact45276RawTerms .large 45108 (.finite 202072841853861888) (some (45110))

def event45277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53227⟩⟩) 0 ⟨51935⟩ 45276

def event45278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53227⟩⟩) 1 ⟨53226⟩ 45098

def event45279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53227⟩⟩) (.sum [.predecessor 0 45277 .coefficient, .predecessor 1 45278 .coefficient])

def event45280 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53227⟩⟩, .operator (⟨45276, 0⟩, ⟨45098, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩, (1)⟩)

def event45281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53227⟩⟩, .operator (⟨45276, 2⟩, ⟨45098, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨52241⟩⟩]⟩, (-1)⟩)

def event45282 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53227⟩⟩) (.sum [.result 45276 .summary, .result 45098 .summary])

def exact45283RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact45283RawTermsValid :
    exact45283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45283 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53227⟩⟩) exact45283RawTerms .large 45279 (.finite 32189593014266456398474184491008) (some (45282))

def event45284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53228⟩⟩) 0 ⟨53227⟩ 45283

def event45285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53228⟩⟩) 1 ⟨7132⟩ 15802

def event45286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53228⟩⟩) (.product (.predecessor 0 45284 .coefficient) (.predecessor 1 45285 .coefficient) (⟨false, false, none, none, none⟩))

def event45287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53228⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) [⟨.result 15798 .coefficient, false, none⟩])

def event45288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53228⟩⟩) (.product (.result 45283 .summary) (.transfer 45287) (⟨false, false, none, none, none⟩))

def event45289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53228⟩⟩, .operator (⟨45283, 0⟩, ⟨15802, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩)

def event45290 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53228⟩⟩, .operator (⟨45283, 1⟩, ⟨15802, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩)

def event45291 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53228⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7131⟩⟩) ⟨7031⟩ 15795)

def event45292 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53228⟩⟩, .relation 45291 0, ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact45293RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩]

theorem exact45293RawTermsValid :
    exact45293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53228⟩⟩) exact45293RawTerms .large 45286 (.finite 345633123169561229153141416722874415185920) (some (45288))

def event45294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33181⟩⟩) 0 ⟨7177⟩ 15500

def event45295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33181⟩⟩) 1 ⟨33180⟩ 38770

def event45296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33181⟩⟩) (.authority (.operator))

def exact45297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33181⟩⟩]⟩, (1)⟩]

theorem exact45297RawTermsValid :
    exact45297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33181⟩⟩) exact45297RawTerms .large 45296 .exactZero (none)

def event45298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34164⟩⟩) 0 ⟨33181⟩ 45297

def event45299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34164⟩⟩) (.authority (.operator))

def exact45300RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨34164⟩⟩]⟩, (1)⟩]

theorem exact45300RawTermsValid :
    exact45300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34164⟩⟩) exact45300RawTerms (.finite 8192) 45299 .exactZero (none)

def event45301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34166⟩⟩) 0 ⟨33560⟩ 39054

def event45302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34166⟩⟩) 1 ⟨34164⟩ 45300

def event45303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34166⟩⟩) (.product (.predecessor 0 45301 .coefficient) (.predecessor 1 45302 .coefficient) (⟨false, false, none, none, none⟩))

def event45304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34166⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨34164⟩⟩]⟩) [⟨.result 45300 .coefficient, false, none⟩])

def event45305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34166⟩⟩) (.product (.result 39054 .summary) (.transfer 45304) (⟨false, false, none, none, none⟩))

def event45306 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34166⟩⟩, .operator (⟨39054, 0⟩, ⟨45300, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34164⟩⟩]⟩, (1)⟩)

def event45307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34166⟩⟩, .operator (⟨39054, 1⟩, ⟨45300, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34164⟩⟩]⟩, (-1)⟩)

def event45308 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨34166⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34164⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34164⟩⟩) ⟨33181⟩ 45297)

def event45309 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34166⟩⟩, .relation 45308 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31900⟩⟩], [⟨.program ⟨257⟩, ⟨33181⟩⟩]⟩, (-1)⟩)

def exact45310RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31900⟩⟩], [⟨.program ⟨257⟩, ⟨33181⟩⟩]⟩, (-1)⟩]

theorem exact45310RawTermsValid :
    exact45310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event45310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34166⟩⟩) exact45310RawTerms .large 45303 (.finite 32189200113374879571150551121920) (some (45305))

def event45311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32872⟩⟩) 0 ⟨31901⟩ 1181

def eventLeaf2816 : Array AnnotatedEvent := #[
  { event := event45056
    frameStart := 44954 },
  { event := event45057
    frameStart := 44954 },
  { event := event45058
    frameStart := 0 },
  { event := event45059
    frameStart := 0 },
  { event := event45060
    frameStart := 0 },
  { event := event45061
    frameStart := 0 },
  { event := event45062
    frameStart := 0 },
  { event := event45063
    frameStart := 0 },
  { event := event45064
    frameStart := 0 },
  { event := event45065
    frameStart := 0 },
  { event := event45066
    frameStart := 0 },
  { event := event45067
    frameStart := 0 },
  { event := event45068
    frameStart := 0 },
  { event := event45069
    frameStart := 0 },
  { event := event45070
    frameStart := 0 },
  { event := event45071
    frameStart := 0 }
]

def eventLeaf2817 : Array AnnotatedEvent := #[
  { event := event45072
    frameStart := 0 },
  { event := event45073
    frameStart := 0 },
  { event := event45074
    frameStart := 0 },
  { event := event45075
    frameStart := 0 },
  { event := event45076
    frameStart := 0 },
  { event := event45077
    frameStart := 0 },
  { event := event45078
    frameStart := 0 },
  { event := event45079
    frameStart := 0 },
  { event := event45080
    frameStart := 0 },
  { event := event45081
    frameStart := 0 },
  { event := event45082
    frameStart := 0 },
  { event := event45083
    frameStart := 0 },
  { event := event45084
    frameStart := 0 },
  { event := event45085
    frameStart := 0 },
  { event := event45086
    frameStart := 0 },
  { event := event45087
    frameStart := 0 }
]

def eventLeaf2818 : Array AnnotatedEvent := #[
  { event := event45088
    frameStart := 0 },
  { event := event45089
    frameStart := 0 },
  { event := event45090
    frameStart := 0 },
  { event := event45091
    frameStart := 0 },
  { event := event45092
    frameStart := 0 },
  { event := event45093
    frameStart := 0 },
  { event := event45094
    frameStart := 0 },
  { event := event45095
    frameStart := 0 },
  { event := event45096
    frameStart := 0 },
  { event := event45097
    frameStart := 0 },
  { event := event45098
    frameStart := 0 },
  { event := event45099
    frameStart := 0 },
  { event := event45100
    frameStart := 0 },
  { event := event45101
    frameStart := 0 },
  { event := event45102
    frameStart := 0 },
  { event := event45103
    frameStart := 0 }
]

def eventLeaf2819 : Array AnnotatedEvent := #[
  { event := event45104
    frameStart := 0 },
  { event := event45105
    frameStart := 0 },
  { event := event45106
    frameStart := 0 },
  { event := event45107
    frameStart := 0 },
  { event := event45108
    frameStart := 0 },
  { event := event45109
    frameStart := 0 },
  { event := event45110
    frameStart := 0 },
  { event := event45111
    frameStart := 0 },
  { event := event45112
    frameStart := 45112 },
  { event := event45113
    frameStart := 45112 },
  { event := event45114
    frameStart := 45112 },
  { event := event45115
    frameStart := 45112 },
  { event := event45116
    frameStart := 45112 },
  { event := event45117
    frameStart := 45112 },
  { event := event45118
    frameStart := 45112 },
  { event := event45119
    frameStart := 45112 }
]

def eventLeaf2820 : Array AnnotatedEvent := #[
  { event := event45120
    frameStart := 45112 },
  { event := event45121
    frameStart := 45112 },
  { event := event45122
    frameStart := 45112 },
  { event := event45123
    frameStart := 45112 },
  { event := event45124
    frameStart := 45112 },
  { event := event45125
    frameStart := 45112 },
  { event := event45126
    frameStart := 45112 },
  { event := event45127
    frameStart := 45112 },
  { event := event45128
    frameStart := 45112 },
  { event := event45129
    frameStart := 45112 },
  { event := event45130
    frameStart := 45112 },
  { event := event45131
    frameStart := 45112 },
  { event := event45132
    frameStart := 45112 },
  { event := event45133
    frameStart := 45112 },
  { event := event45134
    frameStart := 45112 },
  { event := event45135
    frameStart := 45112 }
]

def eventLeaf2821 : Array AnnotatedEvent := #[
  { event := event45136
    frameStart := 45112 },
  { event := event45137
    frameStart := 45112 },
  { event := event45138
    frameStart := 45112 },
  { event := event45139
    frameStart := 45112 },
  { event := event45140
    frameStart := 45112 },
  { event := event45141
    frameStart := 45112 },
  { event := event45142
    frameStart := 45112 },
  { event := event45143
    frameStart := 45112 },
  { event := event45144
    frameStart := 45112 },
  { event := event45145
    frameStart := 45112 },
  { event := event45146
    frameStart := 45112 },
  { event := event45147
    frameStart := 45112 },
  { event := event45148
    frameStart := 45112 },
  { event := event45149
    frameStart := 45112 },
  { event := event45150
    frameStart := 45112 },
  { event := event45151
    frameStart := 45112 }
]

def eventLeaf2822 : Array AnnotatedEvent := #[
  { event := event45152
    frameStart := 45112 },
  { event := event45153
    frameStart := 45112 },
  { event := event45154
    frameStart := 45112 },
  { event := event45155
    frameStart := 45112 },
  { event := event45156
    frameStart := 45112 },
  { event := event45157
    frameStart := 45112 },
  { event := event45158
    frameStart := 45112 },
  { event := event45159
    frameStart := 45112 },
  { event := event45160
    frameStart := 45112 },
  { event := event45161
    frameStart := 45112 },
  { event := event45162
    frameStart := 45112 },
  { event := event45163
    frameStart := 45112 },
  { event := event45164
    frameStart := 45112 },
  { event := event45165
    frameStart := 45112 },
  { event := event45166
    frameStart := 45166 },
  { event := event45167
    frameStart := 45166 }
]

def eventLeaf2823 : Array AnnotatedEvent := #[
  { event := event45168
    frameStart := 45166 },
  { event := event45169
    frameStart := 45166 },
  { event := event45170
    frameStart := 45166 },
  { event := event45171
    frameStart := 45166 },
  { event := event45172
    frameStart := 45166 },
  { event := event45173
    frameStart := 45166 },
  { event := event45174
    frameStart := 45166 },
  { event := event45175
    frameStart := 45166 },
  { event := event45176
    frameStart := 45166 },
  { event := event45177
    frameStart := 45166 },
  { event := event45178
    frameStart := 45166 },
  { event := event45179
    frameStart := 45166 },
  { event := event45180
    frameStart := 45166 },
  { event := event45181
    frameStart := 45166 },
  { event := event45182
    frameStart := 45166 },
  { event := event45183
    frameStart := 45166 }
]

def eventLeaf2824 : Array AnnotatedEvent := #[
  { event := event45184
    frameStart := 45166 },
  { event := event45185
    frameStart := 45166 },
  { event := event45186
    frameStart := 45166 },
  { event := event45187
    frameStart := 45166 },
  { event := event45188
    frameStart := 45166 },
  { event := event45189
    frameStart := 45166 },
  { event := event45190
    frameStart := 45166 },
  { event := event45191
    frameStart := 45166 },
  { event := event45192
    frameStart := 45166 },
  { event := event45193
    frameStart := 45166 },
  { event := event45194
    frameStart := 45166 },
  { event := event45195
    frameStart := 45166 },
  { event := event45196
    frameStart := 45166 },
  { event := event45197
    frameStart := 45166 },
  { event := event45198
    frameStart := 45166 },
  { event := event45199
    frameStart := 45166 }
]

def eventLeaf2825 : Array AnnotatedEvent := #[
  { event := event45200
    frameStart := 45166 },
  { event := event45201
    frameStart := 45166 },
  { event := event45202
    frameStart := 45166 },
  { event := event45203
    frameStart := 45166 },
  { event := event45204
    frameStart := 45166 },
  { event := event45205
    frameStart := 45166 },
  { event := event45206
    frameStart := 45166 },
  { event := event45207
    frameStart := 45166 },
  { event := event45208
    frameStart := 45166 },
  { event := event45209
    frameStart := 45166 },
  { event := event45210
    frameStart := 45166 },
  { event := event45211
    frameStart := 45166 },
  { event := event45212
    frameStart := 45166 },
  { event := event45213
    frameStart := 45166 },
  { event := event45214
    frameStart := 45166 },
  { event := event45215
    frameStart := 45166 }
]

def eventLeaf2826 : Array AnnotatedEvent := #[
  { event := event45216
    frameStart := 45166 },
  { event := event45217
    frameStart := 45166 },
  { event := event45218
    frameStart := 45166 },
  { event := event45219
    frameStart := 45166 },
  { event := event45220
    frameStart := 45166 },
  { event := event45221
    frameStart := 45166 },
  { event := event45222
    frameStart := 45166 },
  { event := event45223
    frameStart := 45166 },
  { event := event45224
    frameStart := 45166 },
  { event := event45225
    frameStart := 45166 },
  { event := event45226
    frameStart := 45166 },
  { event := event45227
    frameStart := 45166 },
  { event := event45228
    frameStart := 45166 },
  { event := event45229
    frameStart := 45166 },
  { event := event45230
    frameStart := 45166 },
  { event := event45231
    frameStart := 45166 }
]

def eventLeaf2827 : Array AnnotatedEvent := #[
  { event := event45232
    frameStart := 45166 },
  { event := event45233
    frameStart := 45166 },
  { event := event45234
    frameStart := 45166 },
  { event := event45235
    frameStart := 45166 },
  { event := event45236
    frameStart := 45166 },
  { event := event45237
    frameStart := 45166 },
  { event := event45238
    frameStart := 45166 },
  { event := event45239
    frameStart := 45166 },
  { event := event45240
    frameStart := 45166 },
  { event := event45241
    frameStart := 45166 },
  { event := event45242
    frameStart := 45166 },
  { event := event45243
    frameStart := 45166 },
  { event := event45244
    frameStart := 45166 },
  { event := event45245
    frameStart := 45166 },
  { event := event45246
    frameStart := 45166 },
  { event := event45247
    frameStart := 45166 }
]

def eventLeaf2828 : Array AnnotatedEvent := #[
  { event := event45248
    frameStart := 45166 },
  { event := event45249
    frameStart := 45166 },
  { event := event45250
    frameStart := 45166 },
  { event := event45251
    frameStart := 45166 },
  { event := event45252
    frameStart := 45166 },
  { event := event45253
    frameStart := 45166 },
  { event := event45254
    frameStart := 45166 },
  { event := event45255
    frameStart := 45166 },
  { event := event45256
    frameStart := 45166 },
  { event := event45257
    frameStart := 45166 },
  { event := event45258
    frameStart := 45166 },
  { event := event45259
    frameStart := 45166 },
  { event := event45260
    frameStart := 45166 },
  { event := event45261
    frameStart := 45166 },
  { event := event45262
    frameStart := 45166 },
  { event := event45263
    frameStart := 45166 }
]

def eventLeaf2829 : Array AnnotatedEvent := #[
  { event := event45264
    frameStart := 45166 },
  { event := event45265
    frameStart := 45166 },
  { event := event45266
    frameStart := 45166 },
  { event := event45267
    frameStart := 45166 },
  { event := event45268
    frameStart := 45166 },
  { event := event45269
    frameStart := 45166 },
  { event := event45270
    frameStart := 0 },
  { event := event45271
    frameStart := 0 },
  { event := event45272
    frameStart := 0 },
  { event := event45273
    frameStart := 0 },
  { event := event45274
    frameStart := 0 },
  { event := event45275
    frameStart := 0 },
  { event := event45276
    frameStart := 0 },
  { event := event45277
    frameStart := 0 },
  { event := event45278
    frameStart := 0 },
  { event := event45279
    frameStart := 0 }
]

def eventLeaf2830 : Array AnnotatedEvent := #[
  { event := event45280
    frameStart := 0 },
  { event := event45281
    frameStart := 0 },
  { event := event45282
    frameStart := 0 },
  { event := event45283
    frameStart := 0 },
  { event := event45284
    frameStart := 0 },
  { event := event45285
    frameStart := 0 },
  { event := event45286
    frameStart := 0 },
  { event := event45287
    frameStart := 0 },
  { event := event45288
    frameStart := 0 },
  { event := event45289
    frameStart := 0 },
  { event := event45290
    frameStart := 0 },
  { event := event45291
    frameStart := 0 },
  { event := event45292
    frameStart := 0 },
  { event := event45293
    frameStart := 0 },
  { event := event45294
    frameStart := 0 },
  { event := event45295
    frameStart := 0 }
]

def eventLeaf2831 : Array AnnotatedEvent := #[
  { event := event45296
    frameStart := 0 },
  { event := event45297
    frameStart := 0 },
  { event := event45298
    frameStart := 0 },
  { event := event45299
    frameStart := 0 },
  { event := event45300
    frameStart := 0 },
  { event := event45301
    frameStart := 0 },
  { event := event45302
    frameStart := 0 },
  { event := event45303
    frameStart := 0 },
  { event := event45304
    frameStart := 0 },
  { event := event45305
    frameStart := 0 },
  { event := event45306
    frameStart := 0 },
  { event := event45307
    frameStart := 0 },
  { event := event45308
    frameStart := 0 },
  { event := event45309
    frameStart := 0 },
  { event := event45310
    frameStart := 0 },
  { event := event45311
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events176
