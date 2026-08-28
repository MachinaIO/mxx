import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events414

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event105984 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27608⟩⟩) 1 ⟨27607⟩ 105959

def event105985 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27608⟩⟩) (.product (.predecessor 0 105983 .coefficient) (.predecessor 1 105984 .coefficient) (⟨false, false, none, none, none⟩))

def event105986 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27608⟩⟩, .operator (⟨105982, 0⟩, ⟨105959, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (1)⟩)

def event105987 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27608⟩⟩, .operator (⟨105982, 1⟩, ⟨105959, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (-1)⟩)

def event105988 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27608⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27607⟩⟩) ⟨24089⟩ 105956)

def event105989 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27608⟩⟩, .relation 105988 0, ⟨[⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24089⟩⟩]⟩, (-1)⟩)

def exact105990RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24089⟩⟩]⟩, (-1)⟩]

theorem exact105990RawTermsValid :
    exact105990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105990 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27608⟩⟩) exact105990RawTerms .large 105985 .exactZero (none)

def event105991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17211⟩⟩) 0 ⟨15812⟩ 105948

def event105992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17211⟩⟩) (.authority (.programFamilyFact))

def exact105993RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩]

theorem exact105993RawTermsValid :
    exact105993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17211⟩⟩) exact105993RawTerms (.finite 16) 105992 .exactZero (none)

def event105994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17213⟩⟩) 0 ⟨6544⟩ 105970

def event105995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17213⟩⟩) 1 ⟨17211⟩ 105993

def event105996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17213⟩⟩) (.product (.predecessor 0 105994 .coefficient) (.predecessor 1 105995 .coefficient) (⟨false, true, none, none, some 1⟩))

def event105997 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17213⟩⟩, .operator (⟨105970, 0⟩, ⟨105993, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact105998RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105998RawTermsValid :
    exact105998RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105998 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17213⟩⟩) exact105998RawTerms .large 105996 .exactZero (none)

def event105999 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6720⟩⟩) 0 ⟨6689⟩ 105952

def event106000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6720⟩⟩) (.authority (.operator))

def exact106001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩]

theorem exact106001RawTermsValid :
    exact106001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6720⟩⟩) exact106001RawTerms .large 106000 .exactZero (none)

def event106002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17214⟩⟩) 0 ⟨6720⟩ 106001

def event106003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17214⟩⟩) 1 ⟨17213⟩ 105998

def event106004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17214⟩⟩) (.sum [.predecessor 0 106002 .coefficient, .predecessor 1 106003 .coefficient])

def exact106005RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106005RawTermsValid :
    exact106005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106005 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17214⟩⟩) exact106005RawTerms .large 106004 .exactZero (none)

def event106006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27613⟩⟩) 0 ⟨17214⟩ 106005

def event106007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27613⟩⟩) 1 ⟨27608⟩ 105990

def event106008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27613⟩⟩) (.sum [.predecessor 0 106006 .coefficient, .predecessor 1 106007 .coefficient])

def exact106009RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24089⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106009RawTermsValid :
    exact106009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27613⟩⟩) exact106009RawTerms .large 106008 .exactZero (none)

def event106010 : Event := .preFoldPolynomial 106009 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24089⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact106011RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24089⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event106011 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27613⟩⟩) 106010 exact106011RawTerms .large 106008 .exactZero (none)

def event106012 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15812⟩⟩) ⟨⟨133⟩, ⟨40⟩, ⟨109⟩⟩ ⟨105878, 106012⟩

def event106013 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21176⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21173⟩⟩]⟩) (1) 0 2 (.universal 106012 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21173⟩⟩]⟩) (none) 106011)

def event106014 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21176⟩⟩, .relation 106013 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩)

def event106015 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21176⟩⟩, .relation 106013 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (-1)⟩)

def event106016 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21176⟩⟩, .relation 106013 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24089⟩⟩]⟩, (1)⟩)

def event106017 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21176⟩⟩, .relation 106013 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106018RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24089⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106018RawTermsValid :
    exact106018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106018 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21176⟩⟩) exact106018RawTerms .large 105874 (.finite 1811303510016) (some (105876))

def event106019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27610⟩⟩) 0 ⟨21176⟩ 106018

def event106020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27610⟩⟩) 1 ⟨27609⟩ 105864

def event106021 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27610⟩⟩) (.sum [.predecessor 0 106019 .coefficient, .predecessor 1 106020 .coefficient])

def event106022 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27610⟩⟩, .operator (⟨106018, 0⟩, ⟨105864, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩, (1)⟩)

def event106023 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27610⟩⟩, .operator (⟨106018, 2⟩, ⟨105864, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24089⟩⟩]⟩, (-1)⟩)

def event106024 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27610⟩⟩) (.sum [.result 106018 .summary, .result 105864 .summary])

def exact106025RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106025RawTermsValid :
    exact106025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106025 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27610⟩⟩) exact106025RawTerms .large 106021 (.finite 1292046061494565744640) (some (106024))

def event106026 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27611⟩⟩) 0 ⟨27610⟩ 106025

def event106027 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27611⟩⟩) 1 ⟨6644⟩ 5739

def event106028 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27611⟩⟩) (.product (.predecessor 0 106026 .coefficient) (.predecessor 1 106027 .coefficient) (⟨false, false, none, none, none⟩))

def event106029 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27611⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) [⟨.result 5735 .coefficient, false, none⟩])

def event106030 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27611⟩⟩) (.product (.result 106025 .summary) (.transfer 106029) (⟨false, false, none, none, none⟩))

def event106031 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27611⟩⟩, .operator (⟨106025, 0⟩, ⟨5739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩)

def event106032 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27611⟩⟩, .operator (⟨106025, 1⟩, ⟨5739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (-1)⟩)

def event106033 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27611⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6643⟩⟩) ⟨6593⟩ 5732)

def event106034 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27611⟩⟩, .relation 106033 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106035RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106035RawTermsValid :
    exact106035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106035 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27611⟩⟩) exact106035RawTerms .large 106028 (.finite 4741829718422040195880714240) (some (106030))

def event106036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24026⟩⟩) 0 ⟨6689⟩ 5477

def event106037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24026⟩⟩) 1 ⟨24025⟩ 99572

def event106038 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24026⟩⟩) (.authority (.operator))

def exact106039RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (1)⟩]

theorem exact106039RawTermsValid :
    exact106039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106039 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24026⟩⟩) exact106039RawTerms .large 106038 .exactZero (none)

def event106040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27390⟩⟩) 0 ⟨24026⟩ 106039

def event106041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27390⟩⟩) (.authority (.operator))

def exact106042RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (1)⟩]

theorem exact106042RawTermsValid :
    exact106042RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106042 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27390⟩⟩) exact106042RawTerms (.finite 8192) 106041 .exactZero (none)

def event106043 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27392⟩⟩) 0 ⟨25901⟩ 99832

def event106044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27392⟩⟩) 1 ⟨27390⟩ 106042

def event106045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27392⟩⟩) (.product (.predecessor 0 106043 .coefficient) (.predecessor 1 106044 .coefficient) (⟨false, false, none, none, none⟩))

def event106046 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27392⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩) [⟨.result 106042 .coefficient, false, none⟩])

def event106047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27392⟩⟩) (.product (.result 99832 .summary) (.transfer 106046) (⟨false, false, none, none, none⟩))

def event106048 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27392⟩⟩, .operator (⟨99832, 0⟩, ⟨106042, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (1)⟩)

def event106049 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27392⟩⟩, .operator (⟨99832, 1⟩, ⟨106042, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (-1)⟩)

def event106050 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27392⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27390⟩⟩) ⟨24026⟩ 106039)

def event106051 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27392⟩⟩, .relation 106050 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (-1)⟩)

def exact106052RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (-1)⟩]

theorem exact106052RawTermsValid :
    exact106052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106052 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27392⟩⟩) exact106052RawTerms .large 106045 (.finite 1292001234793221062656) (some (106047))

def event106053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21029⟩⟩) 0 ⟨15693⟩ 4859

def event106054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21029⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact106055RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩, (1)⟩]

theorem exact106055RawTermsValid :
    exact106055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106055 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21029⟩⟩) exact106055RawTerms (.finite 136065468) 106054 .exactZero (none)

def event106056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21031⟩⟩) 0 ⟨21029⟩ 106055

def event106057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21031⟩⟩) 1 ⟨2348⟩ 4

def event106058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21031⟩⟩) (.scale (.predecessor 0 106056 .coefficient) (.value (.predecessor 1 106057 .coefficient)))

def exact106059RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩, (1)⟩]

theorem exact106059RawTermsValid :
    exact106059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106059 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21031⟩⟩) exact106059RawTerms (.finite 136065468) 106058 .exactZero (none)

def event106060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21032⟩⟩) 0 ⟨5509⟩ 94462

def event106061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21032⟩⟩) 1 ⟨21031⟩ 106059

def event106062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21032⟩⟩) (.product (.predecessor 0 106060 .coefficient) (.predecessor 1 106061 .coefficient) (⟨false, false, none, none, none⟩))

def event106063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21032⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩) [⟨.result 106055 .coefficient, false, none⟩])

def event106064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21032⟩⟩) (.product (.result 94462 .summary) (.transfer 106063) (⟨false, false, none, none, none⟩))

def event106065 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21032⟩⟩, .operator (⟨94462, 0⟩, ⟨106059, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩, (1)⟩)

def event106066 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21030⟩⟩)

def event106067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event106068 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event106069 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event106070 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event106071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 106070

def event106072 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 106068

def event106073 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 106071 .coefficient) (.value (.predecessor 1 106072 .coefficient)))

def event106074 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event106075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11289⟩⟩) 0 ⟨5503⟩ 106074

def event106076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11289⟩⟩) (.authority (.programFamilyFact))

def exact106077RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩], []⟩, (1)⟩]

theorem exact106077RawTermsValid :
    exact106077RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106077 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11289⟩⟩) exact106077RawTerms (.finite 12) 106076 .exactZero (none)

def event106078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13746⟩⟩) 0 ⟨5503⟩ 106074

def event106079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13746⟩⟩) (.authority (.programFamilyFact))

def exact106080RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13746⟩⟩], []⟩, (1)⟩]

theorem exact106080RawTermsValid :
    exact106080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106080 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13746⟩⟩) exact106080RawTerms (.finite 12) 106079 .exactZero (none)

def event106081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13747⟩⟩) 0 ⟨13746⟩ 106080

def event106082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13747⟩⟩) 1 ⟨11289⟩ 106077

def event106083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13747⟩⟩) (.product (.predecessor 0 106081 .coefficient) (.predecessor 1 106082 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13747⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], []⟩) [⟨.result 106080 .coefficient, true, some 1⟩, ⟨.result 106077 .coefficient, true, some 1⟩])

def event106085 : Event := .survivorFold (1) 106084

def exact106086RawTerms : List Term := []

theorem exact106086RawTermsValid :
    exact106086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106086 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13747⟩⟩) exact106086RawTerms (.finite 144) 106083 (.finite 144) (some (106084))

def event106087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13748⟩⟩) 0 ⟨13747⟩ 106086

def event106088 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13748⟩⟩) (.identity (.predecessor 0 106087 .coefficient))

def event106089 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13748⟩⟩) (.finite 144)

def event106090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15692⟩⟩) 0 ⟨13748⟩ 106089

def event106091 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15692⟩⟩) (.authority (.programFamilyFact))

def exact106092RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], []⟩, (1)⟩]

theorem exact106092RawTermsValid :
    exact106092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15692⟩⟩) exact106092RawTerms (.finite 12) 106091 .exactZero (none)

def event106093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15693⟩⟩) 0 ⟨15692⟩ 106092

def event106094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15693⟩⟩) (.identity (.predecessor 0 106093 .coefficient))

def event106095 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15693⟩⟩) (.finite 12)

def event106096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21029⟩⟩) 0 ⟨15693⟩ 106095

def event106097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21029⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact106098RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩, (1)⟩]

theorem exact106098RawTermsValid :
    exact106098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106098 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21029⟩⟩) exact106098RawTerms (.finite 136065468) 106097 .exactZero (none)

def event106099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact106100RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact106100RawTermsValid :
    exact106100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106100 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact106100RawTerms .large 106099 .exactZero (none)

def event106101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21030⟩⟩) 0 ⟨6⟩ 106100

def event106102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21030⟩⟩) 1 ⟨21029⟩ 106098

def event106103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21030⟩⟩) (.product (.predecessor 0 106101 .coefficient) (.predecessor 1 106102 .coefficient) (⟨false, false, none, none, none⟩))

def event106104 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21030⟩⟩, .operator (⟨106100, 0⟩, ⟨106098, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩, (1)⟩)

def exact106105RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩, (1)⟩]

theorem exact106105RawTermsValid :
    exact106105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106105 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21030⟩⟩) exact106105RawTerms .large 106103 .exactZero (none)

def event106106 : Event := .preFoldPolynomial 106105 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩, (1)⟩] .exactZero none

def exact106107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩, (1)⟩]

def event106107 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21030⟩⟩) 106106 exact106107RawTerms .large 106103 .exactZero (none)

def event106108 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27396⟩⟩)

def event106109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event106110 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event106111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event106112 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event106113 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 106112

def event106114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 106110

def event106115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 106113 .coefficient) (.value (.predecessor 1 106114 .coefficient)))

def event106116 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event106117 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11289⟩⟩) 0 ⟨5503⟩ 106116

def event106118 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11289⟩⟩) (.authority (.programFamilyFact))

def exact106119RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩], []⟩, (1)⟩]

theorem exact106119RawTermsValid :
    exact106119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106119 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11289⟩⟩) exact106119RawTerms (.finite 12) 106118 .exactZero (none)

def event106120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13746⟩⟩) 0 ⟨5503⟩ 106116

def event106121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13746⟩⟩) (.authority (.programFamilyFact))

def exact106122RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13746⟩⟩], []⟩, (1)⟩]

theorem exact106122RawTermsValid :
    exact106122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106122 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13746⟩⟩) exact106122RawTerms (.finite 12) 106121 .exactZero (none)

def event106123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13747⟩⟩) 0 ⟨13746⟩ 106122

def event106124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13747⟩⟩) 1 ⟨11289⟩ 106119

def event106125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13747⟩⟩) (.product (.predecessor 0 106123 .coefficient) (.predecessor 1 106124 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106126 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13747⟩⟩, .operator (⟨106122, 0⟩, ⟨106119, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], []⟩, (1)⟩)

def exact106127RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], []⟩, (1)⟩]

theorem exact106127RawTermsValid :
    exact106127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106127 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13747⟩⟩) exact106127RawTerms (.finite 144) 106125 .exactZero (none)

def event106128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13748⟩⟩) 0 ⟨13747⟩ 106127

def event106129 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13748⟩⟩) (.identity (.predecessor 0 106128 .coefficient))

def event106130 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13748⟩⟩) (.finite 144)

def event106131 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15692⟩⟩) 0 ⟨13748⟩ 106130

def event106132 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15692⟩⟩) (.authority (.programFamilyFact))

def exact106133RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], []⟩, (1)⟩]

theorem exact106133RawTermsValid :
    exact106133RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106133 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15692⟩⟩) exact106133RawTerms (.finite 12) 106132 .exactZero (none)

def event106134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15693⟩⟩) 0 ⟨15692⟩ 106133

def event106135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15693⟩⟩) (.identity (.predecessor 0 106134 .coefficient))

def event106136 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15693⟩⟩) (.finite 12)

def event106137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24025⟩⟩) 0 ⟨15693⟩ 106136

def event106138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24025⟩⟩) (.authority (.programFamilyFact))

def event106139 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24025⟩⟩) (.finite 3720)

def event106140 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event106141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24026⟩⟩) 0 ⟨6689⟩ 106140

def event106142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24026⟩⟩) 1 ⟨24025⟩ 106139

def event106143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24026⟩⟩) (.authority (.operator))

def exact106144RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (1)⟩]

theorem exact106144RawTermsValid :
    exact106144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24026⟩⟩) exact106144RawTerms .large 106143 .exactZero (none)

def event106145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27390⟩⟩) 0 ⟨24026⟩ 106144

def event106146 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27390⟩⟩) (.authority (.operator))

def exact106147RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (1)⟩]

theorem exact106147RawTermsValid :
    exact106147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106147 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27390⟩⟩) exact106147RawTerms (.finite 8192) 106146 .exactZero (none)

def event106148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event106149 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event106150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15769⟩⟩) 0 ⟨15693⟩ 106136

def event106151 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15769⟩⟩) 1 ⟨110⟩ 106149

def event106152 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15769⟩⟩) (.sum [.predecessor 0 106150 .coefficient, .predecessor 1 106151 .coefficient])

def event106153 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15769⟩⟩) (.finite 12)

def event106154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15770⟩⟩) 0 ⟨15769⟩ 106153

def event106155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15770⟩⟩) (.identity (.predecessor 0 106154 .coefficient))

def exact106156RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], []⟩, (1)⟩]

theorem exact106156RawTermsValid :
    exact106156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106156 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15770⟩⟩) exact106156RawTerms (.finite 12) 106155 .exactZero (none)

def event106157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact106158RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106158RawTermsValid :
    exact106158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106158 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact106158RawTerms .large 106157 .exactZero (none)

def event106159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15771⟩⟩) 0 ⟨6544⟩ 106158

def event106160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15771⟩⟩) 1 ⟨15770⟩ 106156

def event106161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15771⟩⟩) (.product (.predecessor 0 106159 .coefficient) (.predecessor 1 106160 .coefficient) (⟨false, false, none, none, none⟩))

def event106162 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15771⟩⟩, .operator (⟨106158, 0⟩, ⟨106156, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact106163RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106163RawTermsValid :
    exact106163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106163 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15771⟩⟩) exact106163RawTerms .large 106161 .exactZero (none)

def event106164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6695⟩⟩) 0 ⟨6689⟩ 106140

def event106165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6695⟩⟩) (.authority (.operator))

def exact106166RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩]

theorem exact106166RawTermsValid :
    exact106166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106166 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6695⟩⟩) exact106166RawTerms .large 106165 .exactZero (none)

def event106167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15772⟩⟩) 0 ⟨6695⟩ 106166

def event106168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15772⟩⟩) 1 ⟨15771⟩ 106163

def event106169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15772⟩⟩) (.sum [.predecessor 0 106167 .coefficient, .predecessor 1 106168 .coefficient])

def exact106170RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106170RawTermsValid :
    exact106170RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106170 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15772⟩⟩) exact106170RawTerms .large 106169 .exactZero (none)

def event106171 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27391⟩⟩) 0 ⟨15772⟩ 106170

def event106172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27391⟩⟩) 1 ⟨27390⟩ 106147

def event106173 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27391⟩⟩) (.product (.predecessor 0 106171 .coefficient) (.predecessor 1 106172 .coefficient) (⟨false, false, none, none, none⟩))

def event106174 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27391⟩⟩, .operator (⟨106170, 0⟩, ⟨106147, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (1)⟩)

def event106175 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27391⟩⟩, .operator (⟨106170, 1⟩, ⟨106147, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (-1)⟩)

def event106176 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27391⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27390⟩⟩) ⟨24026⟩ 106144)

def event106177 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27391⟩⟩, .relation 106176 0, ⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (-1)⟩)

def exact106178RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (-1)⟩]

theorem exact106178RawTermsValid :
    exact106178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106178 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27391⟩⟩) exact106178RawTerms .large 106173 .exactZero (none)

def event106179 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17428⟩⟩) 0 ⟨15693⟩ 106136

def event106180 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17428⟩⟩) (.authority (.programFamilyFact))

def exact106181RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩]

theorem exact106181RawTermsValid :
    exact106181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106181 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17428⟩⟩) exact106181RawTerms (.finite 12) 106180 .exactZero (none)

def event106182 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17430⟩⟩) 0 ⟨6544⟩ 106158

def event106183 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17430⟩⟩) 1 ⟨17428⟩ 106181

def event106184 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17430⟩⟩) (.product (.predecessor 0 106182 .coefficient) (.predecessor 1 106183 .coefficient) (⟨false, true, none, none, some 1⟩))

def event106185 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17430⟩⟩, .operator (⟨106158, 0⟩, ⟨106181, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact106186RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106186RawTermsValid :
    exact106186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106186 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17430⟩⟩) exact106186RawTerms .large 106184 .exactZero (none)

def event106187 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6718⟩⟩) 0 ⟨6689⟩ 106140

def event106188 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6718⟩⟩) (.authority (.operator))

def exact106189RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩]

theorem exact106189RawTermsValid :
    exact106189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106189 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6718⟩⟩) exact106189RawTerms .large 106188 .exactZero (none)

def event106190 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17431⟩⟩) 0 ⟨6718⟩ 106189

def event106191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17431⟩⟩) 1 ⟨17430⟩ 106186

def event106192 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17431⟩⟩) (.sum [.predecessor 0 106190 .coefficient, .predecessor 1 106191 .coefficient])

def exact106193RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106193RawTermsValid :
    exact106193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106193 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17431⟩⟩) exact106193RawTerms .large 106192 .exactZero (none)

def event106194 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27396⟩⟩) 0 ⟨17431⟩ 106193

def event106195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27396⟩⟩) 1 ⟨27391⟩ 106178

def event106196 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27396⟩⟩) (.sum [.predecessor 0 106194 .coefficient, .predecessor 1 106195 .coefficient])

def exact106197RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106197RawTermsValid :
    exact106197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106197 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27396⟩⟩) exact106197RawTerms .large 106196 .exactZero (none)

def event106198 : Event := .preFoldPolynomial 106197 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact106199RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event106199 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27396⟩⟩) 106198 exact106199RawTerms .large 106196 .exactZero (none)

def event106200 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15693⟩⟩) ⟨⟨131⟩, ⟨38⟩, ⟨109⟩⟩ ⟨106066, 106200⟩

def event106201 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21032⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩) (1) 0 2 (.universal 106200 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩) (none) 106199)

def event106202 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21032⟩⟩, .relation 106201 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩)

def event106203 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21032⟩⟩, .relation 106201 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (-1)⟩)

def event106204 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21032⟩⟩, .relation 106201 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (1)⟩)

def event106205 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21032⟩⟩, .relation 106201 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106206RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106206RawTermsValid :
    exact106206RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106206 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21032⟩⟩) exact106206RawTerms .large 106062 (.finite 1811303510016) (some (106064))

def event106207 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27393⟩⟩) 0 ⟨21032⟩ 106206

def event106208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27393⟩⟩) 1 ⟨27392⟩ 106052

def event106209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27393⟩⟩) (.sum [.predecessor 0 106207 .coefficient, .predecessor 1 106208 .coefficient])

def event106210 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27393⟩⟩, .operator (⟨106206, 0⟩, ⟨106052, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27390⟩⟩]⟩, (1)⟩)

def event106211 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27393⟩⟩, .operator (⟨106206, 2⟩, ⟨106052, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨24026⟩⟩]⟩, (-1)⟩)

def event106212 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27393⟩⟩) (.sum [.result 106206 .summary, .result 106052 .summary])

def exact106213RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106213RawTermsValid :
    exact106213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106213 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27393⟩⟩) exact106213RawTerms .large 106209 (.finite 1292001236604524572672) (some (106212))

def event106214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27394⟩⟩) 0 ⟨27393⟩ 106213

def event106215 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27394⟩⟩) 1 ⟨6648⟩ 5759

def event106216 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27394⟩⟩) (.product (.predecessor 0 106214 .coefficient) (.predecessor 1 106215 .coefficient) (⟨false, false, none, none, none⟩))

def event106217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27394⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩) [⟨.result 5755 .coefficient, false, none⟩])

def event106218 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27394⟩⟩) (.product (.result 106213 .summary) (.transfer 106217) (⟨false, false, none, none, none⟩))

def event106219 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27394⟩⟩, .operator (⟨106213, 0⟩, ⟨5759, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩)

def event106220 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27394⟩⟩, .operator (⟨106213, 1⟩, ⟨5759, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (-1)⟩)

def event106221 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27394⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6647⟩⟩) ⟨6595⟩ 5752)

def event106222 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27394⟩⟩, .relation 106221 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106223RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106223RawTermsValid :
    exact106223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106223 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27394⟩⟩) exact106223RawTerms .large 106216 (.finite 4741665210358390854099402752) (some (106218))

def event106224 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23963⟩⟩) 0 ⟨6689⟩ 5477

def event106225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23963⟩⟩) 1 ⟨23962⟩ 100006

def event106226 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23963⟩⟩) (.authority (.operator))

def exact106227RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (1)⟩]

theorem exact106227RawTermsValid :
    exact106227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106227 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23963⟩⟩) exact106227RawTerms .large 106226 .exactZero (none)

def event106228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27173⟩⟩) 0 ⟨23963⟩ 106227

def event106229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27173⟩⟩) (.authority (.operator))

def exact106230RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (1)⟩]

theorem exact106230RawTermsValid :
    exact106230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106230 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27173⟩⟩) exact106230RawTerms (.finite 8192) 106229 .exactZero (none)

def event106231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27175⟩⟩) 0 ⟨25824⟩ 100266

def event106232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27175⟩⟩) 1 ⟨27173⟩ 106230

def event106233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27175⟩⟩) (.product (.predecessor 0 106231 .coefficient) (.predecessor 1 106232 .coefficient) (⟨false, false, none, none, none⟩))

def event106234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27175⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩) [⟨.result 106230 .coefficient, false, none⟩])

def event106235 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27175⟩⟩) (.product (.result 100266 .summary) (.transfer 106234) (⟨false, false, none, none, none⟩))

def event106236 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27175⟩⟩, .operator (⟨100266, 0⟩, ⟨106230, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (1)⟩)

def event106237 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27175⟩⟩, .operator (⟨100266, 1⟩, ⟨106230, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (-1)⟩)

def event106238 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27175⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27173⟩⟩) ⟨23963⟩ 106227)

def event106239 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27175⟩⟩, .relation 106238 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (-1)⟩)

def eventLeaf6624 : Array AnnotatedEvent := #[
  { event := event105984
    frameStart := 105920 },
  { event := event105985
    frameStart := 105920 },
  { event := event105986
    frameStart := 105920 },
  { event := event105987
    frameStart := 105920 },
  { event := event105988
    frameStart := 105920 },
  { event := event105989
    frameStart := 105920 },
  { event := event105990
    frameStart := 105920 },
  { event := event105991
    frameStart := 105920 },
  { event := event105992
    frameStart := 105920 },
  { event := event105993
    frameStart := 105920 },
  { event := event105994
    frameStart := 105920 },
  { event := event105995
    frameStart := 105920 },
  { event := event105996
    frameStart := 105920 },
  { event := event105997
    frameStart := 105920 },
  { event := event105998
    frameStart := 105920 },
  { event := event105999
    frameStart := 105920 }
]

def eventLeaf6625 : Array AnnotatedEvent := #[
  { event := event106000
    frameStart := 105920 },
  { event := event106001
    frameStart := 105920 },
  { event := event106002
    frameStart := 105920 },
  { event := event106003
    frameStart := 105920 },
  { event := event106004
    frameStart := 105920 },
  { event := event106005
    frameStart := 105920 },
  { event := event106006
    frameStart := 105920 },
  { event := event106007
    frameStart := 105920 },
  { event := event106008
    frameStart := 105920 },
  { event := event106009
    frameStart := 105920 },
  { event := event106010
    frameStart := 105920 },
  { event := event106011
    frameStart := 105920 },
  { event := event106012
    frameStart := 0 },
  { event := event106013
    frameStart := 0 },
  { event := event106014
    frameStart := 0 },
  { event := event106015
    frameStart := 0 }
]

def eventLeaf6626 : Array AnnotatedEvent := #[
  { event := event106016
    frameStart := 0 },
  { event := event106017
    frameStart := 0 },
  { event := event106018
    frameStart := 0 },
  { event := event106019
    frameStart := 0 },
  { event := event106020
    frameStart := 0 },
  { event := event106021
    frameStart := 0 },
  { event := event106022
    frameStart := 0 },
  { event := event106023
    frameStart := 0 },
  { event := event106024
    frameStart := 0 },
  { event := event106025
    frameStart := 0 },
  { event := event106026
    frameStart := 0 },
  { event := event106027
    frameStart := 0 },
  { event := event106028
    frameStart := 0 },
  { event := event106029
    frameStart := 0 },
  { event := event106030
    frameStart := 0 },
  { event := event106031
    frameStart := 0 }
]

def eventLeaf6627 : Array AnnotatedEvent := #[
  { event := event106032
    frameStart := 0 },
  { event := event106033
    frameStart := 0 },
  { event := event106034
    frameStart := 0 },
  { event := event106035
    frameStart := 0 },
  { event := event106036
    frameStart := 0 },
  { event := event106037
    frameStart := 0 },
  { event := event106038
    frameStart := 0 },
  { event := event106039
    frameStart := 0 },
  { event := event106040
    frameStart := 0 },
  { event := event106041
    frameStart := 0 },
  { event := event106042
    frameStart := 0 },
  { event := event106043
    frameStart := 0 },
  { event := event106044
    frameStart := 0 },
  { event := event106045
    frameStart := 0 },
  { event := event106046
    frameStart := 0 },
  { event := event106047
    frameStart := 0 }
]

def eventLeaf6628 : Array AnnotatedEvent := #[
  { event := event106048
    frameStart := 0 },
  { event := event106049
    frameStart := 0 },
  { event := event106050
    frameStart := 0 },
  { event := event106051
    frameStart := 0 },
  { event := event106052
    frameStart := 0 },
  { event := event106053
    frameStart := 0 },
  { event := event106054
    frameStart := 0 },
  { event := event106055
    frameStart := 0 },
  { event := event106056
    frameStart := 0 },
  { event := event106057
    frameStart := 0 },
  { event := event106058
    frameStart := 0 },
  { event := event106059
    frameStart := 0 },
  { event := event106060
    frameStart := 0 },
  { event := event106061
    frameStart := 0 },
  { event := event106062
    frameStart := 0 },
  { event := event106063
    frameStart := 0 }
]

def eventLeaf6629 : Array AnnotatedEvent := #[
  { event := event106064
    frameStart := 0 },
  { event := event106065
    frameStart := 0 },
  { event := event106066
    frameStart := 106066 },
  { event := event106067
    frameStart := 106066 },
  { event := event106068
    frameStart := 106066 },
  { event := event106069
    frameStart := 106066 },
  { event := event106070
    frameStart := 106066 },
  { event := event106071
    frameStart := 106066 },
  { event := event106072
    frameStart := 106066 },
  { event := event106073
    frameStart := 106066 },
  { event := event106074
    frameStart := 106066 },
  { event := event106075
    frameStart := 106066 },
  { event := event106076
    frameStart := 106066 },
  { event := event106077
    frameStart := 106066 },
  { event := event106078
    frameStart := 106066 },
  { event := event106079
    frameStart := 106066 }
]

def eventLeaf6630 : Array AnnotatedEvent := #[
  { event := event106080
    frameStart := 106066 },
  { event := event106081
    frameStart := 106066 },
  { event := event106082
    frameStart := 106066 },
  { event := event106083
    frameStart := 106066 },
  { event := event106084
    frameStart := 106066 },
  { event := event106085
    frameStart := 106066 },
  { event := event106086
    frameStart := 106066 },
  { event := event106087
    frameStart := 106066 },
  { event := event106088
    frameStart := 106066 },
  { event := event106089
    frameStart := 106066 },
  { event := event106090
    frameStart := 106066 },
  { event := event106091
    frameStart := 106066 },
  { event := event106092
    frameStart := 106066 },
  { event := event106093
    frameStart := 106066 },
  { event := event106094
    frameStart := 106066 },
  { event := event106095
    frameStart := 106066 }
]

def eventLeaf6631 : Array AnnotatedEvent := #[
  { event := event106096
    frameStart := 106066 },
  { event := event106097
    frameStart := 106066 },
  { event := event106098
    frameStart := 106066 },
  { event := event106099
    frameStart := 106066 },
  { event := event106100
    frameStart := 106066 },
  { event := event106101
    frameStart := 106066 },
  { event := event106102
    frameStart := 106066 },
  { event := event106103
    frameStart := 106066 },
  { event := event106104
    frameStart := 106066 },
  { event := event106105
    frameStart := 106066 },
  { event := event106106
    frameStart := 106066 },
  { event := event106107
    frameStart := 106066 },
  { event := event106108
    frameStart := 106108 },
  { event := event106109
    frameStart := 106108 },
  { event := event106110
    frameStart := 106108 },
  { event := event106111
    frameStart := 106108 }
]

def eventLeaf6632 : Array AnnotatedEvent := #[
  { event := event106112
    frameStart := 106108 },
  { event := event106113
    frameStart := 106108 },
  { event := event106114
    frameStart := 106108 },
  { event := event106115
    frameStart := 106108 },
  { event := event106116
    frameStart := 106108 },
  { event := event106117
    frameStart := 106108 },
  { event := event106118
    frameStart := 106108 },
  { event := event106119
    frameStart := 106108 },
  { event := event106120
    frameStart := 106108 },
  { event := event106121
    frameStart := 106108 },
  { event := event106122
    frameStart := 106108 },
  { event := event106123
    frameStart := 106108 },
  { event := event106124
    frameStart := 106108 },
  { event := event106125
    frameStart := 106108 },
  { event := event106126
    frameStart := 106108 },
  { event := event106127
    frameStart := 106108 }
]

def eventLeaf6633 : Array AnnotatedEvent := #[
  { event := event106128
    frameStart := 106108 },
  { event := event106129
    frameStart := 106108 },
  { event := event106130
    frameStart := 106108 },
  { event := event106131
    frameStart := 106108 },
  { event := event106132
    frameStart := 106108 },
  { event := event106133
    frameStart := 106108 },
  { event := event106134
    frameStart := 106108 },
  { event := event106135
    frameStart := 106108 },
  { event := event106136
    frameStart := 106108 },
  { event := event106137
    frameStart := 106108 },
  { event := event106138
    frameStart := 106108 },
  { event := event106139
    frameStart := 106108 },
  { event := event106140
    frameStart := 106108 },
  { event := event106141
    frameStart := 106108 },
  { event := event106142
    frameStart := 106108 },
  { event := event106143
    frameStart := 106108 }
]

def eventLeaf6634 : Array AnnotatedEvent := #[
  { event := event106144
    frameStart := 106108 },
  { event := event106145
    frameStart := 106108 },
  { event := event106146
    frameStart := 106108 },
  { event := event106147
    frameStart := 106108 },
  { event := event106148
    frameStart := 106108 },
  { event := event106149
    frameStart := 106108 },
  { event := event106150
    frameStart := 106108 },
  { event := event106151
    frameStart := 106108 },
  { event := event106152
    frameStart := 106108 },
  { event := event106153
    frameStart := 106108 },
  { event := event106154
    frameStart := 106108 },
  { event := event106155
    frameStart := 106108 },
  { event := event106156
    frameStart := 106108 },
  { event := event106157
    frameStart := 106108 },
  { event := event106158
    frameStart := 106108 },
  { event := event106159
    frameStart := 106108 }
]

def eventLeaf6635 : Array AnnotatedEvent := #[
  { event := event106160
    frameStart := 106108 },
  { event := event106161
    frameStart := 106108 },
  { event := event106162
    frameStart := 106108 },
  { event := event106163
    frameStart := 106108 },
  { event := event106164
    frameStart := 106108 },
  { event := event106165
    frameStart := 106108 },
  { event := event106166
    frameStart := 106108 },
  { event := event106167
    frameStart := 106108 },
  { event := event106168
    frameStart := 106108 },
  { event := event106169
    frameStart := 106108 },
  { event := event106170
    frameStart := 106108 },
  { event := event106171
    frameStart := 106108 },
  { event := event106172
    frameStart := 106108 },
  { event := event106173
    frameStart := 106108 },
  { event := event106174
    frameStart := 106108 },
  { event := event106175
    frameStart := 106108 }
]

def eventLeaf6636 : Array AnnotatedEvent := #[
  { event := event106176
    frameStart := 106108 },
  { event := event106177
    frameStart := 106108 },
  { event := event106178
    frameStart := 106108 },
  { event := event106179
    frameStart := 106108 },
  { event := event106180
    frameStart := 106108 },
  { event := event106181
    frameStart := 106108 },
  { event := event106182
    frameStart := 106108 },
  { event := event106183
    frameStart := 106108 },
  { event := event106184
    frameStart := 106108 },
  { event := event106185
    frameStart := 106108 },
  { event := event106186
    frameStart := 106108 },
  { event := event106187
    frameStart := 106108 },
  { event := event106188
    frameStart := 106108 },
  { event := event106189
    frameStart := 106108 },
  { event := event106190
    frameStart := 106108 },
  { event := event106191
    frameStart := 106108 }
]

def eventLeaf6637 : Array AnnotatedEvent := #[
  { event := event106192
    frameStart := 106108 },
  { event := event106193
    frameStart := 106108 },
  { event := event106194
    frameStart := 106108 },
  { event := event106195
    frameStart := 106108 },
  { event := event106196
    frameStart := 106108 },
  { event := event106197
    frameStart := 106108 },
  { event := event106198
    frameStart := 106108 },
  { event := event106199
    frameStart := 106108 },
  { event := event106200
    frameStart := 0 },
  { event := event106201
    frameStart := 0 },
  { event := event106202
    frameStart := 0 },
  { event := event106203
    frameStart := 0 },
  { event := event106204
    frameStart := 0 },
  { event := event106205
    frameStart := 0 },
  { event := event106206
    frameStart := 0 },
  { event := event106207
    frameStart := 0 }
]

def eventLeaf6638 : Array AnnotatedEvent := #[
  { event := event106208
    frameStart := 0 },
  { event := event106209
    frameStart := 0 },
  { event := event106210
    frameStart := 0 },
  { event := event106211
    frameStart := 0 },
  { event := event106212
    frameStart := 0 },
  { event := event106213
    frameStart := 0 },
  { event := event106214
    frameStart := 0 },
  { event := event106215
    frameStart := 0 },
  { event := event106216
    frameStart := 0 },
  { event := event106217
    frameStart := 0 },
  { event := event106218
    frameStart := 0 },
  { event := event106219
    frameStart := 0 },
  { event := event106220
    frameStart := 0 },
  { event := event106221
    frameStart := 0 },
  { event := event106222
    frameStart := 0 },
  { event := event106223
    frameStart := 0 }
]

def eventLeaf6639 : Array AnnotatedEvent := #[
  { event := event106224
    frameStart := 0 },
  { event := event106225
    frameStart := 0 },
  { event := event106226
    frameStart := 0 },
  { event := event106227
    frameStart := 0 },
  { event := event106228
    frameStart := 0 },
  { event := event106229
    frameStart := 0 },
  { event := event106230
    frameStart := 0 },
  { event := event106231
    frameStart := 0 },
  { event := event106232
    frameStart := 0 },
  { event := event106233
    frameStart := 0 },
  { event := event106234
    frameStart := 0 },
  { event := event106235
    frameStart := 0 },
  { event := event106236
    frameStart := 0 },
  { event := event106237
    frameStart := 0 },
  { event := event106238
    frameStart := 0 },
  { event := event106239
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events414
