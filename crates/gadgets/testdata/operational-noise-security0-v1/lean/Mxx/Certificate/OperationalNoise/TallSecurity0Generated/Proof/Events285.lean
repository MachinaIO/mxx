import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events285

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event72960 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26769⟩⟩, .operator (⟨72955, 1⟩, ⟨72932, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (-1)⟩)

def event72961 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26769⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26768⟩⟩) ⟨23844⟩ 72929)

def event72962 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26769⟩⟩, .relation 72961 0, ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (-1)⟩)

def exact72963RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (-1)⟩]

theorem exact72963RawTermsValid :
    exact72963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72963 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26769⟩⟩) exact72963RawTerms .large 72958 .exactZero (none)

def event72964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15362⟩⟩) 0 ⟨15111⟩ 72921

def event72965 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15362⟩⟩) (.authority (.programFamilyFact))

def exact72966RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩]

theorem exact72966RawTermsValid :
    exact72966RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72966 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15362⟩⟩) exact72966RawTerms (.finite 51) 72965 .exactZero (none)

def event72967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15364⟩⟩) 0 ⟨6544⟩ 72943

def event72968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15364⟩⟩) 1 ⟨15362⟩ 72966

def event72969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15364⟩⟩) (.product (.predecessor 0 72967 .coefficient) (.predecessor 1 72968 .coefficient) (⟨false, true, none, none, some 1⟩))

def event72970 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15364⟩⟩, .operator (⟨72943, 0⟩, ⟨72966, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact72971RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72971RawTermsValid :
    exact72971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72971 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15364⟩⟩) exact72971RawTerms .large 72969 .exactZero (none)

def event72972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6713⟩⟩) 0 ⟨6689⟩ 72925

def event72973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6713⟩⟩) (.authority (.operator))

def exact72974RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩]

theorem exact72974RawTermsValid :
    exact72974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72974 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6713⟩⟩) exact72974RawTerms .large 72973 .exactZero (none)

def event72975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15365⟩⟩) 0 ⟨6713⟩ 72974

def event72976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15365⟩⟩) 1 ⟨15364⟩ 72971

def event72977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15365⟩⟩) (.sum [.predecessor 0 72975 .coefficient, .predecessor 1 72976 .coefficient])

def exact72978RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72978RawTermsValid :
    exact72978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72978 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15365⟩⟩) exact72978RawTerms .large 72977 .exactZero (none)

def event72979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26773⟩⟩) 0 ⟨15365⟩ 72978

def event72980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26773⟩⟩) 1 ⟨26769⟩ 72963

def event72981 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26773⟩⟩) (.sum [.predecessor 0 72979 .coefficient, .predecessor 1 72980 .coefficient])

def exact72982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72982RawTermsValid :
    exact72982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72982 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26773⟩⟩) exact72982RawTerms .large 72981 .exactZero (none)

def event72983 : Event := .preFoldPolynomial 72982 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact72984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event72984 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26773⟩⟩) 72983 exact72984RawTerms .large 72981 .exactZero (none)

def event72985 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15111⟩⟩) ⟨⟨126⟩, ⟨32⟩, ⟨109⟩⟩ ⟨72827, 72985⟩

def event72986 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20679⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩) (1) 0 2 (.universal 72985 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩) (none) 72984)

def event72987 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20679⟩⟩, .relation 72986 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩)

def event72988 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20679⟩⟩, .relation 72986 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (-1)⟩)

def event72989 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20679⟩⟩, .relation 72986 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (1)⟩)

def event72990 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20679⟩⟩, .relation 72986 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact72991RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72991RawTermsValid :
    exact72991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20679⟩⟩) exact72991RawTerms .large 72823 (.finite 1811303510016) (some (72825))

def event72992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26771⟩⟩) 0 ⟨20679⟩ 72991

def event72993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26771⟩⟩) 1 ⟨26770⟩ 72813

def event72994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26771⟩⟩) (.sum [.predecessor 0 72992 .coefficient, .predecessor 1 72993 .coefficient])

def event72995 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26771⟩⟩, .operator (⟨72991, 0⟩, ⟨72813, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (1)⟩)

def event72996 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26771⟩⟩, .operator (⟨72991, 2⟩, ⟨72813, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (-1)⟩)

def event72997 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26771⟩⟩) (.sum [.result 72991 .summary, .result 72813 .summary])

def exact72998RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72998RawTermsValid :
    exact72998RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72998 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26771⟩⟩) exact72998RawTerms .large 72994 (.finite 1291911586824442228736) (some (72997))

def event72999 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23779⟩⟩) 0 ⟨14950⟩ 3471

def event73000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23779⟩⟩) (.authority (.programFamilyFact))

def event73001 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23779⟩⟩) (.finite 3720)

def event73002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23781⟩⟩) 0 ⟨6689⟩ 5477

def event73003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23781⟩⟩) 1 ⟨23779⟩ 73001

def event73004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23781⟩⟩) (.authority (.operator))

def exact73005RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23781⟩⟩]⟩, (1)⟩]

theorem exact73005RawTermsValid :
    exact73005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73005 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23781⟩⟩) exact73005RawTerms .large 73004 .exactZero (none)

def event73006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26551⟩⟩) 0 ⟨23781⟩ 73005

def event73007 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26551⟩⟩) (.authority (.operator))

def exact73008RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩, (1)⟩]

theorem exact73008RawTermsValid :
    exact73008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73008 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26551⟩⟩) exact73008RawTerms (.finite 8192) 73007 .exactZero (none)

def event73009 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22993⟩⟩) 0 ⟨10670⟩ 3465

def event73010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22993⟩⟩) (.authority (.programFamilyFact))

def event73011 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨22993⟩⟩) (.finite 3720)

def event73012 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22994⟩⟩) 0 ⟨6689⟩ 5477

def event73013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22994⟩⟩) 1 ⟨22993⟩ 73011

def event73014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22994⟩⟩) (.authority (.operator))

def exact73015RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22994⟩⟩]⟩, (1)⟩]

theorem exact73015RawTermsValid :
    exact73015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73015 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22994⟩⟩) exact73015RawTerms .large 73014 .exactZero (none)

def event73016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24983⟩⟩) 0 ⟨22994⟩ 73015

def event73017 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24983⟩⟩) (.authority (.operator))

def exact73018RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩, (1)⟩]

theorem exact73018RawTermsValid :
    exact73018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73018 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24983⟩⟩) exact73018RawTerms (.finite 8192) 73017 .exactZero (none)

def event73019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10671⟩⟩) 0 ⟨10668⟩ 3454

def event73020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10671⟩⟩) 1 ⟨6566⟩ 65295

def event73021 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10671⟩⟩) (.tensor (.predecessor 0 73019 .coefficient) (.predecessor 1 73020 .coefficient) true false)

def event73022 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10671⟩⟩, .operator (⟨3454, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73023RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73023RawTermsValid :
    exact73023RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73023 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10671⟩⟩) exact73023RawTerms .large 73021 .exactZero (none)

def event73024 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7191⟩⟩) 0 ⟨5533⟩ 65165

def event73025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7191⟩⟩) 1 ⟨6773⟩ 14488

def event73026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7191⟩⟩) (.product (.predecessor 0 73024 .coefficient) (.predecessor 1 73025 .coefficient) (⟨false, false, none, none, none⟩))

def event73027 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7191⟩⟩, .operator (⟨65165, 0⟩, ⟨14488, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩)

def exact73028RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩]

theorem exact73028RawTermsValid :
    exact73028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73028 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7191⟩⟩) exact73028RawTerms .large 73026 .exactZero (none)

def event73029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10672⟩⟩) 0 ⟨7191⟩ 73028

def event73030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10672⟩⟩) 1 ⟨10671⟩ 73023

def event73031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10672⟩⟩) (.sum [.predecessor 0 73029 .coefficient, .predecessor 1 73030 .coefficient])

def exact73032RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73032RawTermsValid :
    exact73032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10672⟩⟩) exact73032RawTerms .large 73031 .exactZero (none)

def event73033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10673⟩⟩) 0 ⟨10672⟩ 73032

def event73034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10673⟩⟩) 1 ⟨87⟩ 14480

def event73035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10673⟩⟩) (.sum [.predecessor 0 73033 .coefficient, .predecessor 1 73034 .coefficient])

def event73036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10673⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨87⟩⟩]⟩) [⟨.result 14480 .coefficient, false, none⟩])

def event73037 : Event := .survivorFold (1) 73036

def exact73038RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73038RawTermsValid :
    exact73038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73038 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10673⟩⟩) exact73038RawTerms .large 73035 (.finite 26) (some (73036))

def event73039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10674⟩⟩) 0 ⟨10673⟩ 73038

def event73040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10674⟩⟩) 1 ⟨9500⟩ 3457

def event73041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10674⟩⟩) (.product (.predecessor 0 73039 .coefficient) (.predecessor 1 73040 .coefficient) (⟨false, true, none, none, some 1⟩))

def event73042 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10674⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩], []⟩) [⟨.result 3457 .coefficient, true, some 1⟩])

def event73043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10674⟩⟩) (.product (.result 73038 .summary) (.transfer 73042) (⟨false, false, none, none, none⟩))

def event73044 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10674⟩⟩, .operator (⟨73038, 1⟩, ⟨3457, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event73045 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10674⟩⟩, .operator (⟨73038, 0⟩, ⟨3457, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩)

def exact73046RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73046RawTermsValid :
    exact73046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73046 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10674⟩⟩) exact73046RawTerms .large 73041 (.finite 2496) (some (73043))

def event73047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9501⟩⟩) 0 ⟨9500⟩ 3457

def event73048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9501⟩⟩) 1 ⟨6566⟩ 65295

def event73049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9501⟩⟩) (.tensor (.predecessor 0 73047 .coefficient) (.predecessor 1 73048 .coefficient) true false)

def event73050 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9501⟩⟩, .operator (⟨3457, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73051RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73051RawTermsValid :
    exact73051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73051 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9501⟩⟩) exact73051RawTerms .large 73049 .exactZero (none)

def event73052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7200⟩⟩) 0 ⟨5533⟩ 65165

def event73053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7200⟩⟩) 1 ⟨6782⟩ 14529

def event73054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7200⟩⟩) (.product (.predecessor 0 73052 .coefficient) (.predecessor 1 73053 .coefficient) (⟨false, false, none, none, none⟩))

def event73055 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7200⟩⟩, .operator (⟨65165, 0⟩, ⟨14529, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩]⟩, (1)⟩)

def exact73056RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩]⟩, (1)⟩]

theorem exact73056RawTermsValid :
    exact73056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73056 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7200⟩⟩) exact73056RawTerms .large 73054 .exactZero (none)

def event73057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9502⟩⟩) 0 ⟨7200⟩ 73056

def event73058 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9502⟩⟩) 1 ⟨9501⟩ 73051

def event73059 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9502⟩⟩) (.sum [.predecessor 0 73057 .coefficient, .predecessor 1 73058 .coefficient])

def exact73060RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73060RawTermsValid :
    exact73060RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73060 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9502⟩⟩) exact73060RawTerms .large 73059 .exactZero (none)

def event73061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9503⟩⟩) 0 ⟨9502⟩ 73060

def event73062 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9503⟩⟩) 1 ⟨96⟩ 14521

def event73063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9503⟩⟩) (.sum [.predecessor 0 73061 .coefficient, .predecessor 1 73062 .coefficient])

def event73064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9503⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨96⟩⟩]⟩) [⟨.result 14521 .coefficient, false, none⟩])

def event73065 : Event := .survivorFold (1) 73064

def exact73066RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73066RawTermsValid :
    exact73066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73066 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9503⟩⟩) exact73066RawTerms .large 73063 (.finite 26) (some (73064))

def event73067 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9504⟩⟩) 0 ⟨9503⟩ 73066

def event73068 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9504⟩⟩) 1 ⟨7835⟩ 14518

def event73069 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9504⟩⟩) (.product (.predecessor 0 73067 .coefficient) (.predecessor 1 73068 .coefficient) (⟨false, false, none, none, none⟩))

def event73070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9504⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) [⟨.result 14514 .coefficient, false, none⟩])

def event73071 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9504⟩⟩) (.product (.result 73066 .summary) (.transfer 73070) (⟨false, false, none, none, none⟩))

def event73072 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9504⟩⟩, .operator (⟨73066, 1⟩, ⟨14518, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (-1)⟩)

def event73073 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9504⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7834⟩⟩) ⟨6773⟩ 14488)

def event73074 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9504⟩⟩, .relation 73073 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (-1)⟩)

def event73075 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9504⟩⟩, .operator (⟨73066, 0⟩, ⟨14518, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩)

def exact73076RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (-1)⟩]

theorem exact73076RawTermsValid :
    exact73076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73076 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9504⟩⟩) exact73076RawTerms .large 73069 (.finite 95420416) (some (73071))

def event73077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10675⟩⟩) 0 ⟨9504⟩ 73076

def event73078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10675⟩⟩) 1 ⟨10674⟩ 73046

def event73079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10675⟩⟩) (.sum [.predecessor 0 73077 .coefficient, .predecessor 1 73078 .coefficient])

def event73080 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10675⟩⟩, .operator (⟨73076, 1⟩, ⟨73046, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩)

def event73081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10675⟩⟩) (.sum [.result 73076 .summary, .result 73046 .summary])

def exact73082RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73082RawTermsValid :
    exact73082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73082 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10675⟩⟩) exact73082RawTerms .large 73079 (.finite 95422912) (some (73081))

def event73083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24984⟩⟩) 0 ⟨10675⟩ 73082

def event73084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24984⟩⟩) 1 ⟨24983⟩ 73018

def event73085 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24984⟩⟩) (.product (.predecessor 0 73083 .coefficient) (.predecessor 1 73084 .coefficient) (⟨false, false, none, none, none⟩))

def event73086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24984⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) [⟨.result 73018 .coefficient, false, none⟩])

def event73087 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24984⟩⟩) (.product (.result 73082 .summary) (.transfer 73086) (⟨false, false, none, none, none⟩))

def event73088 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24984⟩⟩, .operator (⟨73082, 1⟩, ⟨73018, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩, (-1)⟩)

def event73089 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨24984⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24983⟩⟩) ⟨22994⟩ 73015)

def event73090 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24984⟩⟩, .relation 73089 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨22994⟩⟩]⟩, (-1)⟩)

def event73091 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24984⟩⟩, .operator (⟨73082, 0⟩, ⟨73018, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩, (1)⟩)

def exact73092RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨22994⟩⟩]⟩, (-1)⟩]

theorem exact73092RawTermsValid :
    exact73092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24984⟩⟩) exact73092RawTerms .large 73085 (.finite 350203613806592) (some (73087))

def event73093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19092⟩⟩) 0 ⟨10670⟩ 3465

def event73094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19092⟩⟩) (.authority (.relationPreimageSource ⟨8⟩))

def exact73095RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩, (1)⟩]

theorem exact73095RawTermsValid :
    exact73095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73095 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19092⟩⟩) exact73095RawTerms (.finite 136065468) 73094 .exactZero (none)

def event73096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19094⟩⟩) 0 ⟨19092⟩ 73095

def event73097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19094⟩⟩) 1 ⟨2348⟩ 4

def event73098 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19094⟩⟩) (.scale (.predecessor 0 73096 .coefficient) (.value (.predecessor 1 73097 .coefficient)))

def exact73099RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩, (1)⟩]

theorem exact73099RawTermsValid :
    exact73099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73099 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19094⟩⟩) exact73099RawTerms (.finite 136065468) 73098 .exactZero (none)

def event73100 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19095⟩⟩) 0 ⟨5535⟩ 65387

def event73101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19095⟩⟩) 1 ⟨19094⟩ 73099

def event73102 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19095⟩⟩) (.product (.predecessor 0 73100 .coefficient) (.predecessor 1 73101 .coefficient) (⟨false, false, none, none, none⟩))

def event73103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19095⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩) [⟨.result 73095 .coefficient, false, none⟩])

def event73104 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19095⟩⟩) (.product (.result 65387 .summary) (.transfer 73103) (⟨false, false, none, none, none⟩))

def event73105 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19095⟩⟩, .operator (⟨65387, 0⟩, ⟨73099, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩, (1)⟩)

def event73106 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19093⟩⟩)

def event73107 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event73108 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event73109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event73110 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event73111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event73112 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event73113 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event73114 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event73115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 73114

def event73116 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 73112

def event73117 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 73115 .coefficient) (.value (.predecessor 1 73116 .coefficient)))

def event73118 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event73119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 73118

def event73120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 73110

def event73121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 73119 .coefficient, .predecessor 1 73120 .coefficient])

def event73122 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event73123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 73122

def event73124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 73108

def event73125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 73124 .coefficient))

def event73126 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event73127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10668⟩⟩) 0 ⟨5530⟩ 73126

def event73128 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10668⟩⟩) (.authority (.programFamilyFact))

def exact73129RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact73129RawTermsValid :
    exact73129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10668⟩⟩) exact73129RawTerms (.finite 3) 73128 .exactZero (none)

def event73130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9500⟩⟩) 0 ⟨5530⟩ 73126

def event73131 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9500⟩⟩) (.authority (.programFamilyFact))

def exact73132RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩], []⟩, (1)⟩]

theorem exact73132RawTermsValid :
    exact73132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73132 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9500⟩⟩) exact73132RawTerms (.finite 3) 73131 .exactZero (none)

def event73133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 0 ⟨9500⟩ 73132

def event73134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 1 ⟨10668⟩ 73129

def event73135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10669⟩⟩) (.product (.predecessor 0 73133 .coefficient) (.predecessor 1 73134 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event73136 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10669⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩) [⟨.result 73132 .coefficient, true, some 1⟩, ⟨.result 73129 .coefficient, true, some 1⟩])

def event73137 : Event := .survivorFold (1) 73136

def exact73138RawTerms : List Term := []

theorem exact73138RawTermsValid :
    exact73138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73138 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10669⟩⟩) exact73138RawTerms (.finite 9) 73135 (.finite 9) (some (73136))

def event73139 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10670⟩⟩) 0 ⟨10669⟩ 73138

def event73140 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.identity (.predecessor 0 73139 .coefficient))

def event73141 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.finite 9)

def event73142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19092⟩⟩) 0 ⟨10670⟩ 73141

def event73143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19092⟩⟩) (.authority (.relationPreimageSource ⟨8⟩))

def exact73144RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩, (1)⟩]

theorem exact73144RawTermsValid :
    exact73144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19092⟩⟩) exact73144RawTerms (.finite 136065468) 73143 .exactZero (none)

def event73145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact73146RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact73146RawTermsValid :
    exact73146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact73146RawTerms .large 73145 .exactZero (none)

def event73147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19093⟩⟩) 0 ⟨6⟩ 73146

def event73148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19093⟩⟩) 1 ⟨19092⟩ 73144

def event73149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19093⟩⟩) (.product (.predecessor 0 73147 .coefficient) (.predecessor 1 73148 .coefficient) (⟨false, false, none, none, none⟩))

def event73150 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19093⟩⟩, .operator (⟨73146, 0⟩, ⟨73144, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩, (1)⟩)

def exact73151RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩, (1)⟩]

theorem exact73151RawTermsValid :
    exact73151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73151 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19093⟩⟩) exact73151RawTerms .large 73149 .exactZero (none)

def event73152 : Event := .preFoldPolynomial 73151 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩, (1)⟩] .exactZero none

def exact73153RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩, (1)⟩]

def event73153 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19093⟩⟩) 73152 exact73153RawTerms .large 73149 .exactZero (none)

def event73154 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨24987⟩⟩)

def event73155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event73156 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event73157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event73158 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event73159 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event73160 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event73161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event73162 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event73163 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 73162

def event73164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 73160

def event73165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 73163 .coefficient) (.value (.predecessor 1 73164 .coefficient)))

def event73166 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event73167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 73166

def event73168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 73158

def event73169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 73167 .coefficient, .predecessor 1 73168 .coefficient])

def event73170 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event73171 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 73170

def event73172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 73156

def event73173 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 73172 .coefficient))

def event73174 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event73175 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10668⟩⟩) 0 ⟨5530⟩ 73174

def event73176 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10668⟩⟩) (.authority (.programFamilyFact))

def exact73177RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact73177RawTermsValid :
    exact73177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73177 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10668⟩⟩) exact73177RawTerms (.finite 3) 73176 .exactZero (none)

def event73178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9500⟩⟩) 0 ⟨5530⟩ 73174

def event73179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9500⟩⟩) (.authority (.programFamilyFact))

def exact73180RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩], []⟩, (1)⟩]

theorem exact73180RawTermsValid :
    exact73180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73180 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9500⟩⟩) exact73180RawTerms (.finite 3) 73179 .exactZero (none)

def event73181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 0 ⟨9500⟩ 73180

def event73182 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 1 ⟨10668⟩ 73177

def event73183 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10669⟩⟩) (.product (.predecessor 0 73181 .coefficient) (.predecessor 1 73182 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event73184 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10669⟩⟩, .operator (⟨73180, 0⟩, ⟨73177, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩)

def exact73185RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact73185RawTermsValid :
    exact73185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73185 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10669⟩⟩) exact73185RawTerms (.finite 9) 73183 .exactZero (none)

def event73186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10670⟩⟩) 0 ⟨10669⟩ 73185

def event73187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.identity (.predecessor 0 73186 .coefficient))

def event73188 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.finite 9)

def event73189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22993⟩⟩) 0 ⟨10670⟩ 73188

def event73190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22993⟩⟩) (.authority (.programFamilyFact))

def event73191 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨22993⟩⟩) (.finite 3720)

def event73192 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event73193 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22994⟩⟩) 0 ⟨6689⟩ 73192

def event73194 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22994⟩⟩) 1 ⟨22993⟩ 73191

def event73195 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22994⟩⟩) (.authority (.operator))

def exact73196RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22994⟩⟩]⟩, (1)⟩]

theorem exact73196RawTermsValid :
    exact73196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73196 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22994⟩⟩) exact73196RawTerms .large 73195 .exactZero (none)

def event73197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24983⟩⟩) 0 ⟨22994⟩ 73196

def event73198 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24983⟩⟩) (.authority (.operator))

def exact73199RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩, (1)⟩]

theorem exact73199RawTermsValid :
    exact73199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73199 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24983⟩⟩) exact73199RawTerms (.finite 8192) 73198 .exactZero (none)

def event73200 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event73201 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event73202 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10768⟩⟩) 0 ⟨10670⟩ 73188

def event73203 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10768⟩⟩) 1 ⟨110⟩ 73201

def event73204 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10768⟩⟩) (.sum [.predecessor 0 73202 .coefficient, .predecessor 1 73203 .coefficient])

def event73205 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10768⟩⟩) (.finite 9)

def event73206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10769⟩⟩) 0 ⟨10768⟩ 73205

def event73207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10769⟩⟩) (.identity (.predecessor 0 73206 .coefficient))

def exact73208RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact73208RawTermsValid :
    exact73208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73208 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10769⟩⟩) exact73208RawTerms (.finite 9) 73207 .exactZero (none)

def event73209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact73210RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73210RawTermsValid :
    exact73210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73210 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact73210RawTerms .large 73209 .exactZero (none)

def event73211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10770⟩⟩) 0 ⟨6544⟩ 73210

def event73212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10770⟩⟩) 1 ⟨10769⟩ 73208

def event73213 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10770⟩⟩) (.product (.predecessor 0 73211 .coefficient) (.predecessor 1 73212 .coefficient) (⟨false, false, none, none, none⟩))

def event73214 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10770⟩⟩, .operator (⟨73210, 0⟩, ⟨73208, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73215RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73215RawTermsValid :
    exact73215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73215 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10770⟩⟩) exact73215RawTerms .large 73213 .exactZero (none)

def eventLeaf4560 : Array AnnotatedEvent := #[
  { event := event72960
    frameStart := 72881 },
  { event := event72961
    frameStart := 72881 },
  { event := event72962
    frameStart := 72881 },
  { event := event72963
    frameStart := 72881 },
  { event := event72964
    frameStart := 72881 },
  { event := event72965
    frameStart := 72881 },
  { event := event72966
    frameStart := 72881 },
  { event := event72967
    frameStart := 72881 },
  { event := event72968
    frameStart := 72881 },
  { event := event72969
    frameStart := 72881 },
  { event := event72970
    frameStart := 72881 },
  { event := event72971
    frameStart := 72881 },
  { event := event72972
    frameStart := 72881 },
  { event := event72973
    frameStart := 72881 },
  { event := event72974
    frameStart := 72881 },
  { event := event72975
    frameStart := 72881 }
]

def eventLeaf4561 : Array AnnotatedEvent := #[
  { event := event72976
    frameStart := 72881 },
  { event := event72977
    frameStart := 72881 },
  { event := event72978
    frameStart := 72881 },
  { event := event72979
    frameStart := 72881 },
  { event := event72980
    frameStart := 72881 },
  { event := event72981
    frameStart := 72881 },
  { event := event72982
    frameStart := 72881 },
  { event := event72983
    frameStart := 72881 },
  { event := event72984
    frameStart := 72881 },
  { event := event72985
    frameStart := 0 },
  { event := event72986
    frameStart := 0 },
  { event := event72987
    frameStart := 0 },
  { event := event72988
    frameStart := 0 },
  { event := event72989
    frameStart := 0 },
  { event := event72990
    frameStart := 0 },
  { event := event72991
    frameStart := 0 }
]

def eventLeaf4562 : Array AnnotatedEvent := #[
  { event := event72992
    frameStart := 0 },
  { event := event72993
    frameStart := 0 },
  { event := event72994
    frameStart := 0 },
  { event := event72995
    frameStart := 0 },
  { event := event72996
    frameStart := 0 },
  { event := event72997
    frameStart := 0 },
  { event := event72998
    frameStart := 0 },
  { event := event72999
    frameStart := 0 },
  { event := event73000
    frameStart := 0 },
  { event := event73001
    frameStart := 0 },
  { event := event73002
    frameStart := 0 },
  { event := event73003
    frameStart := 0 },
  { event := event73004
    frameStart := 0 },
  { event := event73005
    frameStart := 0 },
  { event := event73006
    frameStart := 0 },
  { event := event73007
    frameStart := 0 }
]

def eventLeaf4563 : Array AnnotatedEvent := #[
  { event := event73008
    frameStart := 0 },
  { event := event73009
    frameStart := 0 },
  { event := event73010
    frameStart := 0 },
  { event := event73011
    frameStart := 0 },
  { event := event73012
    frameStart := 0 },
  { event := event73013
    frameStart := 0 },
  { event := event73014
    frameStart := 0 },
  { event := event73015
    frameStart := 0 },
  { event := event73016
    frameStart := 0 },
  { event := event73017
    frameStart := 0 },
  { event := event73018
    frameStart := 0 },
  { event := event73019
    frameStart := 0 },
  { event := event73020
    frameStart := 0 },
  { event := event73021
    frameStart := 0 },
  { event := event73022
    frameStart := 0 },
  { event := event73023
    frameStart := 0 }
]

def eventLeaf4564 : Array AnnotatedEvent := #[
  { event := event73024
    frameStart := 0 },
  { event := event73025
    frameStart := 0 },
  { event := event73026
    frameStart := 0 },
  { event := event73027
    frameStart := 0 },
  { event := event73028
    frameStart := 0 },
  { event := event73029
    frameStart := 0 },
  { event := event73030
    frameStart := 0 },
  { event := event73031
    frameStart := 0 },
  { event := event73032
    frameStart := 0 },
  { event := event73033
    frameStart := 0 },
  { event := event73034
    frameStart := 0 },
  { event := event73035
    frameStart := 0 },
  { event := event73036
    frameStart := 0 },
  { event := event73037
    frameStart := 0 },
  { event := event73038
    frameStart := 0 },
  { event := event73039
    frameStart := 0 }
]

def eventLeaf4565 : Array AnnotatedEvent := #[
  { event := event73040
    frameStart := 0 },
  { event := event73041
    frameStart := 0 },
  { event := event73042
    frameStart := 0 },
  { event := event73043
    frameStart := 0 },
  { event := event73044
    frameStart := 0 },
  { event := event73045
    frameStart := 0 },
  { event := event73046
    frameStart := 0 },
  { event := event73047
    frameStart := 0 },
  { event := event73048
    frameStart := 0 },
  { event := event73049
    frameStart := 0 },
  { event := event73050
    frameStart := 0 },
  { event := event73051
    frameStart := 0 },
  { event := event73052
    frameStart := 0 },
  { event := event73053
    frameStart := 0 },
  { event := event73054
    frameStart := 0 },
  { event := event73055
    frameStart := 0 }
]

def eventLeaf4566 : Array AnnotatedEvent := #[
  { event := event73056
    frameStart := 0 },
  { event := event73057
    frameStart := 0 },
  { event := event73058
    frameStart := 0 },
  { event := event73059
    frameStart := 0 },
  { event := event73060
    frameStart := 0 },
  { event := event73061
    frameStart := 0 },
  { event := event73062
    frameStart := 0 },
  { event := event73063
    frameStart := 0 },
  { event := event73064
    frameStart := 0 },
  { event := event73065
    frameStart := 0 },
  { event := event73066
    frameStart := 0 },
  { event := event73067
    frameStart := 0 },
  { event := event73068
    frameStart := 0 },
  { event := event73069
    frameStart := 0 },
  { event := event73070
    frameStart := 0 },
  { event := event73071
    frameStart := 0 }
]

def eventLeaf4567 : Array AnnotatedEvent := #[
  { event := event73072
    frameStart := 0 },
  { event := event73073
    frameStart := 0 },
  { event := event73074
    frameStart := 0 },
  { event := event73075
    frameStart := 0 },
  { event := event73076
    frameStart := 0 },
  { event := event73077
    frameStart := 0 },
  { event := event73078
    frameStart := 0 },
  { event := event73079
    frameStart := 0 },
  { event := event73080
    frameStart := 0 },
  { event := event73081
    frameStart := 0 },
  { event := event73082
    frameStart := 0 },
  { event := event73083
    frameStart := 0 },
  { event := event73084
    frameStart := 0 },
  { event := event73085
    frameStart := 0 },
  { event := event73086
    frameStart := 0 },
  { event := event73087
    frameStart := 0 }
]

def eventLeaf4568 : Array AnnotatedEvent := #[
  { event := event73088
    frameStart := 0 },
  { event := event73089
    frameStart := 0 },
  { event := event73090
    frameStart := 0 },
  { event := event73091
    frameStart := 0 },
  { event := event73092
    frameStart := 0 },
  { event := event73093
    frameStart := 0 },
  { event := event73094
    frameStart := 0 },
  { event := event73095
    frameStart := 0 },
  { event := event73096
    frameStart := 0 },
  { event := event73097
    frameStart := 0 },
  { event := event73098
    frameStart := 0 },
  { event := event73099
    frameStart := 0 },
  { event := event73100
    frameStart := 0 },
  { event := event73101
    frameStart := 0 },
  { event := event73102
    frameStart := 0 },
  { event := event73103
    frameStart := 0 }
]

def eventLeaf4569 : Array AnnotatedEvent := #[
  { event := event73104
    frameStart := 0 },
  { event := event73105
    frameStart := 0 },
  { event := event73106
    frameStart := 73106 },
  { event := event73107
    frameStart := 73106 },
  { event := event73108
    frameStart := 73106 },
  { event := event73109
    frameStart := 73106 },
  { event := event73110
    frameStart := 73106 },
  { event := event73111
    frameStart := 73106 },
  { event := event73112
    frameStart := 73106 },
  { event := event73113
    frameStart := 73106 },
  { event := event73114
    frameStart := 73106 },
  { event := event73115
    frameStart := 73106 },
  { event := event73116
    frameStart := 73106 },
  { event := event73117
    frameStart := 73106 },
  { event := event73118
    frameStart := 73106 },
  { event := event73119
    frameStart := 73106 }
]

def eventLeaf4570 : Array AnnotatedEvent := #[
  { event := event73120
    frameStart := 73106 },
  { event := event73121
    frameStart := 73106 },
  { event := event73122
    frameStart := 73106 },
  { event := event73123
    frameStart := 73106 },
  { event := event73124
    frameStart := 73106 },
  { event := event73125
    frameStart := 73106 },
  { event := event73126
    frameStart := 73106 },
  { event := event73127
    frameStart := 73106 },
  { event := event73128
    frameStart := 73106 },
  { event := event73129
    frameStart := 73106 },
  { event := event73130
    frameStart := 73106 },
  { event := event73131
    frameStart := 73106 },
  { event := event73132
    frameStart := 73106 },
  { event := event73133
    frameStart := 73106 },
  { event := event73134
    frameStart := 73106 },
  { event := event73135
    frameStart := 73106 }
]

def eventLeaf4571 : Array AnnotatedEvent := #[
  { event := event73136
    frameStart := 73106 },
  { event := event73137
    frameStart := 73106 },
  { event := event73138
    frameStart := 73106 },
  { event := event73139
    frameStart := 73106 },
  { event := event73140
    frameStart := 73106 },
  { event := event73141
    frameStart := 73106 },
  { event := event73142
    frameStart := 73106 },
  { event := event73143
    frameStart := 73106 },
  { event := event73144
    frameStart := 73106 },
  { event := event73145
    frameStart := 73106 },
  { event := event73146
    frameStart := 73106 },
  { event := event73147
    frameStart := 73106 },
  { event := event73148
    frameStart := 73106 },
  { event := event73149
    frameStart := 73106 },
  { event := event73150
    frameStart := 73106 },
  { event := event73151
    frameStart := 73106 }
]

def eventLeaf4572 : Array AnnotatedEvent := #[
  { event := event73152
    frameStart := 73106 },
  { event := event73153
    frameStart := 73106 },
  { event := event73154
    frameStart := 73154 },
  { event := event73155
    frameStart := 73154 },
  { event := event73156
    frameStart := 73154 },
  { event := event73157
    frameStart := 73154 },
  { event := event73158
    frameStart := 73154 },
  { event := event73159
    frameStart := 73154 },
  { event := event73160
    frameStart := 73154 },
  { event := event73161
    frameStart := 73154 },
  { event := event73162
    frameStart := 73154 },
  { event := event73163
    frameStart := 73154 },
  { event := event73164
    frameStart := 73154 },
  { event := event73165
    frameStart := 73154 },
  { event := event73166
    frameStart := 73154 },
  { event := event73167
    frameStart := 73154 }
]

def eventLeaf4573 : Array AnnotatedEvent := #[
  { event := event73168
    frameStart := 73154 },
  { event := event73169
    frameStart := 73154 },
  { event := event73170
    frameStart := 73154 },
  { event := event73171
    frameStart := 73154 },
  { event := event73172
    frameStart := 73154 },
  { event := event73173
    frameStart := 73154 },
  { event := event73174
    frameStart := 73154 },
  { event := event73175
    frameStart := 73154 },
  { event := event73176
    frameStart := 73154 },
  { event := event73177
    frameStart := 73154 },
  { event := event73178
    frameStart := 73154 },
  { event := event73179
    frameStart := 73154 },
  { event := event73180
    frameStart := 73154 },
  { event := event73181
    frameStart := 73154 },
  { event := event73182
    frameStart := 73154 },
  { event := event73183
    frameStart := 73154 }
]

def eventLeaf4574 : Array AnnotatedEvent := #[
  { event := event73184
    frameStart := 73154 },
  { event := event73185
    frameStart := 73154 },
  { event := event73186
    frameStart := 73154 },
  { event := event73187
    frameStart := 73154 },
  { event := event73188
    frameStart := 73154 },
  { event := event73189
    frameStart := 73154 },
  { event := event73190
    frameStart := 73154 },
  { event := event73191
    frameStart := 73154 },
  { event := event73192
    frameStart := 73154 },
  { event := event73193
    frameStart := 73154 },
  { event := event73194
    frameStart := 73154 },
  { event := event73195
    frameStart := 73154 },
  { event := event73196
    frameStart := 73154 },
  { event := event73197
    frameStart := 73154 },
  { event := event73198
    frameStart := 73154 },
  { event := event73199
    frameStart := 73154 }
]

def eventLeaf4575 : Array AnnotatedEvent := #[
  { event := event73200
    frameStart := 73154 },
  { event := event73201
    frameStart := 73154 },
  { event := event73202
    frameStart := 73154 },
  { event := event73203
    frameStart := 73154 },
  { event := event73204
    frameStart := 73154 },
  { event := event73205
    frameStart := 73154 },
  { event := event73206
    frameStart := 73154 },
  { event := event73207
    frameStart := 73154 },
  { event := event73208
    frameStart := 73154 },
  { event := event73209
    frameStart := 73154 },
  { event := event73210
    frameStart := 73154 },
  { event := event73211
    frameStart := 73154 },
  { event := event73212
    frameStart := 73154 },
  { event := event73213
    frameStart := 73154 },
  { event := event73214
    frameStart := 73154 },
  { event := event73215
    frameStart := 73154 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events285
