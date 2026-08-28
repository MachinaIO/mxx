import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events305

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event78080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15895⟩⟩) 0 ⟨6696⟩ 78079

def event78081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15895⟩⟩) 1 ⟨15894⟩ 78076

def event78082 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15895⟩⟩) (.sum [.predecessor 0 78080 .coefficient, .predecessor 1 78081 .coefficient])

def exact78083RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78083RawTermsValid :
    exact78083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78083 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15895⟩⟩) exact78083RawTerms .large 78082 .exactZero (none)

def event78084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27630⟩⟩) 0 ⟨15895⟩ 78083

def event78085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27630⟩⟩) 1 ⟨27629⟩ 78060

def event78086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27630⟩⟩) (.product (.predecessor 0 78084 .coefficient) (.predecessor 1 78085 .coefficient) (⟨false, false, none, none, none⟩))

def event78087 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27630⟩⟩, .operator (⟨78083, 0⟩, ⟨78060, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (1)⟩)

def event78088 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27630⟩⟩, .operator (⟨78083, 1⟩, ⟨78060, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (-1)⟩)

def event78089 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27630⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27629⟩⟩) ⟨24095⟩ 78057)

def event78090 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27630⟩⟩, .relation 78089 0, ⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (-1)⟩)

def exact78091RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (-1)⟩]

theorem exact78091RawTermsValid :
    exact78091RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78091 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27630⟩⟩) exact78091RawTerms .large 78086 .exactZero (none)

def event78092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17217⟩⟩) 0 ⟨15818⟩ 78049

def event78093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17217⟩⟩) (.authority (.programFamilyFact))

def exact78094RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17217⟩⟩], []⟩, (1)⟩]

theorem exact78094RawTermsValid :
    exact78094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78094 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17217⟩⟩) exact78094RawTerms (.finite 16) 78093 .exactZero (none)

def event78095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17219⟩⟩) 0 ⟨6544⟩ 78071

def event78096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17219⟩⟩) 1 ⟨17217⟩ 78094

def event78097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17219⟩⟩) (.product (.predecessor 0 78095 .coefficient) (.predecessor 1 78096 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78098 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17219⟩⟩, .operator (⟨78071, 0⟩, ⟨78094, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78099RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78099RawTermsValid :
    exact78099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78099 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17219⟩⟩) exact78099RawTerms .large 78097 .exactZero (none)

def event78100 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6720⟩⟩) 0 ⟨6689⟩ 78053

def event78101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6720⟩⟩) (.authority (.operator))

def exact78102RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩]

theorem exact78102RawTermsValid :
    exact78102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78102 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6720⟩⟩) exact78102RawTerms .large 78101 .exactZero (none)

def event78103 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17220⟩⟩) 0 ⟨6720⟩ 78102

def event78104 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17220⟩⟩) 1 ⟨17219⟩ 78099

def event78105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17220⟩⟩) (.sum [.predecessor 0 78103 .coefficient, .predecessor 1 78104 .coefficient])

def exact78106RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78106RawTermsValid :
    exact78106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17220⟩⟩) exact78106RawTerms .large 78105 .exactZero (none)

def event78107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27635⟩⟩) 0 ⟨17220⟩ 78106

def event78108 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27635⟩⟩) 1 ⟨27630⟩ 78091

def event78109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27635⟩⟩) (.sum [.predecessor 0 78107 .coefficient, .predecessor 1 78108 .coefficient])

def exact78110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78110RawTermsValid :
    exact78110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78110 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27635⟩⟩) exact78110RawTerms .large 78109 .exactZero (none)

def event78111 : Event := .preFoldPolynomial 78110 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event78112 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27635⟩⟩) 78111 exact78112RawTerms .large 78109 .exactZero (none)

def event78113 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15818⟩⟩) ⟨⟨133⟩, ⟨40⟩, ⟨109⟩⟩ ⟨77955, 78113⟩

def event78114 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21183⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩) (1) 0 2 (.universal 78113 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩) (none) 78112)

def event78115 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21183⟩⟩, .relation 78114 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩)

def event78116 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21183⟩⟩, .relation 78114 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (-1)⟩)

def event78117 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21183⟩⟩, .relation 78114 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (1)⟩)

def event78118 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21183⟩⟩, .relation 78114 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78119RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78119RawTermsValid :
    exact78119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78119 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21183⟩⟩) exact78119RawTerms .large 77951 (.finite 1811303510016) (some (77953))

def event78120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27632⟩⟩) 0 ⟨21183⟩ 78119

def event78121 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27632⟩⟩) 1 ⟨27631⟩ 77941

def event78122 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27632⟩⟩) (.sum [.predecessor 0 78120 .coefficient, .predecessor 1 78121 .coefficient])

def event78123 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27632⟩⟩, .operator (⟨78119, 0⟩, ⟨77941, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (1)⟩)

def event78124 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27632⟩⟩, .operator (⟨78119, 2⟩, ⟨77941, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (-1)⟩)

def event78125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27632⟩⟩) (.sum [.result 78119 .summary, .result 77941 .summary])

def exact78126RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78126RawTermsValid :
    exact78126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78126 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27632⟩⟩) exact78126RawTerms .large 78122 (.finite 1292046061494565744640) (some (78125))

def event78127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27633⟩⟩) 0 ⟨27632⟩ 78126

def event78128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27633⟩⟩) 1 ⟨6644⟩ 5739

def event78129 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27633⟩⟩) (.product (.predecessor 0 78127 .coefficient) (.predecessor 1 78128 .coefficient) (⟨false, false, none, none, none⟩))

def event78130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27633⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) [⟨.result 5735 .coefficient, false, none⟩])

def event78131 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27633⟩⟩) (.product (.result 78126 .summary) (.transfer 78130) (⟨false, false, none, none, none⟩))

def event78132 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27633⟩⟩, .operator (⟨78126, 0⟩, ⟨5739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩)

def event78133 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27633⟩⟩, .operator (⟨78126, 1⟩, ⟨5739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (-1)⟩)

def event78134 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27633⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6643⟩⟩) ⟨6593⟩ 5732)

def event78135 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27633⟩⟩, .relation 78134 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78136RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78136RawTermsValid :
    exact78136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78136 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27633⟩⟩) exact78136RawTerms .large 78129 (.finite 4741829718422040195880714240) (some (78131))

def event78137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24032⟩⟩) 0 ⟨6689⟩ 5477

def event78138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24032⟩⟩) 1 ⟨24031⟩ 71073

def event78139 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24032⟩⟩) (.authority (.operator))

def exact78140RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (1)⟩]

theorem exact78140RawTermsValid :
    exact78140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78140 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24032⟩⟩) exact78140RawTerms .large 78139 .exactZero (none)

def event78141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27412⟩⟩) 0 ⟨24032⟩ 78140

def event78142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27412⟩⟩) (.authority (.operator))

def exact78143RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (1)⟩]

theorem exact78143RawTermsValid :
    exact78143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78143 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27412⟩⟩) exact78143RawTerms (.finite 8192) 78142 .exactZero (none)

def event78144 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27414⟩⟩) 0 ⟨25909⟩ 71357

def event78145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27414⟩⟩) 1 ⟨27412⟩ 78143

def event78146 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27414⟩⟩) (.product (.predecessor 0 78144 .coefficient) (.predecessor 1 78145 .coefficient) (⟨false, false, none, none, none⟩))

def event78147 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27414⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩) [⟨.result 78143 .coefficient, false, none⟩])

def event78148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27414⟩⟩) (.product (.result 71357 .summary) (.transfer 78147) (⟨false, false, none, none, none⟩))

def event78149 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27414⟩⟩, .operator (⟨71357, 0⟩, ⟨78143, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (1)⟩)

def event78150 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27414⟩⟩, .operator (⟨71357, 1⟩, ⟨78143, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (-1)⟩)

def event78151 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27414⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27412⟩⟩) ⟨24032⟩ 78140)

def event78152 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27414⟩⟩, .relation 78151 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (-1)⟩)

def exact78153RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (-1)⟩]

theorem exact78153RawTermsValid :
    exact78153RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78153 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27414⟩⟩) exact78153RawTerms .large 78146 (.finite 1292001234793221062656) (some (78148))

def event78154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21036⟩⟩) 0 ⟨15699⟩ 3379

def event78155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21036⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact78156RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩, (1)⟩]

theorem exact78156RawTermsValid :
    exact78156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78156 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21036⟩⟩) exact78156RawTerms (.finite 136065468) 78155 .exactZero (none)

def event78157 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21038⟩⟩) 0 ⟨21036⟩ 78156

def event78158 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21038⟩⟩) 1 ⟨2348⟩ 4

def event78159 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21038⟩⟩) (.scale (.predecessor 0 78157 .coefficient) (.value (.predecessor 1 78158 .coefficient)))

def exact78160RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩, (1)⟩]

theorem exact78160RawTermsValid :
    exact78160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78160 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21038⟩⟩) exact78160RawTerms (.finite 136065468) 78159 .exactZero (none)

def event78161 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21039⟩⟩) 0 ⟨5535⟩ 65387

def event78162 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21039⟩⟩) 1 ⟨21038⟩ 78160

def event78163 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21039⟩⟩) (.product (.predecessor 0 78161 .coefficient) (.predecessor 1 78162 .coefficient) (⟨false, false, none, none, none⟩))

def event78164 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21039⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩) [⟨.result 78156 .coefficient, false, none⟩])

def event78165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21039⟩⟩) (.product (.result 65387 .summary) (.transfer 78164) (⟨false, false, none, none, none⟩))

def event78166 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21039⟩⟩, .operator (⟨65387, 0⟩, ⟨78160, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩, (1)⟩)

def event78167 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21037⟩⟩)

def event78168 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78169 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78171 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78172 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78173 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78175 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78175

def event78177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78173

def event78178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78176 .coefficient) (.value (.predecessor 1 78177 .coefficient)))

def event78179 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78180 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78179

def event78181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78171

def event78182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78180 .coefficient, .predecessor 1 78181 .coefficient])

def event78183 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78184 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78183

def event78185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78169

def event78186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78185 .coefficient))

def event78187 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11297⟩⟩) 0 ⟨5530⟩ 78187

def event78189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11297⟩⟩) (.authority (.programFamilyFact))

def exact78190RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩], []⟩, (1)⟩]

theorem exact78190RawTermsValid :
    exact78190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78190 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11297⟩⟩) exact78190RawTerms (.finite 12) 78189 .exactZero (none)

def event78191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13764⟩⟩) 0 ⟨5530⟩ 78187

def event78192 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13764⟩⟩) (.authority (.programFamilyFact))

def exact78193RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact78193RawTermsValid :
    exact78193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78193 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13764⟩⟩) exact78193RawTerms (.finite 12) 78192 .exactZero (none)

def event78194 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 0 ⟨13764⟩ 78193

def event78195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 1 ⟨11297⟩ 78190

def event78196 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.product (.predecessor 0 78194 .coefficient) (.predecessor 1 78195 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩) [⟨.result 78193 .coefficient, true, some 1⟩, ⟨.result 78190 .coefficient, true, some 1⟩])

def event78198 : Event := .survivorFold (1) 78197

def exact78199RawTerms : List Term := []

theorem exact78199RawTermsValid :
    exact78199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78199 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13765⟩⟩) exact78199RawTerms (.finite 144) 78196 (.finite 144) (some (78197))

def event78200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13766⟩⟩) 0 ⟨13765⟩ 78199

def event78201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.identity (.predecessor 0 78200 .coefficient))

def event78202 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.finite 144)

def event78203 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15698⟩⟩) 0 ⟨13766⟩ 78202

def event78204 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15698⟩⟩) (.authority (.programFamilyFact))

def exact78205RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], []⟩, (1)⟩]

theorem exact78205RawTermsValid :
    exact78205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78205 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15698⟩⟩) exact78205RawTerms (.finite 12) 78204 .exactZero (none)

def event78206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15699⟩⟩) 0 ⟨15698⟩ 78205

def event78207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.identity (.predecessor 0 78206 .coefficient))

def event78208 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.finite 12)

def event78209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21036⟩⟩) 0 ⟨15699⟩ 78208

def event78210 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21036⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact78211RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩, (1)⟩]

theorem exact78211RawTermsValid :
    exact78211RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78211 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21036⟩⟩) exact78211RawTerms (.finite 136065468) 78210 .exactZero (none)

def event78212 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact78213RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact78213RawTermsValid :
    exact78213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78213 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact78213RawTerms .large 78212 .exactZero (none)

def event78214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21037⟩⟩) 0 ⟨6⟩ 78213

def event78215 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21037⟩⟩) 1 ⟨21036⟩ 78211

def event78216 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21037⟩⟩) (.product (.predecessor 0 78214 .coefficient) (.predecessor 1 78215 .coefficient) (⟨false, false, none, none, none⟩))

def event78217 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21037⟩⟩, .operator (⟨78213, 0⟩, ⟨78211, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩, (1)⟩)

def exact78218RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩, (1)⟩]

theorem exact78218RawTermsValid :
    exact78218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21037⟩⟩) exact78218RawTerms .large 78216 .exactZero (none)

def event78219 : Event := .preFoldPolynomial 78218 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩, (1)⟩] .exactZero none

def exact78220RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩, (1)⟩]

def event78220 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21037⟩⟩) 78219 exact78220RawTerms .large 78216 .exactZero (none)

def event78221 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27418⟩⟩)

def event78222 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78223 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78224 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78225 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78226 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78227 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78228 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78229 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78229

def event78231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78227

def event78232 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78230 .coefficient) (.value (.predecessor 1 78231 .coefficient)))

def event78233 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78234 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78233

def event78235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78225

def event78236 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78234 .coefficient, .predecessor 1 78235 .coefficient])

def event78237 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78238 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78237

def event78239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78223

def event78240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78239 .coefficient))

def event78241 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11297⟩⟩) 0 ⟨5530⟩ 78241

def event78243 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11297⟩⟩) (.authority (.programFamilyFact))

def exact78244RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩], []⟩, (1)⟩]

theorem exact78244RawTermsValid :
    exact78244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78244 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11297⟩⟩) exact78244RawTerms (.finite 12) 78243 .exactZero (none)

def event78245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13764⟩⟩) 0 ⟨5530⟩ 78241

def event78246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13764⟩⟩) (.authority (.programFamilyFact))

def exact78247RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact78247RawTermsValid :
    exact78247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78247 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13764⟩⟩) exact78247RawTerms (.finite 12) 78246 .exactZero (none)

def event78248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 0 ⟨13764⟩ 78247

def event78249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 1 ⟨11297⟩ 78244

def event78250 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.product (.predecessor 0 78248 .coefficient) (.predecessor 1 78249 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78251 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13765⟩⟩, .operator (⟨78247, 0⟩, ⟨78244, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩)

def exact78252RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact78252RawTermsValid :
    exact78252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78252 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13765⟩⟩) exact78252RawTerms (.finite 144) 78250 .exactZero (none)

def event78253 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13766⟩⟩) 0 ⟨13765⟩ 78252

def event78254 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.identity (.predecessor 0 78253 .coefficient))

def event78255 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.finite 144)

def event78256 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15698⟩⟩) 0 ⟨13766⟩ 78255

def event78257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15698⟩⟩) (.authority (.programFamilyFact))

def exact78258RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], []⟩, (1)⟩]

theorem exact78258RawTermsValid :
    exact78258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78258 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15698⟩⟩) exact78258RawTerms (.finite 12) 78257 .exactZero (none)

def event78259 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15699⟩⟩) 0 ⟨15698⟩ 78258

def event78260 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.identity (.predecessor 0 78259 .coefficient))

def event78261 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.finite 12)

def event78262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24031⟩⟩) 0 ⟨15699⟩ 78261

def event78263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24031⟩⟩) (.authority (.programFamilyFact))

def event78264 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24031⟩⟩) (.finite 3720)

def event78265 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event78266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24032⟩⟩) 0 ⟨6689⟩ 78265

def event78267 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24032⟩⟩) 1 ⟨24031⟩ 78264

def event78268 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24032⟩⟩) (.authority (.operator))

def exact78269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (1)⟩]

theorem exact78269RawTermsValid :
    exact78269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78269 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24032⟩⟩) exact78269RawTerms .large 78268 .exactZero (none)

def event78270 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27412⟩⟩) 0 ⟨24032⟩ 78269

def event78271 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27412⟩⟩) (.authority (.operator))

def exact78272RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (1)⟩]

theorem exact78272RawTermsValid :
    exact78272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78272 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27412⟩⟩) exact78272RawTerms (.finite 8192) 78271 .exactZero (none)

def event78273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event78274 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event78275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15773⟩⟩) 0 ⟨15699⟩ 78261

def event78276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15773⟩⟩) 1 ⟨110⟩ 78274

def event78277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15773⟩⟩) (.sum [.predecessor 0 78275 .coefficient, .predecessor 1 78276 .coefficient])

def event78278 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15773⟩⟩) (.finite 12)

def event78279 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15774⟩⟩) 0 ⟨15773⟩ 78278

def event78280 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15774⟩⟩) (.identity (.predecessor 0 78279 .coefficient))

def exact78281RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], []⟩, (1)⟩]

theorem exact78281RawTermsValid :
    exact78281RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78281 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15774⟩⟩) exact78281RawTerms (.finite 12) 78280 .exactZero (none)

def event78282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact78283RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78283RawTermsValid :
    exact78283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78283 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact78283RawTerms .large 78282 .exactZero (none)

def event78284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15775⟩⟩) 0 ⟨6544⟩ 78283

def event78285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15775⟩⟩) 1 ⟨15774⟩ 78281

def event78286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15775⟩⟩) (.product (.predecessor 0 78284 .coefficient) (.predecessor 1 78285 .coefficient) (⟨false, false, none, none, none⟩))

def event78287 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15775⟩⟩, .operator (⟨78283, 0⟩, ⟨78281, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78288RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78288RawTermsValid :
    exact78288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78288 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15775⟩⟩) exact78288RawTerms .large 78286 .exactZero (none)

def event78289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6695⟩⟩) 0 ⟨6689⟩ 78265

def event78290 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6695⟩⟩) (.authority (.operator))

def exact78291RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩]

theorem exact78291RawTermsValid :
    exact78291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78291 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6695⟩⟩) exact78291RawTerms .large 78290 .exactZero (none)

def event78292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15776⟩⟩) 0 ⟨6695⟩ 78291

def event78293 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15776⟩⟩) 1 ⟨15775⟩ 78288

def event78294 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15776⟩⟩) (.sum [.predecessor 0 78292 .coefficient, .predecessor 1 78293 .coefficient])

def exact78295RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78295RawTermsValid :
    exact78295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78295 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15776⟩⟩) exact78295RawTerms .large 78294 .exactZero (none)

def event78296 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27413⟩⟩) 0 ⟨15776⟩ 78295

def event78297 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27413⟩⟩) 1 ⟨27412⟩ 78272

def event78298 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27413⟩⟩) (.product (.predecessor 0 78296 .coefficient) (.predecessor 1 78297 .coefficient) (⟨false, false, none, none, none⟩))

def event78299 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27413⟩⟩, .operator (⟨78295, 0⟩, ⟨78272, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (1)⟩)

def event78300 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27413⟩⟩, .operator (⟨78295, 1⟩, ⟨78272, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (-1)⟩)

def event78301 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27413⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27412⟩⟩) ⟨24032⟩ 78269)

def event78302 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27413⟩⟩, .relation 78301 0, ⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (-1)⟩)

def exact78303RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (-1)⟩]

theorem exact78303RawTermsValid :
    exact78303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78303 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27413⟩⟩) exact78303RawTerms .large 78298 .exactZero (none)

def event78304 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17434⟩⟩) 0 ⟨15699⟩ 78261

def event78305 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17434⟩⟩) (.authority (.programFamilyFact))

def exact78306RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17434⟩⟩], []⟩, (1)⟩]

theorem exact78306RawTermsValid :
    exact78306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78306 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17434⟩⟩) exact78306RawTerms (.finite 12) 78305 .exactZero (none)

def event78307 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17436⟩⟩) 0 ⟨6544⟩ 78283

def event78308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17436⟩⟩) 1 ⟨17434⟩ 78306

def event78309 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17436⟩⟩) (.product (.predecessor 0 78307 .coefficient) (.predecessor 1 78308 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78310 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17436⟩⟩, .operator (⟨78283, 0⟩, ⟨78306, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78311RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78311RawTermsValid :
    exact78311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78311 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17436⟩⟩) exact78311RawTerms .large 78309 .exactZero (none)

def event78312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6718⟩⟩) 0 ⟨6689⟩ 78265

def event78313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6718⟩⟩) (.authority (.operator))

def exact78314RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩]

theorem exact78314RawTermsValid :
    exact78314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78314 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6718⟩⟩) exact78314RawTerms .large 78313 .exactZero (none)

def event78315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17437⟩⟩) 0 ⟨6718⟩ 78314

def event78316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17437⟩⟩) 1 ⟨17436⟩ 78311

def event78317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17437⟩⟩) (.sum [.predecessor 0 78315 .coefficient, .predecessor 1 78316 .coefficient])

def exact78318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78318RawTermsValid :
    exact78318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78318 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17437⟩⟩) exact78318RawTerms .large 78317 .exactZero (none)

def event78319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27418⟩⟩) 0 ⟨17437⟩ 78318

def event78320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27418⟩⟩) 1 ⟨27413⟩ 78303

def event78321 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27418⟩⟩) (.sum [.predecessor 0 78319 .coefficient, .predecessor 1 78320 .coefficient])

def exact78322RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78322RawTermsValid :
    exact78322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78322 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27418⟩⟩) exact78322RawTerms .large 78321 .exactZero (none)

def event78323 : Event := .preFoldPolynomial 78322 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78324RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event78324 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27418⟩⟩) 78323 exact78324RawTerms .large 78321 .exactZero (none)

def event78325 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15699⟩⟩) ⟨⟨131⟩, ⟨38⟩, ⟨109⟩⟩ ⟨78167, 78325⟩

def event78326 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21039⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩) (1) 0 2 (.universal 78325 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩) (none) 78324)

def event78327 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21039⟩⟩, .relation 78326 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩)

def event78328 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21039⟩⟩, .relation 78326 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (-1)⟩)

def event78329 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21039⟩⟩, .relation 78326 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (1)⟩)

def event78330 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21039⟩⟩, .relation 78326 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78331RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78331RawTermsValid :
    exact78331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78331 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21039⟩⟩) exact78331RawTerms .large 78163 (.finite 1811303510016) (some (78165))

def event78332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27415⟩⟩) 0 ⟨21039⟩ 78331

def event78333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27415⟩⟩) 1 ⟨27414⟩ 78153

def event78334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27415⟩⟩) (.sum [.predecessor 0 78332 .coefficient, .predecessor 1 78333 .coefficient])

def event78335 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27415⟩⟩, .operator (⟨78331, 0⟩, ⟨78153, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩, (1)⟩)

def eventLeaf4880 : Array AnnotatedEvent := #[
  { event := event78080
    frameStart := 78009 },
  { event := event78081
    frameStart := 78009 },
  { event := event78082
    frameStart := 78009 },
  { event := event78083
    frameStart := 78009 },
  { event := event78084
    frameStart := 78009 },
  { event := event78085
    frameStart := 78009 },
  { event := event78086
    frameStart := 78009 },
  { event := event78087
    frameStart := 78009 },
  { event := event78088
    frameStart := 78009 },
  { event := event78089
    frameStart := 78009 },
  { event := event78090
    frameStart := 78009 },
  { event := event78091
    frameStart := 78009 },
  { event := event78092
    frameStart := 78009 },
  { event := event78093
    frameStart := 78009 },
  { event := event78094
    frameStart := 78009 },
  { event := event78095
    frameStart := 78009 }
]

def eventLeaf4881 : Array AnnotatedEvent := #[
  { event := event78096
    frameStart := 78009 },
  { event := event78097
    frameStart := 78009 },
  { event := event78098
    frameStart := 78009 },
  { event := event78099
    frameStart := 78009 },
  { event := event78100
    frameStart := 78009 },
  { event := event78101
    frameStart := 78009 },
  { event := event78102
    frameStart := 78009 },
  { event := event78103
    frameStart := 78009 },
  { event := event78104
    frameStart := 78009 },
  { event := event78105
    frameStart := 78009 },
  { event := event78106
    frameStart := 78009 },
  { event := event78107
    frameStart := 78009 },
  { event := event78108
    frameStart := 78009 },
  { event := event78109
    frameStart := 78009 },
  { event := event78110
    frameStart := 78009 },
  { event := event78111
    frameStart := 78009 }
]

def eventLeaf4882 : Array AnnotatedEvent := #[
  { event := event78112
    frameStart := 78009 },
  { event := event78113
    frameStart := 0 },
  { event := event78114
    frameStart := 0 },
  { event := event78115
    frameStart := 0 },
  { event := event78116
    frameStart := 0 },
  { event := event78117
    frameStart := 0 },
  { event := event78118
    frameStart := 0 },
  { event := event78119
    frameStart := 0 },
  { event := event78120
    frameStart := 0 },
  { event := event78121
    frameStart := 0 },
  { event := event78122
    frameStart := 0 },
  { event := event78123
    frameStart := 0 },
  { event := event78124
    frameStart := 0 },
  { event := event78125
    frameStart := 0 },
  { event := event78126
    frameStart := 0 },
  { event := event78127
    frameStart := 0 }
]

def eventLeaf4883 : Array AnnotatedEvent := #[
  { event := event78128
    frameStart := 0 },
  { event := event78129
    frameStart := 0 },
  { event := event78130
    frameStart := 0 },
  { event := event78131
    frameStart := 0 },
  { event := event78132
    frameStart := 0 },
  { event := event78133
    frameStart := 0 },
  { event := event78134
    frameStart := 0 },
  { event := event78135
    frameStart := 0 },
  { event := event78136
    frameStart := 0 },
  { event := event78137
    frameStart := 0 },
  { event := event78138
    frameStart := 0 },
  { event := event78139
    frameStart := 0 },
  { event := event78140
    frameStart := 0 },
  { event := event78141
    frameStart := 0 },
  { event := event78142
    frameStart := 0 },
  { event := event78143
    frameStart := 0 }
]

def eventLeaf4884 : Array AnnotatedEvent := #[
  { event := event78144
    frameStart := 0 },
  { event := event78145
    frameStart := 0 },
  { event := event78146
    frameStart := 0 },
  { event := event78147
    frameStart := 0 },
  { event := event78148
    frameStart := 0 },
  { event := event78149
    frameStart := 0 },
  { event := event78150
    frameStart := 0 },
  { event := event78151
    frameStart := 0 },
  { event := event78152
    frameStart := 0 },
  { event := event78153
    frameStart := 0 },
  { event := event78154
    frameStart := 0 },
  { event := event78155
    frameStart := 0 },
  { event := event78156
    frameStart := 0 },
  { event := event78157
    frameStart := 0 },
  { event := event78158
    frameStart := 0 },
  { event := event78159
    frameStart := 0 }
]

def eventLeaf4885 : Array AnnotatedEvent := #[
  { event := event78160
    frameStart := 0 },
  { event := event78161
    frameStart := 0 },
  { event := event78162
    frameStart := 0 },
  { event := event78163
    frameStart := 0 },
  { event := event78164
    frameStart := 0 },
  { event := event78165
    frameStart := 0 },
  { event := event78166
    frameStart := 0 },
  { event := event78167
    frameStart := 78167 },
  { event := event78168
    frameStart := 78167 },
  { event := event78169
    frameStart := 78167 },
  { event := event78170
    frameStart := 78167 },
  { event := event78171
    frameStart := 78167 },
  { event := event78172
    frameStart := 78167 },
  { event := event78173
    frameStart := 78167 },
  { event := event78174
    frameStart := 78167 },
  { event := event78175
    frameStart := 78167 }
]

def eventLeaf4886 : Array AnnotatedEvent := #[
  { event := event78176
    frameStart := 78167 },
  { event := event78177
    frameStart := 78167 },
  { event := event78178
    frameStart := 78167 },
  { event := event78179
    frameStart := 78167 },
  { event := event78180
    frameStart := 78167 },
  { event := event78181
    frameStart := 78167 },
  { event := event78182
    frameStart := 78167 },
  { event := event78183
    frameStart := 78167 },
  { event := event78184
    frameStart := 78167 },
  { event := event78185
    frameStart := 78167 },
  { event := event78186
    frameStart := 78167 },
  { event := event78187
    frameStart := 78167 },
  { event := event78188
    frameStart := 78167 },
  { event := event78189
    frameStart := 78167 },
  { event := event78190
    frameStart := 78167 },
  { event := event78191
    frameStart := 78167 }
]

def eventLeaf4887 : Array AnnotatedEvent := #[
  { event := event78192
    frameStart := 78167 },
  { event := event78193
    frameStart := 78167 },
  { event := event78194
    frameStart := 78167 },
  { event := event78195
    frameStart := 78167 },
  { event := event78196
    frameStart := 78167 },
  { event := event78197
    frameStart := 78167 },
  { event := event78198
    frameStart := 78167 },
  { event := event78199
    frameStart := 78167 },
  { event := event78200
    frameStart := 78167 },
  { event := event78201
    frameStart := 78167 },
  { event := event78202
    frameStart := 78167 },
  { event := event78203
    frameStart := 78167 },
  { event := event78204
    frameStart := 78167 },
  { event := event78205
    frameStart := 78167 },
  { event := event78206
    frameStart := 78167 },
  { event := event78207
    frameStart := 78167 }
]

def eventLeaf4888 : Array AnnotatedEvent := #[
  { event := event78208
    frameStart := 78167 },
  { event := event78209
    frameStart := 78167 },
  { event := event78210
    frameStart := 78167 },
  { event := event78211
    frameStart := 78167 },
  { event := event78212
    frameStart := 78167 },
  { event := event78213
    frameStart := 78167 },
  { event := event78214
    frameStart := 78167 },
  { event := event78215
    frameStart := 78167 },
  { event := event78216
    frameStart := 78167 },
  { event := event78217
    frameStart := 78167 },
  { event := event78218
    frameStart := 78167 },
  { event := event78219
    frameStart := 78167 },
  { event := event78220
    frameStart := 78167 },
  { event := event78221
    frameStart := 78221 },
  { event := event78222
    frameStart := 78221 },
  { event := event78223
    frameStart := 78221 }
]

def eventLeaf4889 : Array AnnotatedEvent := #[
  { event := event78224
    frameStart := 78221 },
  { event := event78225
    frameStart := 78221 },
  { event := event78226
    frameStart := 78221 },
  { event := event78227
    frameStart := 78221 },
  { event := event78228
    frameStart := 78221 },
  { event := event78229
    frameStart := 78221 },
  { event := event78230
    frameStart := 78221 },
  { event := event78231
    frameStart := 78221 },
  { event := event78232
    frameStart := 78221 },
  { event := event78233
    frameStart := 78221 },
  { event := event78234
    frameStart := 78221 },
  { event := event78235
    frameStart := 78221 },
  { event := event78236
    frameStart := 78221 },
  { event := event78237
    frameStart := 78221 },
  { event := event78238
    frameStart := 78221 },
  { event := event78239
    frameStart := 78221 }
]

def eventLeaf4890 : Array AnnotatedEvent := #[
  { event := event78240
    frameStart := 78221 },
  { event := event78241
    frameStart := 78221 },
  { event := event78242
    frameStart := 78221 },
  { event := event78243
    frameStart := 78221 },
  { event := event78244
    frameStart := 78221 },
  { event := event78245
    frameStart := 78221 },
  { event := event78246
    frameStart := 78221 },
  { event := event78247
    frameStart := 78221 },
  { event := event78248
    frameStart := 78221 },
  { event := event78249
    frameStart := 78221 },
  { event := event78250
    frameStart := 78221 },
  { event := event78251
    frameStart := 78221 },
  { event := event78252
    frameStart := 78221 },
  { event := event78253
    frameStart := 78221 },
  { event := event78254
    frameStart := 78221 },
  { event := event78255
    frameStart := 78221 }
]

def eventLeaf4891 : Array AnnotatedEvent := #[
  { event := event78256
    frameStart := 78221 },
  { event := event78257
    frameStart := 78221 },
  { event := event78258
    frameStart := 78221 },
  { event := event78259
    frameStart := 78221 },
  { event := event78260
    frameStart := 78221 },
  { event := event78261
    frameStart := 78221 },
  { event := event78262
    frameStart := 78221 },
  { event := event78263
    frameStart := 78221 },
  { event := event78264
    frameStart := 78221 },
  { event := event78265
    frameStart := 78221 },
  { event := event78266
    frameStart := 78221 },
  { event := event78267
    frameStart := 78221 },
  { event := event78268
    frameStart := 78221 },
  { event := event78269
    frameStart := 78221 },
  { event := event78270
    frameStart := 78221 },
  { event := event78271
    frameStart := 78221 }
]

def eventLeaf4892 : Array AnnotatedEvent := #[
  { event := event78272
    frameStart := 78221 },
  { event := event78273
    frameStart := 78221 },
  { event := event78274
    frameStart := 78221 },
  { event := event78275
    frameStart := 78221 },
  { event := event78276
    frameStart := 78221 },
  { event := event78277
    frameStart := 78221 },
  { event := event78278
    frameStart := 78221 },
  { event := event78279
    frameStart := 78221 },
  { event := event78280
    frameStart := 78221 },
  { event := event78281
    frameStart := 78221 },
  { event := event78282
    frameStart := 78221 },
  { event := event78283
    frameStart := 78221 },
  { event := event78284
    frameStart := 78221 },
  { event := event78285
    frameStart := 78221 },
  { event := event78286
    frameStart := 78221 },
  { event := event78287
    frameStart := 78221 }
]

def eventLeaf4893 : Array AnnotatedEvent := #[
  { event := event78288
    frameStart := 78221 },
  { event := event78289
    frameStart := 78221 },
  { event := event78290
    frameStart := 78221 },
  { event := event78291
    frameStart := 78221 },
  { event := event78292
    frameStart := 78221 },
  { event := event78293
    frameStart := 78221 },
  { event := event78294
    frameStart := 78221 },
  { event := event78295
    frameStart := 78221 },
  { event := event78296
    frameStart := 78221 },
  { event := event78297
    frameStart := 78221 },
  { event := event78298
    frameStart := 78221 },
  { event := event78299
    frameStart := 78221 },
  { event := event78300
    frameStart := 78221 },
  { event := event78301
    frameStart := 78221 },
  { event := event78302
    frameStart := 78221 },
  { event := event78303
    frameStart := 78221 }
]

def eventLeaf4894 : Array AnnotatedEvent := #[
  { event := event78304
    frameStart := 78221 },
  { event := event78305
    frameStart := 78221 },
  { event := event78306
    frameStart := 78221 },
  { event := event78307
    frameStart := 78221 },
  { event := event78308
    frameStart := 78221 },
  { event := event78309
    frameStart := 78221 },
  { event := event78310
    frameStart := 78221 },
  { event := event78311
    frameStart := 78221 },
  { event := event78312
    frameStart := 78221 },
  { event := event78313
    frameStart := 78221 },
  { event := event78314
    frameStart := 78221 },
  { event := event78315
    frameStart := 78221 },
  { event := event78316
    frameStart := 78221 },
  { event := event78317
    frameStart := 78221 },
  { event := event78318
    frameStart := 78221 },
  { event := event78319
    frameStart := 78221 }
]

def eventLeaf4895 : Array AnnotatedEvent := #[
  { event := event78320
    frameStart := 78221 },
  { event := event78321
    frameStart := 78221 },
  { event := event78322
    frameStart := 78221 },
  { event := event78323
    frameStart := 78221 },
  { event := event78324
    frameStart := 78221 },
  { event := event78325
    frameStart := 0 },
  { event := event78326
    frameStart := 0 },
  { event := event78327
    frameStart := 0 },
  { event := event78328
    frameStart := 0 },
  { event := event78329
    frameStart := 0 },
  { event := event78330
    frameStart := 0 },
  { event := event78331
    frameStart := 0 },
  { event := event78332
    frameStart := 0 },
  { event := event78333
    frameStart := 0 },
  { event := event78334
    frameStart := 0 },
  { event := event78335
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events305
