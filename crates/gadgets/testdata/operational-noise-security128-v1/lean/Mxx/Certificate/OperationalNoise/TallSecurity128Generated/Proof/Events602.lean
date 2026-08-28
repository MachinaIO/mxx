import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events602

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact154112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨60931⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event154112 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨61430⟩⟩) 154111 exact154112RawTerms .large 154109 .exactZero (none)

def event154113 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨59406⟩⟩) ⟨⟨65⟩, ⟨43⟩, ⟨135⟩⟩ ⟨153947, 154113⟩

def event154114 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨60362⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩) (1) 0 2 (.universal 154113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩) (none) 154112)

def event154115 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60362⟩⟩, .relation 154114 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩)

def event154116 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60362⟩⟩, .relation 154114 1, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩, (-1)⟩)

def event154117 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60362⟩⟩, .relation 154114 2, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨60931⟩⟩]⟩, (1)⟩)

def event154118 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60362⟩⟩, .relation 154114 3, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact154119RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨60931⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154119RawTermsValid :
    exact154119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154119 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60362⟩⟩) exact154119RawTerms .large 153943 (.finite 202072841853861888) (some (153945))

def event154120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61428⟩⟩) 0 ⟨60362⟩ 154119

def event154121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61428⟩⟩) 1 ⟨61427⟩ 153933

def event154122 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61428⟩⟩) (.sum [.predecessor 0 154120 .coefficient, .predecessor 1 154121 .coefficient])

def event154123 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61428⟩⟩, .operator (⟨154119, 2⟩, ⟨153933, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨60931⟩⟩]⟩, (-1)⟩)

def event154124 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61428⟩⟩, .operator (⟨154119, 1⟩, ⟨153933, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩, (1)⟩)

def event154125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61428⟩⟩) (.sum [.result 154119 .summary, .result 153933 .summary])

def exact154126RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154126RawTermsValid :
    exact154126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61428⟩⟩) exact154126RawTerms .large 154122 (.finite 2997962647681031733248) (some (154125))

def event154127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61801⟩⟩) 0 ⟨61428⟩ 154126

def event154128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61801⟩⟩) 1 ⟨61799⟩ 153849

def event154129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61801⟩⟩) (.product (.predecessor 0 154127 .coefficient) (.predecessor 1 154128 .coefficient) (⟨false, false, none, none, none⟩))

def event154130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61801⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩) [⟨.result 153849 .coefficient, false, none⟩])

def event154131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61801⟩⟩) (.product (.result 154126 .summary) (.transfer 154130) (⟨false, false, none, none, none⟩))

def event154132 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61801⟩⟩, .operator (⟨154126, 0⟩, ⟨153849, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (1)⟩)

def event154133 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61801⟩⟩, .operator (⟨154126, 1⟩, ⟨153849, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (-1)⟩)

def event154134 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨61801⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61799⟩⟩) ⟨61074⟩ 153846)

def event154135 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61801⟩⟩, .relation 154134 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (-1)⟩)

def exact154136RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (-1)⟩]

theorem exact154136RawTermsValid :
    exact154136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154136 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61801⟩⟩) exact154136RawTerms .large 154129 (.finite 32190378816049003834595889643520) (some (154131))

def event154137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60636⟩⟩) 0 ⟨59805⟩ 7073

def event154138 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60636⟩⟩) (.authority (.relationPreimageSource ⟨72⟩))

def exact154139RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩, (1)⟩]

theorem exact154139RawTermsValid :
    exact154139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154139 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60636⟩⟩) exact154139RawTerms (.finite 5647228698) 154138 .exactZero (none)

def event154140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60638⟩⟩) 0 ⟨60636⟩ 154139

def event154141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60638⟩⟩) 1 ⟨2370⟩ 4

def event154142 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60638⟩⟩) (.scale (.predecessor 0 154140 .coefficient) (.value (.predecessor 1 154141 .coefficient)))

def exact154143RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩, (1)⟩]

theorem exact154143RawTermsValid :
    exact154143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154143 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60638⟩⟩) exact154143RawTerms (.finite 5647228698) 154142 .exactZero (none)

def event154144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60639⟩⟩) 0 ⟨5545⟩ 149120

def event154145 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60639⟩⟩) 1 ⟨60638⟩ 154143

def event154146 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60639⟩⟩) (.product (.predecessor 0 154144 .coefficient) (.predecessor 1 154145 .coefficient) (⟨false, false, none, none, none⟩))

def event154147 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60639⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩) [⟨.result 154139 .coefficient, false, none⟩])

def event154148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60639⟩⟩) (.product (.result 149120 .summary) (.transfer 154147) (⟨false, false, none, none, none⟩))

def event154149 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60639⟩⟩, .operator (⟨149120, 0⟩, ⟨154143, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩, (1)⟩)

def event154150 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨60637⟩⟩)

def event154151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event154152 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event154153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4614⟩⟩) (.authority (.operator))

def event154154 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4614⟩⟩) (.finite 10)

def event154155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event154156 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event154157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event154158 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event154159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 154158

def event154160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 154156

def event154161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 154159 .coefficient) (.value (.predecessor 1 154160 .coefficient)))

def event154162 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event154163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4616⟩⟩) 0 ⟨392⟩ 154162

def event154164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4616⟩⟩) 1 ⟨4614⟩ 154154

def event154165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4616⟩⟩) (.sum [.predecessor 0 154163 .coefficient, .predecessor 1 154164 .coefficient])

def event154166 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4616⟩⟩) (.finite 655350)

def event154167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5541⟩⟩) 0 ⟨4616⟩ 154166

def event154168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5541⟩⟩) 1 ⟨5426⟩ 154152

def event154169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5541⟩⟩) (.identity (.predecessor 1 154168 .coefficient))

def event154170 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5541⟩⟩) (.finite 655360)

def event154171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25214⟩⟩) 0 ⟨5541⟩ 154170

def event154172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25214⟩⟩) (.authority (.programFamilyFact))

def exact154173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩], []⟩, (1)⟩]

theorem exact154173RawTermsValid :
    exact154173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25214⟩⟩) exact154173RawTerms (.finite 18) 154172 .exactZero (none)

def event154174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59404⟩⟩) 0 ⟨5541⟩ 154170

def event154175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59404⟩⟩) (.authority (.programFamilyFact))

def exact154176RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59404⟩⟩], []⟩, (1)⟩]

theorem exact154176RawTermsValid :
    exact154176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154176 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59404⟩⟩) exact154176RawTerms (.finite 18) 154175 .exactZero (none)

def event154177 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59405⟩⟩) 0 ⟨59404⟩ 154176

def event154178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59405⟩⟩) 1 ⟨25214⟩ 154173

def event154179 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59405⟩⟩) (.product (.predecessor 0 154177 .coefficient) (.predecessor 1 154178 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event154180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59405⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], []⟩) [⟨.result 154176 .coefficient, true, some 1⟩, ⟨.result 154173 .coefficient, true, some 1⟩])

def event154181 : Event := .survivorFold (1) 154180

def exact154182RawTerms : List Term := []

theorem exact154182RawTermsValid :
    exact154182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59405⟩⟩) exact154182RawTerms (.finite 324) 154179 (.finite 324) (some (154180))

def event154183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59406⟩⟩) 0 ⟨59405⟩ 154182

def event154184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59406⟩⟩) (.identity (.predecessor 0 154183 .coefficient))

def event154185 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59406⟩⟩) (.finite 324)

def event154186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59804⟩⟩) 0 ⟨59406⟩ 154185

def event154187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59804⟩⟩) (.authority (.programFamilyFact))

def exact154188RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], []⟩, (1)⟩]

theorem exact154188RawTermsValid :
    exact154188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59804⟩⟩) exact154188RawTerms (.finite 18) 154187 .exactZero (none)

def event154189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59805⟩⟩) 0 ⟨59804⟩ 154188

def event154190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59805⟩⟩) (.identity (.predecessor 0 154189 .coefficient))

def event154191 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59805⟩⟩) (.finite 18)

def event154192 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60636⟩⟩) 0 ⟨59805⟩ 154191

def event154193 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60636⟩⟩) (.authority (.relationPreimageSource ⟨72⟩))

def exact154194RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩, (1)⟩]

theorem exact154194RawTermsValid :
    exact154194RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154194 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60636⟩⟩) exact154194RawTerms (.finite 5647228698) 154193 .exactZero (none)

def event154195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact154196RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact154196RawTermsValid :
    exact154196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154196 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact154196RawTerms .large 154195 .exactZero (none)

def event154197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60637⟩⟩) 0 ⟨35⟩ 154196

def event154198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60637⟩⟩) 1 ⟨60636⟩ 154194

def event154199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60637⟩⟩) (.product (.predecessor 0 154197 .coefficient) (.predecessor 1 154198 .coefficient) (⟨false, false, none, none, none⟩))

def event154200 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60637⟩⟩, .operator (⟨154196, 0⟩, ⟨154194, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩, (1)⟩)

def exact154201RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩, (1)⟩]

theorem exact154201RawTermsValid :
    exact154201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60637⟩⟩) exact154201RawTerms .large 154199 .exactZero (none)

def event154202 : Event := .preFoldPolynomial 154201 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩, (1)⟩] .exactZero none

def exact154203RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩, (1)⟩]

def event154203 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨60637⟩⟩) 154202 exact154203RawTerms .large 154199 .exactZero (none)

def event154204 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨61804⟩⟩)

def event154205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event154206 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event154207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4614⟩⟩) (.authority (.operator))

def event154208 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4614⟩⟩) (.finite 10)

def event154209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event154210 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event154211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event154212 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event154213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 154212

def event154214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 154210

def event154215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 154213 .coefficient) (.value (.predecessor 1 154214 .coefficient)))

def event154216 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event154217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4616⟩⟩) 0 ⟨392⟩ 154216

def event154218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4616⟩⟩) 1 ⟨4614⟩ 154208

def event154219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4616⟩⟩) (.sum [.predecessor 0 154217 .coefficient, .predecessor 1 154218 .coefficient])

def event154220 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4616⟩⟩) (.finite 655350)

def event154221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5541⟩⟩) 0 ⟨4616⟩ 154220

def event154222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5541⟩⟩) 1 ⟨5426⟩ 154206

def event154223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5541⟩⟩) (.identity (.predecessor 1 154222 .coefficient))

def event154224 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5541⟩⟩) (.finite 655360)

def event154225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25214⟩⟩) 0 ⟨5541⟩ 154224

def event154226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25214⟩⟩) (.authority (.programFamilyFact))

def exact154227RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩], []⟩, (1)⟩]

theorem exact154227RawTermsValid :
    exact154227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25214⟩⟩) exact154227RawTerms (.finite 18) 154226 .exactZero (none)

def event154228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59404⟩⟩) 0 ⟨5541⟩ 154224

def event154229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59404⟩⟩) (.authority (.programFamilyFact))

def exact154230RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59404⟩⟩], []⟩, (1)⟩]

theorem exact154230RawTermsValid :
    exact154230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154230 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59404⟩⟩) exact154230RawTerms (.finite 18) 154229 .exactZero (none)

def event154231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59405⟩⟩) 0 ⟨59404⟩ 154230

def event154232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59405⟩⟩) 1 ⟨25214⟩ 154227

def event154233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59405⟩⟩) (.product (.predecessor 0 154231 .coefficient) (.predecessor 1 154232 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event154234 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59405⟩⟩, .operator (⟨154230, 0⟩, ⟨154227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], []⟩, (1)⟩)

def exact154235RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], []⟩, (1)⟩]

theorem exact154235RawTermsValid :
    exact154235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59405⟩⟩) exact154235RawTerms (.finite 324) 154233 .exactZero (none)

def event154236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59406⟩⟩) 0 ⟨59405⟩ 154235

def event154237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59406⟩⟩) (.identity (.predecessor 0 154236 .coefficient))

def event154238 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59406⟩⟩) (.finite 324)

def event154239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59804⟩⟩) 0 ⟨59406⟩ 154238

def event154240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59804⟩⟩) (.authority (.programFamilyFact))

def exact154241RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], []⟩, (1)⟩]

theorem exact154241RawTermsValid :
    exact154241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154241 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59804⟩⟩) exact154241RawTerms (.finite 18) 154240 .exactZero (none)

def event154242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59805⟩⟩) 0 ⟨59804⟩ 154241

def event154243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59805⟩⟩) (.identity (.predecessor 0 154242 .coefficient))

def event154244 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59805⟩⟩) (.finite 18)

def event154245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61072⟩⟩) 0 ⟨59805⟩ 154244

def event154246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61072⟩⟩) (.authority (.programFamilyFact))

def event154247 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨61072⟩⟩) (.finite 3720)

def event154248 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event154249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61074⟩⟩) 0 ⟨7177⟩ 154248

def event154250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61074⟩⟩) 1 ⟨61072⟩ 154247

def event154251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61074⟩⟩) (.authority (.operator))

def exact154252RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (1)⟩]

theorem exact154252RawTermsValid :
    exact154252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154252 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61074⟩⟩) exact154252RawTerms .large 154251 .exactZero (none)

def event154253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61799⟩⟩) 0 ⟨61074⟩ 154252

def event154254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61799⟩⟩) (.authority (.operator))

def exact154255RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (1)⟩]

theorem exact154255RawTermsValid :
    exact154255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154255 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61799⟩⟩) exact154255RawTerms (.finite 8192) 154254 .exactZero (none)

def event154256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event154257 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event154258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61294⟩⟩) 0 ⟨59805⟩ 154244

def event154259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61294⟩⟩) 1 ⟨136⟩ 154257

def event154260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61294⟩⟩) (.sum [.predecessor 0 154258 .coefficient, .predecessor 1 154259 .coefficient])

def event154261 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨61294⟩⟩) (.finite 18)

def event154262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61295⟩⟩) 0 ⟨61294⟩ 154261

def event154263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61295⟩⟩) (.identity (.predecessor 0 154262 .coefficient))

def exact154264RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], []⟩, (1)⟩]

theorem exact154264RawTermsValid :
    exact154264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154264 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61295⟩⟩) exact154264RawTerms (.finite 18) 154263 .exactZero (none)

def event154265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact154266RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact154266RawTermsValid :
    exact154266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact154266RawTerms .large 154265 .exactZero (none)

def event154267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61296⟩⟩) 0 ⟨6908⟩ 154266

def event154268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61296⟩⟩) 1 ⟨61295⟩ 154264

def event154269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61296⟩⟩) (.product (.predecessor 0 154267 .coefficient) (.predecessor 1 154268 .coefficient) (⟨false, false, none, none, none⟩))

def event154270 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61296⟩⟩, .operator (⟨154266, 0⟩, ⟨154264, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact154271RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact154271RawTermsValid :
    exact154271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154271 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61296⟩⟩) exact154271RawTerms .large 154269 .exactZero (none)

def event154272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7186⟩⟩) 0 ⟨7177⟩ 154248

def event154273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7186⟩⟩) (.authority (.operator))

def exact154274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩]

theorem exact154274RawTermsValid :
    exact154274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7186⟩⟩) exact154274RawTerms .large 154273 .exactZero (none)

def event154275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61297⟩⟩) 0 ⟨7186⟩ 154274

def event154276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61297⟩⟩) 1 ⟨61296⟩ 154271

def event154277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61297⟩⟩) (.sum [.predecessor 0 154275 .coefficient, .predecessor 1 154276 .coefficient])

def exact154278RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154278RawTermsValid :
    exact154278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154278 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61297⟩⟩) exact154278RawTerms .large 154277 .exactZero (none)

def event154279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61800⟩⟩) 0 ⟨61297⟩ 154278

def event154280 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61800⟩⟩) 1 ⟨61799⟩ 154255

def event154281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61800⟩⟩) (.product (.predecessor 0 154279 .coefficient) (.predecessor 1 154280 .coefficient) (⟨false, false, none, none, none⟩))

def event154282 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61800⟩⟩, .operator (⟨154278, 0⟩, ⟨154255, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (1)⟩)

def event154283 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61800⟩⟩, .operator (⟨154278, 1⟩, ⟨154255, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (-1)⟩)

def event154284 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨61800⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61799⟩⟩) ⟨61074⟩ 154252)

def event154285 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61800⟩⟩, .relation 154284 0, ⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (-1)⟩)

def exact154286RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (-1)⟩]

theorem exact154286RawTermsValid :
    exact154286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154286 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61800⟩⟩) exact154286RawTerms .large 154281 .exactZero (none)

def event154287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60044⟩⟩) 0 ⟨59805⟩ 154244

def event154288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60044⟩⟩) (.authority (.programFamilyFact))

def exact154289RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], []⟩, (1)⟩]

theorem exact154289RawTermsValid :
    exact154289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60044⟩⟩) exact154289RawTerms (.finite 61) 154288 .exactZero (none)

def event154290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60046⟩⟩) 0 ⟨6908⟩ 154266

def event154291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60046⟩⟩) 1 ⟨60044⟩ 154289

def event154292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60046⟩⟩) (.product (.predecessor 0 154290 .coefficient) (.predecessor 1 154291 .coefficient) (⟨false, true, none, none, some 1⟩))

def event154293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60046⟩⟩, .operator (⟨154266, 0⟩, ⟨154289, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact154294RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact154294RawTermsValid :
    exact154294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154294 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60046⟩⟩) exact154294RawTerms .large 154292 .exactZero (none)

def event154295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 154248

def event154296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact154297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact154297RawTermsValid :
    exact154297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact154297RawTerms .large 154296 .exactZero (none)

def event154298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60047⟩⟩) 0 ⟨7212⟩ 154297

def event154299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60047⟩⟩) 1 ⟨60046⟩ 154294

def event154300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60047⟩⟩) (.sum [.predecessor 0 154298 .coefficient, .predecessor 1 154299 .coefficient])

def exact154301RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154301RawTermsValid :
    exact154301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154301 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60047⟩⟩) exact154301RawTerms .large 154300 .exactZero (none)

def event154302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61804⟩⟩) 0 ⟨60047⟩ 154301

def event154303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61804⟩⟩) 1 ⟨61800⟩ 154286

def event154304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61804⟩⟩) (.sum [.predecessor 0 154302 .coefficient, .predecessor 1 154303 .coefficient])

def exact154305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154305RawTermsValid :
    exact154305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61804⟩⟩) exact154305RawTerms .large 154304 .exactZero (none)

def event154306 : Event := .preFoldPolynomial 154305 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact154307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event154307 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨61804⟩⟩) 154306 exact154307RawTerms .large 154304 .exactZero (none)

def event154308 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨59805⟩⟩) ⟨⟨91⟩, ⟨72⟩, ⟨135⟩⟩ ⟨154150, 154308⟩

def event154309 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨60639⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩) (1) 0 2 (.universal 154308 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60636⟩⟩]⟩) (none) 154307)

def event154310 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60639⟩⟩, .relation 154309 1, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩)

def event154311 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60639⟩⟩, .relation 154309 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (-1)⟩)

def event154312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60639⟩⟩, .relation 154309 2, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (1)⟩)

def event154313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60639⟩⟩, .relation 154309 3, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact154314RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154314RawTermsValid :
    exact154314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60639⟩⟩) exact154314RawTerms .large 154146 (.finite 202072841853861888) (some (154148))

def event154315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61802⟩⟩) 0 ⟨60639⟩ 154314

def event154316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61802⟩⟩) 1 ⟨61801⟩ 154136

def event154317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61802⟩⟩) (.sum [.predecessor 0 154315 .coefficient, .predecessor 1 154316 .coefficient])

def event154318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61802⟩⟩, .operator (⟨154314, 0⟩, ⟨154136, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61799⟩⟩]⟩, (1)⟩)

def event154319 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61802⟩⟩, .operator (⟨154314, 2⟩, ⟨154136, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61074⟩⟩]⟩, (-1)⟩)

def event154320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61802⟩⟩) (.sum [.result 154314 .summary, .result 154136 .summary])

def exact154321RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154321RawTermsValid :
    exact154321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61802⟩⟩) exact154321RawTerms .large 154317 (.finite 32190378816049205907437743505408) (some (154320))

def event154322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58092⟩⟩) 0 ⟨56825⟩ 7096

def event154323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58092⟩⟩) (.authority (.programFamilyFact))

def event154324 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58092⟩⟩) (.finite 3720)

def event154325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58094⟩⟩) 0 ⟨7177⟩ 15500

def event154326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58094⟩⟩) 1 ⟨58092⟩ 154324

def event154327 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58094⟩⟩) (.authority (.operator))

def exact154328RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58094⟩⟩]⟩, (1)⟩]

theorem exact154328RawTermsValid :
    exact154328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154328 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58094⟩⟩) exact154328RawTerms .large 154327 .exactZero (none)

def event154329 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58819⟩⟩) 0 ⟨58094⟩ 154328

def event154330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58819⟩⟩) (.authority (.operator))

def exact154331RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58819⟩⟩]⟩, (1)⟩]

theorem exact154331RawTermsValid :
    exact154331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154331 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58819⟩⟩) exact154331RawTerms (.finite 8192) 154330 .exactZero (none)

def event154332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57950⟩⟩) 0 ⟨56426⟩ 7090

def event154333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57950⟩⟩) (.authority (.programFamilyFact))

def event154334 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨57950⟩⟩) (.finite 3720)

def event154335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57951⟩⟩) 0 ⟨7177⟩ 15500

def event154336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57951⟩⟩) 1 ⟨57950⟩ 154334

def event154337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57951⟩⟩) (.authority (.operator))

def exact154338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57951⟩⟩]⟩, (1)⟩]

theorem exact154338RawTermsValid :
    exact154338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57951⟩⟩) exact154338RawTerms .large 154337 .exactZero (none)

def event154339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58446⟩⟩) 0 ⟨57951⟩ 154338

def event154340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58446⟩⟩) (.authority (.operator))

def exact154341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58446⟩⟩]⟩, (1)⟩]

theorem exact154341RawTermsValid :
    exact154341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154341 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58446⟩⟩) exact154341RawTerms (.finite 8192) 154340 .exactZero (none)

def event154342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24975⟩⟩) 0 ⟨24974⟩ 7079

def event154343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24975⟩⟩) 1 ⟨6931⟩ 149028

def event154344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24975⟩⟩) (.tensor (.predecessor 0 154342 .coefficient) (.predecessor 1 154343 .coefficient) true false)

def event154345 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24975⟩⟩, .operator (⟨7079, 0⟩, ⟨149028, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact154346RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact154346RawTermsValid :
    exact154346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24975⟩⟩) exact154346RawTerms .large 154344 .exactZero (none)

def event154347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8237⟩⟩) 0 ⟨5543⟩ 148898

def event154348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8237⟩⟩) 1 ⟨7273⟩ 22591

def event154349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8237⟩⟩) (.product (.predecessor 0 154347 .coefficient) (.predecessor 1 154348 .coefficient) (⟨false, false, none, none, none⟩))

def event154350 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8237⟩⟩, .operator (⟨148898, 0⟩, ⟨22591, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩)

def exact154351RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩]

theorem exact154351RawTermsValid :
    exact154351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8237⟩⟩) exact154351RawTerms .large 154349 .exactZero (none)

def event154352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24976⟩⟩) 0 ⟨8237⟩ 154351

def event154353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24976⟩⟩) 1 ⟨24975⟩ 154346

def event154354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24976⟩⟩) (.sum [.predecessor 0 154352 .coefficient, .predecessor 1 154353 .coefficient])

def exact154355RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154355RawTermsValid :
    exact154355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24976⟩⟩) exact154355RawTerms .large 154354 .exactZero (none)

def event154356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24977⟩⟩) 0 ⟨24976⟩ 154355

def event154357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24977⟩⟩) 1 ⟨99⟩ 22583

def event154358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24977⟩⟩) (.sum [.predecessor 0 154356 .coefficient, .predecessor 1 154357 .coefficient])

def event154359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24977⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨99⟩⟩]⟩) [⟨.result 22583 .coefficient, false, none⟩])

def event154360 : Event := .survivorFold (1) 154359

def exact154361RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact154361RawTermsValid :
    exact154361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event154361 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24977⟩⟩) exact154361RawTerms .large 154358 (.finite 26) (some (154359))

def event154362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56427⟩⟩) 0 ⟨24977⟩ 154361

def event154363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56427⟩⟩) 1 ⟨56424⟩ 7082

def event154364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56427⟩⟩) (.product (.predecessor 0 154362 .coefficient) (.predecessor 1 154363 .coefficient) (⟨false, true, none, none, some 1⟩))

def event154365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56427⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨56424⟩⟩], []⟩) [⟨.result 7082 .coefficient, true, some 1⟩])

def event154366 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56427⟩⟩) (.product (.result 154361 .summary) (.transfer 154365) (⟨false, false, none, none, none⟩))

def event154367 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56427⟩⟩, .operator (⟨154361, 1⟩, ⟨7082, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨24974⟩⟩, ⟨.program ⟨257⟩, ⟨56424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def eventLeaf9632 : Array AnnotatedEvent := #[
  { event := event154112
    frameStart := 153995 },
  { event := event154113
    frameStart := 0 },
  { event := event154114
    frameStart := 0 },
  { event := event154115
    frameStart := 0 },
  { event := event154116
    frameStart := 0 },
  { event := event154117
    frameStart := 0 },
  { event := event154118
    frameStart := 0 },
  { event := event154119
    frameStart := 0 },
  { event := event154120
    frameStart := 0 },
  { event := event154121
    frameStart := 0 },
  { event := event154122
    frameStart := 0 },
  { event := event154123
    frameStart := 0 },
  { event := event154124
    frameStart := 0 },
  { event := event154125
    frameStart := 0 },
  { event := event154126
    frameStart := 0 },
  { event := event154127
    frameStart := 0 }
]

def eventLeaf9633 : Array AnnotatedEvent := #[
  { event := event154128
    frameStart := 0 },
  { event := event154129
    frameStart := 0 },
  { event := event154130
    frameStart := 0 },
  { event := event154131
    frameStart := 0 },
  { event := event154132
    frameStart := 0 },
  { event := event154133
    frameStart := 0 },
  { event := event154134
    frameStart := 0 },
  { event := event154135
    frameStart := 0 },
  { event := event154136
    frameStart := 0 },
  { event := event154137
    frameStart := 0 },
  { event := event154138
    frameStart := 0 },
  { event := event154139
    frameStart := 0 },
  { event := event154140
    frameStart := 0 },
  { event := event154141
    frameStart := 0 },
  { event := event154142
    frameStart := 0 },
  { event := event154143
    frameStart := 0 }
]

def eventLeaf9634 : Array AnnotatedEvent := #[
  { event := event154144
    frameStart := 0 },
  { event := event154145
    frameStart := 0 },
  { event := event154146
    frameStart := 0 },
  { event := event154147
    frameStart := 0 },
  { event := event154148
    frameStart := 0 },
  { event := event154149
    frameStart := 0 },
  { event := event154150
    frameStart := 154150 },
  { event := event154151
    frameStart := 154150 },
  { event := event154152
    frameStart := 154150 },
  { event := event154153
    frameStart := 154150 },
  { event := event154154
    frameStart := 154150 },
  { event := event154155
    frameStart := 154150 },
  { event := event154156
    frameStart := 154150 },
  { event := event154157
    frameStart := 154150 },
  { event := event154158
    frameStart := 154150 },
  { event := event154159
    frameStart := 154150 }
]

def eventLeaf9635 : Array AnnotatedEvent := #[
  { event := event154160
    frameStart := 154150 },
  { event := event154161
    frameStart := 154150 },
  { event := event154162
    frameStart := 154150 },
  { event := event154163
    frameStart := 154150 },
  { event := event154164
    frameStart := 154150 },
  { event := event154165
    frameStart := 154150 },
  { event := event154166
    frameStart := 154150 },
  { event := event154167
    frameStart := 154150 },
  { event := event154168
    frameStart := 154150 },
  { event := event154169
    frameStart := 154150 },
  { event := event154170
    frameStart := 154150 },
  { event := event154171
    frameStart := 154150 },
  { event := event154172
    frameStart := 154150 },
  { event := event154173
    frameStart := 154150 },
  { event := event154174
    frameStart := 154150 },
  { event := event154175
    frameStart := 154150 }
]

def eventLeaf9636 : Array AnnotatedEvent := #[
  { event := event154176
    frameStart := 154150 },
  { event := event154177
    frameStart := 154150 },
  { event := event154178
    frameStart := 154150 },
  { event := event154179
    frameStart := 154150 },
  { event := event154180
    frameStart := 154150 },
  { event := event154181
    frameStart := 154150 },
  { event := event154182
    frameStart := 154150 },
  { event := event154183
    frameStart := 154150 },
  { event := event154184
    frameStart := 154150 },
  { event := event154185
    frameStart := 154150 },
  { event := event154186
    frameStart := 154150 },
  { event := event154187
    frameStart := 154150 },
  { event := event154188
    frameStart := 154150 },
  { event := event154189
    frameStart := 154150 },
  { event := event154190
    frameStart := 154150 },
  { event := event154191
    frameStart := 154150 }
]

def eventLeaf9637 : Array AnnotatedEvent := #[
  { event := event154192
    frameStart := 154150 },
  { event := event154193
    frameStart := 154150 },
  { event := event154194
    frameStart := 154150 },
  { event := event154195
    frameStart := 154150 },
  { event := event154196
    frameStart := 154150 },
  { event := event154197
    frameStart := 154150 },
  { event := event154198
    frameStart := 154150 },
  { event := event154199
    frameStart := 154150 },
  { event := event154200
    frameStart := 154150 },
  { event := event154201
    frameStart := 154150 },
  { event := event154202
    frameStart := 154150 },
  { event := event154203
    frameStart := 154150 },
  { event := event154204
    frameStart := 154204 },
  { event := event154205
    frameStart := 154204 },
  { event := event154206
    frameStart := 154204 },
  { event := event154207
    frameStart := 154204 }
]

def eventLeaf9638 : Array AnnotatedEvent := #[
  { event := event154208
    frameStart := 154204 },
  { event := event154209
    frameStart := 154204 },
  { event := event154210
    frameStart := 154204 },
  { event := event154211
    frameStart := 154204 },
  { event := event154212
    frameStart := 154204 },
  { event := event154213
    frameStart := 154204 },
  { event := event154214
    frameStart := 154204 },
  { event := event154215
    frameStart := 154204 },
  { event := event154216
    frameStart := 154204 },
  { event := event154217
    frameStart := 154204 },
  { event := event154218
    frameStart := 154204 },
  { event := event154219
    frameStart := 154204 },
  { event := event154220
    frameStart := 154204 },
  { event := event154221
    frameStart := 154204 },
  { event := event154222
    frameStart := 154204 },
  { event := event154223
    frameStart := 154204 }
]

def eventLeaf9639 : Array AnnotatedEvent := #[
  { event := event154224
    frameStart := 154204 },
  { event := event154225
    frameStart := 154204 },
  { event := event154226
    frameStart := 154204 },
  { event := event154227
    frameStart := 154204 },
  { event := event154228
    frameStart := 154204 },
  { event := event154229
    frameStart := 154204 },
  { event := event154230
    frameStart := 154204 },
  { event := event154231
    frameStart := 154204 },
  { event := event154232
    frameStart := 154204 },
  { event := event154233
    frameStart := 154204 },
  { event := event154234
    frameStart := 154204 },
  { event := event154235
    frameStart := 154204 },
  { event := event154236
    frameStart := 154204 },
  { event := event154237
    frameStart := 154204 },
  { event := event154238
    frameStart := 154204 },
  { event := event154239
    frameStart := 154204 }
]

def eventLeaf9640 : Array AnnotatedEvent := #[
  { event := event154240
    frameStart := 154204 },
  { event := event154241
    frameStart := 154204 },
  { event := event154242
    frameStart := 154204 },
  { event := event154243
    frameStart := 154204 },
  { event := event154244
    frameStart := 154204 },
  { event := event154245
    frameStart := 154204 },
  { event := event154246
    frameStart := 154204 },
  { event := event154247
    frameStart := 154204 },
  { event := event154248
    frameStart := 154204 },
  { event := event154249
    frameStart := 154204 },
  { event := event154250
    frameStart := 154204 },
  { event := event154251
    frameStart := 154204 },
  { event := event154252
    frameStart := 154204 },
  { event := event154253
    frameStart := 154204 },
  { event := event154254
    frameStart := 154204 },
  { event := event154255
    frameStart := 154204 }
]

def eventLeaf9641 : Array AnnotatedEvent := #[
  { event := event154256
    frameStart := 154204 },
  { event := event154257
    frameStart := 154204 },
  { event := event154258
    frameStart := 154204 },
  { event := event154259
    frameStart := 154204 },
  { event := event154260
    frameStart := 154204 },
  { event := event154261
    frameStart := 154204 },
  { event := event154262
    frameStart := 154204 },
  { event := event154263
    frameStart := 154204 },
  { event := event154264
    frameStart := 154204 },
  { event := event154265
    frameStart := 154204 },
  { event := event154266
    frameStart := 154204 },
  { event := event154267
    frameStart := 154204 },
  { event := event154268
    frameStart := 154204 },
  { event := event154269
    frameStart := 154204 },
  { event := event154270
    frameStart := 154204 },
  { event := event154271
    frameStart := 154204 }
]

def eventLeaf9642 : Array AnnotatedEvent := #[
  { event := event154272
    frameStart := 154204 },
  { event := event154273
    frameStart := 154204 },
  { event := event154274
    frameStart := 154204 },
  { event := event154275
    frameStart := 154204 },
  { event := event154276
    frameStart := 154204 },
  { event := event154277
    frameStart := 154204 },
  { event := event154278
    frameStart := 154204 },
  { event := event154279
    frameStart := 154204 },
  { event := event154280
    frameStart := 154204 },
  { event := event154281
    frameStart := 154204 },
  { event := event154282
    frameStart := 154204 },
  { event := event154283
    frameStart := 154204 },
  { event := event154284
    frameStart := 154204 },
  { event := event154285
    frameStart := 154204 },
  { event := event154286
    frameStart := 154204 },
  { event := event154287
    frameStart := 154204 }
]

def eventLeaf9643 : Array AnnotatedEvent := #[
  { event := event154288
    frameStart := 154204 },
  { event := event154289
    frameStart := 154204 },
  { event := event154290
    frameStart := 154204 },
  { event := event154291
    frameStart := 154204 },
  { event := event154292
    frameStart := 154204 },
  { event := event154293
    frameStart := 154204 },
  { event := event154294
    frameStart := 154204 },
  { event := event154295
    frameStart := 154204 },
  { event := event154296
    frameStart := 154204 },
  { event := event154297
    frameStart := 154204 },
  { event := event154298
    frameStart := 154204 },
  { event := event154299
    frameStart := 154204 },
  { event := event154300
    frameStart := 154204 },
  { event := event154301
    frameStart := 154204 },
  { event := event154302
    frameStart := 154204 },
  { event := event154303
    frameStart := 154204 }
]

def eventLeaf9644 : Array AnnotatedEvent := #[
  { event := event154304
    frameStart := 154204 },
  { event := event154305
    frameStart := 154204 },
  { event := event154306
    frameStart := 154204 },
  { event := event154307
    frameStart := 154204 },
  { event := event154308
    frameStart := 0 },
  { event := event154309
    frameStart := 0 },
  { event := event154310
    frameStart := 0 },
  { event := event154311
    frameStart := 0 },
  { event := event154312
    frameStart := 0 },
  { event := event154313
    frameStart := 0 },
  { event := event154314
    frameStart := 0 },
  { event := event154315
    frameStart := 0 },
  { event := event154316
    frameStart := 0 },
  { event := event154317
    frameStart := 0 },
  { event := event154318
    frameStart := 0 },
  { event := event154319
    frameStart := 0 }
]

def eventLeaf9645 : Array AnnotatedEvent := #[
  { event := event154320
    frameStart := 0 },
  { event := event154321
    frameStart := 0 },
  { event := event154322
    frameStart := 0 },
  { event := event154323
    frameStart := 0 },
  { event := event154324
    frameStart := 0 },
  { event := event154325
    frameStart := 0 },
  { event := event154326
    frameStart := 0 },
  { event := event154327
    frameStart := 0 },
  { event := event154328
    frameStart := 0 },
  { event := event154329
    frameStart := 0 },
  { event := event154330
    frameStart := 0 },
  { event := event154331
    frameStart := 0 },
  { event := event154332
    frameStart := 0 },
  { event := event154333
    frameStart := 0 },
  { event := event154334
    frameStart := 0 },
  { event := event154335
    frameStart := 0 }
]

def eventLeaf9646 : Array AnnotatedEvent := #[
  { event := event154336
    frameStart := 0 },
  { event := event154337
    frameStart := 0 },
  { event := event154338
    frameStart := 0 },
  { event := event154339
    frameStart := 0 },
  { event := event154340
    frameStart := 0 },
  { event := event154341
    frameStart := 0 },
  { event := event154342
    frameStart := 0 },
  { event := event154343
    frameStart := 0 },
  { event := event154344
    frameStart := 0 },
  { event := event154345
    frameStart := 0 },
  { event := event154346
    frameStart := 0 },
  { event := event154347
    frameStart := 0 },
  { event := event154348
    frameStart := 0 },
  { event := event154349
    frameStart := 0 },
  { event := event154350
    frameStart := 0 },
  { event := event154351
    frameStart := 0 }
]

def eventLeaf9647 : Array AnnotatedEvent := #[
  { event := event154352
    frameStart := 0 },
  { event := event154353
    frameStart := 0 },
  { event := event154354
    frameStart := 0 },
  { event := event154355
    frameStart := 0 },
  { event := event154356
    frameStart := 0 },
  { event := event154357
    frameStart := 0 },
  { event := event154358
    frameStart := 0 },
  { event := event154359
    frameStart := 0 },
  { event := event154360
    frameStart := 0 },
  { event := event154361
    frameStart := 0 },
  { event := event154362
    frameStart := 0 },
  { event := event154363
    frameStart := 0 },
  { event := event154364
    frameStart := 0 },
  { event := event154365
    frameStart := 0 },
  { event := event154366
    frameStart := 0 },
  { event := event154367
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events602
