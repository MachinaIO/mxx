import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events188

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event48128 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event48129 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event48130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event48131 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event48132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 48131

def event48133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 48129

def event48134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 48132 .coefficient) (.value (.predecessor 1 48133 .coefficient)))

def event48135 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event48136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 48135

def event48137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 48127

def event48138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 48136 .coefficient, .predecessor 1 48137 .coefficient])

def event48139 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event48140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 48139

def event48141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 48125

def event48142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 48141 .coefficient))

def event48143 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event48144 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11645⟩⟩) 0 ⟨5548⟩ 48143

def event48145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11645⟩⟩) (.authority (.programFamilyFact))

def exact48146RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩], []⟩, (1)⟩]

theorem exact48146RawTermsValid :
    exact48146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11645⟩⟩) exact48146RawTerms (.finite 28) 48145 .exactZero (none)

def event48147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14659⟩⟩) 0 ⟨5548⟩ 48143

def event48148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14659⟩⟩) (.authority (.programFamilyFact))

def exact48149RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14659⟩⟩], []⟩, (1)⟩]

theorem exact48149RawTermsValid :
    exact48149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48149 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14659⟩⟩) exact48149RawTerms (.finite 28) 48148 .exactZero (none)

def event48150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14660⟩⟩) 0 ⟨14659⟩ 48149

def event48151 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14660⟩⟩) 1 ⟨11645⟩ 48146

def event48152 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14660⟩⟩) (.product (.predecessor 0 48150 .coefficient) (.predecessor 1 48151 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48153 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14660⟩⟩, .operator (⟨48149, 0⟩, ⟨48146, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], []⟩, (1)⟩)

def exact48154RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], []⟩, (1)⟩]

theorem exact48154RawTermsValid :
    exact48154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48154 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14660⟩⟩) exact48154RawTerms (.finite 784) 48152 .exactZero (none)

def event48155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14661⟩⟩) 0 ⟨14660⟩ 48154

def event48156 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14661⟩⟩) (.identity (.predecessor 0 48155 .coefficient))

def event48157 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14661⟩⟩) (.finite 784)

def event48158 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16186⟩⟩) 0 ⟨14661⟩ 48157

def event48159 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16186⟩⟩) (.authority (.programFamilyFact))

def exact48160RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], []⟩, (1)⟩]

theorem exact48160RawTermsValid :
    exact48160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48160 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16186⟩⟩) exact48160RawTerms (.finite 28) 48159 .exactZero (none)

def event48161 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16187⟩⟩) 0 ⟨16186⟩ 48160

def event48162 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16187⟩⟩) (.identity (.predecessor 0 48161 .coefficient))

def event48163 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16187⟩⟩) (.finite 28)

def event48164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24292⟩⟩) 0 ⟨16187⟩ 48163

def event48165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24292⟩⟩) (.authority (.programFamilyFact))

def event48166 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24292⟩⟩) (.finite 3720)

def event48167 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event48168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24293⟩⟩) 0 ⟨6689⟩ 48167

def event48169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24293⟩⟩) 1 ⟨24292⟩ 48166

def event48170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24293⟩⟩) (.authority (.operator))

def exact48171RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (1)⟩]

theorem exact48171RawTermsValid :
    exact48171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48171 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24293⟩⟩) exact48171RawTerms .large 48170 .exactZero (none)

def event48172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28319⟩⟩) 0 ⟨24293⟩ 48171

def event48173 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28319⟩⟩) (.authority (.operator))

def exact48174RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (1)⟩]

theorem exact48174RawTermsValid :
    exact48174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48174 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28319⟩⟩) exact48174RawTerms (.finite 8192) 48173 .exactZero (none)

def event48175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event48176 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event48177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16226⟩⟩) 0 ⟨16187⟩ 48163

def event48178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16226⟩⟩) 1 ⟨110⟩ 48176

def event48179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16226⟩⟩) (.sum [.predecessor 0 48177 .coefficient, .predecessor 1 48178 .coefficient])

def event48180 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16226⟩⟩) (.finite 28)

def event48181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16227⟩⟩) 0 ⟨16226⟩ 48180

def event48182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16227⟩⟩) (.identity (.predecessor 0 48181 .coefficient))

def exact48183RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], []⟩, (1)⟩]

theorem exact48183RawTermsValid :
    exact48183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48183 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16227⟩⟩) exact48183RawTerms (.finite 28) 48182 .exactZero (none)

def event48184 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact48185RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48185RawTermsValid :
    exact48185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48185 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact48185RawTerms .large 48184 .exactZero (none)

def event48186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16228⟩⟩) 0 ⟨6544⟩ 48185

def event48187 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16228⟩⟩) 1 ⟨16227⟩ 48183

def event48188 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16228⟩⟩) (.product (.predecessor 0 48186 .coefficient) (.predecessor 1 48187 .coefficient) (⟨false, false, none, none, none⟩))

def event48189 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16228⟩⟩, .operator (⟨48185, 0⟩, ⟨48183, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48190RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48190RawTermsValid :
    exact48190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48190 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16228⟩⟩) exact48190RawTerms .large 48188 .exactZero (none)

def event48191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 48167

def event48192 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact48193RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact48193RawTermsValid :
    exact48193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48193 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact48193RawTerms .large 48192 .exactZero (none)

def event48194 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16229⟩⟩) 0 ⟨6699⟩ 48193

def event48195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16229⟩⟩) 1 ⟨16228⟩ 48190

def event48196 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16229⟩⟩) (.sum [.predecessor 0 48194 .coefficient, .predecessor 1 48195 .coefficient])

def exact48197RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48197RawTermsValid :
    exact48197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48197 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16229⟩⟩) exact48197RawTerms .large 48196 .exactZero (none)

def event48198 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28320⟩⟩) 0 ⟨16229⟩ 48197

def event48199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28320⟩⟩) 1 ⟨28319⟩ 48174

def event48200 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28320⟩⟩) (.product (.predecessor 0 48198 .coefficient) (.predecessor 1 48199 .coefficient) (⟨false, false, none, none, none⟩))

def event48201 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28320⟩⟩, .operator (⟨48197, 0⟩, ⟨48174, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (1)⟩)

def event48202 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28320⟩⟩, .operator (⟨48197, 1⟩, ⟨48174, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (-1)⟩)

def event48203 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28320⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28319⟩⟩) ⟨24293⟩ 48171)

def event48204 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28320⟩⟩, .relation 48203 0, ⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (-1)⟩)

def exact48205RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (-1)⟩]

theorem exact48205RawTermsValid :
    exact48205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48205 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28320⟩⟩) exact48205RawTerms .large 48200 .exactZero (none)

def event48206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17670⟩⟩) 0 ⟨16187⟩ 48163

def event48207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17670⟩⟩) (.authority (.programFamilyFact))

def exact48208RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17670⟩⟩], []⟩, (1)⟩]

theorem exact48208RawTermsValid :
    exact48208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48208 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17670⟩⟩) exact48208RawTerms (.finite 28) 48207 .exactZero (none)

def event48209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17672⟩⟩) 0 ⟨6544⟩ 48185

def event48210 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17672⟩⟩) 1 ⟨17670⟩ 48208

def event48211 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17672⟩⟩) (.product (.predecessor 0 48209 .coefficient) (.predecessor 1 48210 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48212 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17672⟩⟩, .operator (⟨48185, 0⟩, ⟨48208, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48213RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48213RawTermsValid :
    exact48213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48213 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17672⟩⟩) exact48213RawTerms .large 48211 .exactZero (none)

def event48214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6726⟩⟩) 0 ⟨6689⟩ 48167

def event48215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6726⟩⟩) (.authority (.operator))

def exact48216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩]

theorem exact48216RawTermsValid :
    exact48216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48216 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6726⟩⟩) exact48216RawTerms .large 48215 .exactZero (none)

def event48217 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17673⟩⟩) 0 ⟨6726⟩ 48216

def event48218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17673⟩⟩) 1 ⟨17672⟩ 48213

def event48219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17673⟩⟩) (.sum [.predecessor 0 48217 .coefficient, .predecessor 1 48218 .coefficient])

def exact48220RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48220RawTermsValid :
    exact48220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48220 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17673⟩⟩) exact48220RawTerms .large 48219 .exactZero (none)

def event48221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28325⟩⟩) 0 ⟨17673⟩ 48220

def event48222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28325⟩⟩) 1 ⟨28320⟩ 48205

def event48223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28325⟩⟩) (.sum [.predecessor 0 48221 .coefficient, .predecessor 1 48222 .coefficient])

def exact48224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48224RawTermsValid :
    exact48224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48224 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28325⟩⟩) exact48224RawTerms .large 48223 .exactZero (none)

def event48225 : Event := .preFoldPolynomial 48224 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact48226RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event48226 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28325⟩⟩) 48225 exact48226RawTerms .large 48223 .exactZero (none)

def event48227 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16187⟩⟩) ⟨⟨139⟩, ⟨47⟩, ⟨109⟩⟩ ⟨48069, 48227⟩

def event48228 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21627⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩) (1) 0 2 (.universal 48227 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩) (none) 48226)

def event48229 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21627⟩⟩, .relation 48228 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩)

def event48230 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21627⟩⟩, .relation 48228 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (-1)⟩)

def event48231 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21627⟩⟩, .relation 48228 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (1)⟩)

def event48232 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21627⟩⟩, .relation 48228 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48233RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48233RawTermsValid :
    exact48233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48233 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21627⟩⟩) exact48233RawTerms .large 48065 (.finite 1811303510016) (some (48067))

def event48234 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28322⟩⟩) 0 ⟨21627⟩ 48233

def event48235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28322⟩⟩) 1 ⟨28321⟩ 48055

def event48236 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28322⟩⟩) (.sum [.predecessor 0 48234 .coefficient, .predecessor 1 48235 .coefficient])

def event48237 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28322⟩⟩, .operator (⟨48233, 0⟩, ⟨48055, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (1)⟩)

def event48238 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28322⟩⟩, .operator (⟨48233, 2⟩, ⟨48055, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (-1)⟩)

def event48239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28322⟩⟩) (.sum [.result 48233 .summary, .result 48055 .summary])

def exact48240RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48240RawTermsValid :
    exact48240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48240 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28322⟩⟩) exact48240RawTerms .large 48236 (.finite 1292180536164689260544) (some (48239))

def event48241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28323⟩⟩) 0 ⟨28322⟩ 48240

def event48242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28323⟩⟩) 1 ⟨6682⟩ 5679

def event48243 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28323⟩⟩) (.product (.predecessor 0 48241 .coefficient) (.predecessor 1 48242 .coefficient) (⟨false, false, none, none, none⟩))

def event48244 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28323⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) [⟨.result 5675 .coefficient, false, none⟩])

def event48245 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28323⟩⟩) (.product (.result 48240 .summary) (.transfer 48244) (⟨false, false, none, none, none⟩))

def event48246 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28323⟩⟩, .operator (⟨48240, 0⟩, ⟨5679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩)

def event48247 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28323⟩⟩, .operator (⟨48240, 1⟩, ⟨5679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (-1)⟩)

def event48248 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28323⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6681⟩⟩) ⟨6612⟩ 5672)

def event48249 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28323⟩⟩, .relation 48248 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48250RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48250RawTermsValid :
    exact48250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48250 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28323⟩⟩) exact48250RawTerms .large 48243 (.finite 4742323242612988221224648704) (some (48245))

def event48251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24230⟩⟩) 0 ⟨6689⟩ 5477

def event48252 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24230⟩⟩) 1 ⟨24229⟩ 40377

def event48253 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24230⟩⟩) (.authority (.operator))

def exact48254RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (1)⟩]

theorem exact48254RawTermsValid :
    exact48254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48254 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24230⟩⟩) exact48254RawTerms .large 48253 .exactZero (none)

def event48255 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28102⟩⟩) 0 ⟨24230⟩ 48254

def event48256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28102⟩⟩) (.authority (.operator))

def exact48257RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (1)⟩]

theorem exact48257RawTermsValid :
    exact48257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48257 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28102⟩⟩) exact48257RawTerms (.finite 8192) 48256 .exactZero (none)

def event48258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28104⟩⟩) 0 ⟨26155⟩ 40661

def event48259 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28104⟩⟩) 1 ⟨28102⟩ 48257

def event48260 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28104⟩⟩) (.product (.predecessor 0 48258 .coefficient) (.predecessor 1 48259 .coefficient) (⟨false, false, none, none, none⟩))

def event48261 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28104⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩) [⟨.result 48257 .coefficient, false, none⟩])

def event48262 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28104⟩⟩) (.product (.result 40661 .summary) (.transfer 48261) (⟨false, false, none, none, none⟩))

def event48263 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28104⟩⟩, .operator (⟨40661, 0⟩, ⟨48257, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (1)⟩)

def event48264 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28104⟩⟩, .operator (⟨40661, 1⟩, ⟨48257, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (-1)⟩)

def event48265 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28104⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28102⟩⟩) ⟨24230⟩ 48254)

def event48266 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28104⟩⟩, .relation 48265 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (-1)⟩)

def exact48267RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (-1)⟩]

theorem exact48267RawTermsValid :
    exact48267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48267 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28104⟩⟩) exact48267RawTerms .large 48260 (.finite 1292113297018323992576) (some (48262))

def event48268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21480⟩⟩) 0 ⟨16068⟩ 1814

def event48269 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21480⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact48270RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩, (1)⟩]

theorem exact48270RawTermsValid :
    exact48270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48270 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21480⟩⟩) exact48270RawTerms (.finite 136065468) 48269 .exactZero (none)

def event48271 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21482⟩⟩) 0 ⟨21480⟩ 48270

def event48272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21482⟩⟩) 1 ⟨2348⟩ 4

def event48273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21482⟩⟩) (.scale (.predecessor 0 48271 .coefficient) (.value (.predecessor 1 48272 .coefficient)))

def exact48274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩, (1)⟩]

theorem exact48274RawTermsValid :
    exact48274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48274 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21482⟩⟩) exact48274RawTerms (.finite 136065468) 48273 .exactZero (none)

def event48275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21483⟩⟩) 0 ⟨5553⟩ 36137

def event48276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21483⟩⟩) 1 ⟨21482⟩ 48274

def event48277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21483⟩⟩) (.product (.predecessor 0 48275 .coefficient) (.predecessor 1 48276 .coefficient) (⟨false, false, none, none, none⟩))

def event48278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21483⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩) [⟨.result 48270 .coefficient, false, none⟩])

def event48279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21483⟩⟩) (.product (.result 36137 .summary) (.transfer 48278) (⟨false, false, none, none, none⟩))

def event48280 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21483⟩⟩, .operator (⟨36137, 0⟩, ⟨48274, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩, (1)⟩)

def event48281 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21481⟩⟩)

def event48282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event48283 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event48284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event48285 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event48286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event48287 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event48288 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event48289 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event48290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 48289

def event48291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 48287

def event48292 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 48290 .coefficient) (.value (.predecessor 1 48291 .coefficient)))

def event48293 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event48294 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 48293

def event48295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 48285

def event48296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 48294 .coefficient, .predecessor 1 48295 .coefficient])

def event48297 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event48298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 48297

def event48299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 48283

def event48300 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 48299 .coefficient))

def event48301 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event48302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11561⟩⟩) 0 ⟨5548⟩ 48301

def event48303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11561⟩⟩) (.authority (.programFamilyFact))

def exact48304RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11561⟩⟩], []⟩, (1)⟩]

theorem exact48304RawTermsValid :
    exact48304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48304 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11561⟩⟩) exact48304RawTerms (.finite 22) 48303 .exactZero (none)

def event48305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14442⟩⟩) 0 ⟨5548⟩ 48301

def event48306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14442⟩⟩) (.authority (.programFamilyFact))

def exact48307RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14442⟩⟩], []⟩, (1)⟩]

theorem exact48307RawTermsValid :
    exact48307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48307 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14442⟩⟩) exact48307RawTerms (.finite 22) 48306 .exactZero (none)

def event48308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14443⟩⟩) 0 ⟨14442⟩ 48307

def event48309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14443⟩⟩) 1 ⟨11561⟩ 48304

def event48310 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14443⟩⟩) (.product (.predecessor 0 48308 .coefficient) (.predecessor 1 48309 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14443⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11561⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], []⟩) [⟨.result 48307 .coefficient, true, some 1⟩, ⟨.result 48304 .coefficient, true, some 1⟩])

def event48312 : Event := .survivorFold (1) 48311

def exact48313RawTerms : List Term := []

theorem exact48313RawTermsValid :
    exact48313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48313 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14443⟩⟩) exact48313RawTerms (.finite 484) 48310 (.finite 484) (some (48311))

def event48314 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14444⟩⟩) 0 ⟨14443⟩ 48313

def event48315 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14444⟩⟩) (.identity (.predecessor 0 48314 .coefficient))

def event48316 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14444⟩⟩) (.finite 484)

def event48317 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16067⟩⟩) 0 ⟨14444⟩ 48316

def event48318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16067⟩⟩) (.authority (.programFamilyFact))

def exact48319RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], []⟩, (1)⟩]

theorem exact48319RawTermsValid :
    exact48319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48319 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16067⟩⟩) exact48319RawTerms (.finite 22) 48318 .exactZero (none)

def event48320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16068⟩⟩) 0 ⟨16067⟩ 48319

def event48321 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16068⟩⟩) (.identity (.predecessor 0 48320 .coefficient))

def event48322 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16068⟩⟩) (.finite 22)

def event48323 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21480⟩⟩) 0 ⟨16068⟩ 48322

def event48324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21480⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact48325RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩, (1)⟩]

theorem exact48325RawTermsValid :
    exact48325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48325 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21480⟩⟩) exact48325RawTerms (.finite 136065468) 48324 .exactZero (none)

def event48326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact48327RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact48327RawTermsValid :
    exact48327RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48327 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact48327RawTerms .large 48326 .exactZero (none)

def event48328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21481⟩⟩) 0 ⟨6⟩ 48327

def event48329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21481⟩⟩) 1 ⟨21480⟩ 48325

def event48330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21481⟩⟩) (.product (.predecessor 0 48328 .coefficient) (.predecessor 1 48329 .coefficient) (⟨false, false, none, none, none⟩))

def event48331 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21481⟩⟩, .operator (⟨48327, 0⟩, ⟨48325, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩, (1)⟩)

def exact48332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩, (1)⟩]

theorem exact48332RawTermsValid :
    exact48332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48332 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21481⟩⟩) exact48332RawTerms .large 48330 .exactZero (none)

def event48333 : Event := .preFoldPolynomial 48332 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩, (1)⟩] .exactZero none

def exact48334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩, (1)⟩]

def event48334 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21481⟩⟩) 48333 exact48334RawTerms .large 48330 .exactZero (none)

def event48335 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28108⟩⟩)

def event48336 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event48337 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event48338 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event48339 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event48340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event48341 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event48342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event48343 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event48344 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 48343

def event48345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 48341

def event48346 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 48344 .coefficient) (.value (.predecessor 1 48345 .coefficient)))

def event48347 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event48348 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 48347

def event48349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 48339

def event48350 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 48348 .coefficient, .predecessor 1 48349 .coefficient])

def event48351 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event48352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 48351

def event48353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 48337

def event48354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 48353 .coefficient))

def event48355 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event48356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11561⟩⟩) 0 ⟨5548⟩ 48355

def event48357 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11561⟩⟩) (.authority (.programFamilyFact))

def exact48358RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11561⟩⟩], []⟩, (1)⟩]

theorem exact48358RawTermsValid :
    exact48358RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48358 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11561⟩⟩) exact48358RawTerms (.finite 22) 48357 .exactZero (none)

def event48359 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14442⟩⟩) 0 ⟨5548⟩ 48355

def event48360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14442⟩⟩) (.authority (.programFamilyFact))

def exact48361RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14442⟩⟩], []⟩, (1)⟩]

theorem exact48361RawTermsValid :
    exact48361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48361 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14442⟩⟩) exact48361RawTerms (.finite 22) 48360 .exactZero (none)

def event48362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14443⟩⟩) 0 ⟨14442⟩ 48361

def event48363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14443⟩⟩) 1 ⟨11561⟩ 48358

def event48364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14443⟩⟩) (.product (.predecessor 0 48362 .coefficient) (.predecessor 1 48363 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48365 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14443⟩⟩, .operator (⟨48361, 0⟩, ⟨48358, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11561⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], []⟩, (1)⟩)

def exact48366RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11561⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], []⟩, (1)⟩]

theorem exact48366RawTermsValid :
    exact48366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48366 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14443⟩⟩) exact48366RawTerms (.finite 484) 48364 .exactZero (none)

def event48367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14444⟩⟩) 0 ⟨14443⟩ 48366

def event48368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14444⟩⟩) (.identity (.predecessor 0 48367 .coefficient))

def event48369 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14444⟩⟩) (.finite 484)

def event48370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16067⟩⟩) 0 ⟨14444⟩ 48369

def event48371 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16067⟩⟩) (.authority (.programFamilyFact))

def exact48372RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], []⟩, (1)⟩]

theorem exact48372RawTermsValid :
    exact48372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48372 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16067⟩⟩) exact48372RawTerms (.finite 22) 48371 .exactZero (none)

def event48373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16068⟩⟩) 0 ⟨16067⟩ 48372

def event48374 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16068⟩⟩) (.identity (.predecessor 0 48373 .coefficient))

def event48375 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16068⟩⟩) (.finite 22)

def event48376 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24229⟩⟩) 0 ⟨16068⟩ 48375

def event48377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24229⟩⟩) (.authority (.programFamilyFact))

def event48378 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24229⟩⟩) (.finite 3720)

def event48379 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event48380 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24230⟩⟩) 0 ⟨6689⟩ 48379

def event48381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24230⟩⟩) 1 ⟨24229⟩ 48378

def event48382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24230⟩⟩) (.authority (.operator))

def exact48383RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (1)⟩]

theorem exact48383RawTermsValid :
    exact48383RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48383 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24230⟩⟩) exact48383RawTerms .large 48382 .exactZero (none)

def eventLeaf3008 : Array AnnotatedEvent := #[
  { event := event48128
    frameStart := 48123 },
  { event := event48129
    frameStart := 48123 },
  { event := event48130
    frameStart := 48123 },
  { event := event48131
    frameStart := 48123 },
  { event := event48132
    frameStart := 48123 },
  { event := event48133
    frameStart := 48123 },
  { event := event48134
    frameStart := 48123 },
  { event := event48135
    frameStart := 48123 },
  { event := event48136
    frameStart := 48123 },
  { event := event48137
    frameStart := 48123 },
  { event := event48138
    frameStart := 48123 },
  { event := event48139
    frameStart := 48123 },
  { event := event48140
    frameStart := 48123 },
  { event := event48141
    frameStart := 48123 },
  { event := event48142
    frameStart := 48123 },
  { event := event48143
    frameStart := 48123 }
]

def eventLeaf3009 : Array AnnotatedEvent := #[
  { event := event48144
    frameStart := 48123 },
  { event := event48145
    frameStart := 48123 },
  { event := event48146
    frameStart := 48123 },
  { event := event48147
    frameStart := 48123 },
  { event := event48148
    frameStart := 48123 },
  { event := event48149
    frameStart := 48123 },
  { event := event48150
    frameStart := 48123 },
  { event := event48151
    frameStart := 48123 },
  { event := event48152
    frameStart := 48123 },
  { event := event48153
    frameStart := 48123 },
  { event := event48154
    frameStart := 48123 },
  { event := event48155
    frameStart := 48123 },
  { event := event48156
    frameStart := 48123 },
  { event := event48157
    frameStart := 48123 },
  { event := event48158
    frameStart := 48123 },
  { event := event48159
    frameStart := 48123 }
]

def eventLeaf3010 : Array AnnotatedEvent := #[
  { event := event48160
    frameStart := 48123 },
  { event := event48161
    frameStart := 48123 },
  { event := event48162
    frameStart := 48123 },
  { event := event48163
    frameStart := 48123 },
  { event := event48164
    frameStart := 48123 },
  { event := event48165
    frameStart := 48123 },
  { event := event48166
    frameStart := 48123 },
  { event := event48167
    frameStart := 48123 },
  { event := event48168
    frameStart := 48123 },
  { event := event48169
    frameStart := 48123 },
  { event := event48170
    frameStart := 48123 },
  { event := event48171
    frameStart := 48123 },
  { event := event48172
    frameStart := 48123 },
  { event := event48173
    frameStart := 48123 },
  { event := event48174
    frameStart := 48123 },
  { event := event48175
    frameStart := 48123 }
]

def eventLeaf3011 : Array AnnotatedEvent := #[
  { event := event48176
    frameStart := 48123 },
  { event := event48177
    frameStart := 48123 },
  { event := event48178
    frameStart := 48123 },
  { event := event48179
    frameStart := 48123 },
  { event := event48180
    frameStart := 48123 },
  { event := event48181
    frameStart := 48123 },
  { event := event48182
    frameStart := 48123 },
  { event := event48183
    frameStart := 48123 },
  { event := event48184
    frameStart := 48123 },
  { event := event48185
    frameStart := 48123 },
  { event := event48186
    frameStart := 48123 },
  { event := event48187
    frameStart := 48123 },
  { event := event48188
    frameStart := 48123 },
  { event := event48189
    frameStart := 48123 },
  { event := event48190
    frameStart := 48123 },
  { event := event48191
    frameStart := 48123 }
]

def eventLeaf3012 : Array AnnotatedEvent := #[
  { event := event48192
    frameStart := 48123 },
  { event := event48193
    frameStart := 48123 },
  { event := event48194
    frameStart := 48123 },
  { event := event48195
    frameStart := 48123 },
  { event := event48196
    frameStart := 48123 },
  { event := event48197
    frameStart := 48123 },
  { event := event48198
    frameStart := 48123 },
  { event := event48199
    frameStart := 48123 },
  { event := event48200
    frameStart := 48123 },
  { event := event48201
    frameStart := 48123 },
  { event := event48202
    frameStart := 48123 },
  { event := event48203
    frameStart := 48123 },
  { event := event48204
    frameStart := 48123 },
  { event := event48205
    frameStart := 48123 },
  { event := event48206
    frameStart := 48123 },
  { event := event48207
    frameStart := 48123 }
]

def eventLeaf3013 : Array AnnotatedEvent := #[
  { event := event48208
    frameStart := 48123 },
  { event := event48209
    frameStart := 48123 },
  { event := event48210
    frameStart := 48123 },
  { event := event48211
    frameStart := 48123 },
  { event := event48212
    frameStart := 48123 },
  { event := event48213
    frameStart := 48123 },
  { event := event48214
    frameStart := 48123 },
  { event := event48215
    frameStart := 48123 },
  { event := event48216
    frameStart := 48123 },
  { event := event48217
    frameStart := 48123 },
  { event := event48218
    frameStart := 48123 },
  { event := event48219
    frameStart := 48123 },
  { event := event48220
    frameStart := 48123 },
  { event := event48221
    frameStart := 48123 },
  { event := event48222
    frameStart := 48123 },
  { event := event48223
    frameStart := 48123 }
]

def eventLeaf3014 : Array AnnotatedEvent := #[
  { event := event48224
    frameStart := 48123 },
  { event := event48225
    frameStart := 48123 },
  { event := event48226
    frameStart := 48123 },
  { event := event48227
    frameStart := 0 },
  { event := event48228
    frameStart := 0 },
  { event := event48229
    frameStart := 0 },
  { event := event48230
    frameStart := 0 },
  { event := event48231
    frameStart := 0 },
  { event := event48232
    frameStart := 0 },
  { event := event48233
    frameStart := 0 },
  { event := event48234
    frameStart := 0 },
  { event := event48235
    frameStart := 0 },
  { event := event48236
    frameStart := 0 },
  { event := event48237
    frameStart := 0 },
  { event := event48238
    frameStart := 0 },
  { event := event48239
    frameStart := 0 }
]

def eventLeaf3015 : Array AnnotatedEvent := #[
  { event := event48240
    frameStart := 0 },
  { event := event48241
    frameStart := 0 },
  { event := event48242
    frameStart := 0 },
  { event := event48243
    frameStart := 0 },
  { event := event48244
    frameStart := 0 },
  { event := event48245
    frameStart := 0 },
  { event := event48246
    frameStart := 0 },
  { event := event48247
    frameStart := 0 },
  { event := event48248
    frameStart := 0 },
  { event := event48249
    frameStart := 0 },
  { event := event48250
    frameStart := 0 },
  { event := event48251
    frameStart := 0 },
  { event := event48252
    frameStart := 0 },
  { event := event48253
    frameStart := 0 },
  { event := event48254
    frameStart := 0 },
  { event := event48255
    frameStart := 0 }
]

def eventLeaf3016 : Array AnnotatedEvent := #[
  { event := event48256
    frameStart := 0 },
  { event := event48257
    frameStart := 0 },
  { event := event48258
    frameStart := 0 },
  { event := event48259
    frameStart := 0 },
  { event := event48260
    frameStart := 0 },
  { event := event48261
    frameStart := 0 },
  { event := event48262
    frameStart := 0 },
  { event := event48263
    frameStart := 0 },
  { event := event48264
    frameStart := 0 },
  { event := event48265
    frameStart := 0 },
  { event := event48266
    frameStart := 0 },
  { event := event48267
    frameStart := 0 },
  { event := event48268
    frameStart := 0 },
  { event := event48269
    frameStart := 0 },
  { event := event48270
    frameStart := 0 },
  { event := event48271
    frameStart := 0 }
]

def eventLeaf3017 : Array AnnotatedEvent := #[
  { event := event48272
    frameStart := 0 },
  { event := event48273
    frameStart := 0 },
  { event := event48274
    frameStart := 0 },
  { event := event48275
    frameStart := 0 },
  { event := event48276
    frameStart := 0 },
  { event := event48277
    frameStart := 0 },
  { event := event48278
    frameStart := 0 },
  { event := event48279
    frameStart := 0 },
  { event := event48280
    frameStart := 0 },
  { event := event48281
    frameStart := 48281 },
  { event := event48282
    frameStart := 48281 },
  { event := event48283
    frameStart := 48281 },
  { event := event48284
    frameStart := 48281 },
  { event := event48285
    frameStart := 48281 },
  { event := event48286
    frameStart := 48281 },
  { event := event48287
    frameStart := 48281 }
]

def eventLeaf3018 : Array AnnotatedEvent := #[
  { event := event48288
    frameStart := 48281 },
  { event := event48289
    frameStart := 48281 },
  { event := event48290
    frameStart := 48281 },
  { event := event48291
    frameStart := 48281 },
  { event := event48292
    frameStart := 48281 },
  { event := event48293
    frameStart := 48281 },
  { event := event48294
    frameStart := 48281 },
  { event := event48295
    frameStart := 48281 },
  { event := event48296
    frameStart := 48281 },
  { event := event48297
    frameStart := 48281 },
  { event := event48298
    frameStart := 48281 },
  { event := event48299
    frameStart := 48281 },
  { event := event48300
    frameStart := 48281 },
  { event := event48301
    frameStart := 48281 },
  { event := event48302
    frameStart := 48281 },
  { event := event48303
    frameStart := 48281 }
]

def eventLeaf3019 : Array AnnotatedEvent := #[
  { event := event48304
    frameStart := 48281 },
  { event := event48305
    frameStart := 48281 },
  { event := event48306
    frameStart := 48281 },
  { event := event48307
    frameStart := 48281 },
  { event := event48308
    frameStart := 48281 },
  { event := event48309
    frameStart := 48281 },
  { event := event48310
    frameStart := 48281 },
  { event := event48311
    frameStart := 48281 },
  { event := event48312
    frameStart := 48281 },
  { event := event48313
    frameStart := 48281 },
  { event := event48314
    frameStart := 48281 },
  { event := event48315
    frameStart := 48281 },
  { event := event48316
    frameStart := 48281 },
  { event := event48317
    frameStart := 48281 },
  { event := event48318
    frameStart := 48281 },
  { event := event48319
    frameStart := 48281 }
]

def eventLeaf3020 : Array AnnotatedEvent := #[
  { event := event48320
    frameStart := 48281 },
  { event := event48321
    frameStart := 48281 },
  { event := event48322
    frameStart := 48281 },
  { event := event48323
    frameStart := 48281 },
  { event := event48324
    frameStart := 48281 },
  { event := event48325
    frameStart := 48281 },
  { event := event48326
    frameStart := 48281 },
  { event := event48327
    frameStart := 48281 },
  { event := event48328
    frameStart := 48281 },
  { event := event48329
    frameStart := 48281 },
  { event := event48330
    frameStart := 48281 },
  { event := event48331
    frameStart := 48281 },
  { event := event48332
    frameStart := 48281 },
  { event := event48333
    frameStart := 48281 },
  { event := event48334
    frameStart := 48281 },
  { event := event48335
    frameStart := 48335 }
]

def eventLeaf3021 : Array AnnotatedEvent := #[
  { event := event48336
    frameStart := 48335 },
  { event := event48337
    frameStart := 48335 },
  { event := event48338
    frameStart := 48335 },
  { event := event48339
    frameStart := 48335 },
  { event := event48340
    frameStart := 48335 },
  { event := event48341
    frameStart := 48335 },
  { event := event48342
    frameStart := 48335 },
  { event := event48343
    frameStart := 48335 },
  { event := event48344
    frameStart := 48335 },
  { event := event48345
    frameStart := 48335 },
  { event := event48346
    frameStart := 48335 },
  { event := event48347
    frameStart := 48335 },
  { event := event48348
    frameStart := 48335 },
  { event := event48349
    frameStart := 48335 },
  { event := event48350
    frameStart := 48335 },
  { event := event48351
    frameStart := 48335 }
]

def eventLeaf3022 : Array AnnotatedEvent := #[
  { event := event48352
    frameStart := 48335 },
  { event := event48353
    frameStart := 48335 },
  { event := event48354
    frameStart := 48335 },
  { event := event48355
    frameStart := 48335 },
  { event := event48356
    frameStart := 48335 },
  { event := event48357
    frameStart := 48335 },
  { event := event48358
    frameStart := 48335 },
  { event := event48359
    frameStart := 48335 },
  { event := event48360
    frameStart := 48335 },
  { event := event48361
    frameStart := 48335 },
  { event := event48362
    frameStart := 48335 },
  { event := event48363
    frameStart := 48335 },
  { event := event48364
    frameStart := 48335 },
  { event := event48365
    frameStart := 48335 },
  { event := event48366
    frameStart := 48335 },
  { event := event48367
    frameStart := 48335 }
]

def eventLeaf3023 : Array AnnotatedEvent := #[
  { event := event48368
    frameStart := 48335 },
  { event := event48369
    frameStart := 48335 },
  { event := event48370
    frameStart := 48335 },
  { event := event48371
    frameStart := 48335 },
  { event := event48372
    frameStart := 48335 },
  { event := event48373
    frameStart := 48335 },
  { event := event48374
    frameStart := 48335 },
  { event := event48375
    frameStart := 48335 },
  { event := event48376
    frameStart := 48335 },
  { event := event48377
    frameStart := 48335 },
  { event := event48378
    frameStart := 48335 },
  { event := event48379
    frameStart := 48335 },
  { event := event48380
    frameStart := 48335 },
  { event := event48381
    frameStart := 48335 },
  { event := event48382
    frameStart := 48335 },
  { event := event48383
    frameStart := 48335 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events188
