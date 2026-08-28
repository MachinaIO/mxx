import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events204

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event52224 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57492⟩⟩, .relation 52221 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨58017⟩⟩]⟩, (1)⟩)

def event52225 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57492⟩⟩, .relation 52221 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact52226RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨58017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52226RawTermsValid :
    exact52226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52226 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57492⟩⟩) exact52226RawTerms .large 52050 (.finite 202072841853861888) (some (52052))

def event52227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58569⟩⟩) 0 ⟨57492⟩ 52226

def event52228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58569⟩⟩) 1 ⟨58568⟩ 52040

def event52229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58569⟩⟩) (.sum [.predecessor 0 52227 .coefficient, .predecessor 1 52228 .coefficient])

def event52230 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58569⟩⟩, .operator (⟨52226, 2⟩, ⟨52040, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨58017⟩⟩]⟩, (-1)⟩)

def event52231 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58569⟩⟩, .operator (⟨52226, 1⟩, ⟨52040, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩, (1)⟩)

def event52232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58569⟩⟩) (.sum [.result 52226 .summary, .result 52040 .summary])

def exact52233RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52233RawTermsValid :
    exact52233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52233 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58569⟩⟩) exact52233RawTerms .large 52229 (.finite 2997944351807545540608) (some (52232))

def event52234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59162⟩⟩) 0 ⟨58569⟩ 52233

def event52235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59162⟩⟩) 1 ⟨59160⟩ 51956

def event52236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59162⟩⟩) (.product (.predecessor 0 52234 .coefficient) (.predecessor 1 52235 .coefficient) (⟨false, false, none, none, none⟩))

def event52237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59162⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩) [⟨.result 51956 .coefficient, false, none⟩])

def event52238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59162⟩⟩) (.product (.result 52233 .summary) (.transfer 52237) (⟨false, false, none, none, none⟩))

def event52239 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59162⟩⟩, .operator (⟨52233, 0⟩, ⟨51956, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (1)⟩)

def event52240 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59162⟩⟩, .operator (⟨52233, 1⟩, ⟨51956, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (-1)⟩)

def event52241 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨59162⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59160⟩⟩) ⟨58193⟩ 51953)

def event52242 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59162⟩⟩, .relation 52241 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (-1)⟩)

def exact52243RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (-1)⟩]

theorem exact52243RawTermsValid :
    exact52243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52243 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59162⟩⟩) exact52243RawTerms .large 52236 (.finite 32190182365603316457354999889920) (some (52238))

def event52244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57876⟩⟩) 0 ⟨56913⟩ 1860

def event52245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57876⟩⟩) (.authority (.relationPreimageSource ⟨70⟩))

def exact52246RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩, (1)⟩]

theorem exact52246RawTermsValid :
    exact52246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52246 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57876⟩⟩) exact52246RawTerms (.finite 5647228698) 52245 .exactZero (none)

def event52247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57878⟩⟩) 0 ⟨57876⟩ 52246

def event52248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57878⟩⟩) 1 ⟨2370⟩ 4

def event52249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57878⟩⟩) (.scale (.predecessor 0 52247 .coefficient) (.value (.predecessor 1 52248 .coefficient)))

def exact52250RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩, (1)⟩]

theorem exact52250RawTermsValid :
    exact52250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52250 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57878⟩⟩) exact52250RawTerms (.finite 5647228698) 52249 .exactZero (none)

def event52251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57879⟩⟩) 0 ⟨11216⟩ 46745

def event52252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57879⟩⟩) 1 ⟨57878⟩ 52250

def event52253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57879⟩⟩) (.product (.predecessor 0 52251 .coefficient) (.predecessor 1 52252 .coefficient) (⟨false, false, none, none, none⟩))

def event52254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57879⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩) [⟨.result 52246 .coefficient, false, none⟩])

def event52255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57879⟩⟩) (.product (.result 46745 .summary) (.transfer 52254) (⟨false, false, none, none, none⟩))

def event52256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57879⟩⟩, .operator (⟨46745, 0⟩, ⟨52250, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩, (1)⟩)

def event52257 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨57877⟩⟩)

def event52258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event52259 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event52260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event52261 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event52262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event52263 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event52264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event52265 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event52266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 52265

def event52267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 52263

def event52268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 52266 .coefficient) (.value (.predecessor 1 52267 .coefficient)))

def event52269 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event52270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 52269

def event52271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 52261

def event52272 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 52270 .coefficient, .predecessor 1 52271 .coefficient])

def event52273 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event52274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 52273

def event52275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 52259

def event52276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 52275 .coefficient))

def event52277 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event52278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25106⟩⟩) 0 ⟨11173⟩ 52277

def event52279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25106⟩⟩) (.authority (.programFamilyFact))

def exact52280RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩], []⟩, (1)⟩]

theorem exact52280RawTermsValid :
    exact52280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25106⟩⟩) exact52280RawTerms (.finite 16) 52279 .exactZero (none)

def event52281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56721⟩⟩) 0 ⟨11173⟩ 52277

def event52282 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56721⟩⟩) (.authority (.programFamilyFact))

def exact52283RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56721⟩⟩], []⟩, (1)⟩]

theorem exact52283RawTermsValid :
    exact52283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52283 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56721⟩⟩) exact52283RawTerms (.finite 16) 52282 .exactZero (none)

def event52284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56722⟩⟩) 0 ⟨56721⟩ 52283

def event52285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56722⟩⟩) 1 ⟨25106⟩ 52280

def event52286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56722⟩⟩) (.product (.predecessor 0 52284 .coefficient) (.predecessor 1 52285 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56722⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], []⟩) [⟨.result 52283 .coefficient, true, some 1⟩, ⟨.result 52280 .coefficient, true, some 1⟩])

def event52288 : Event := .survivorFold (1) 52287

def exact52289RawTerms : List Term := []

theorem exact52289RawTermsValid :
    exact52289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56722⟩⟩) exact52289RawTerms (.finite 256) 52286 (.finite 256) (some (52287))

def event52290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56723⟩⟩) 0 ⟨56722⟩ 52289

def event52291 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56723⟩⟩) (.identity (.predecessor 0 52290 .coefficient))

def event52292 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56723⟩⟩) (.finite 256)

def event52293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56912⟩⟩) 0 ⟨56723⟩ 52292

def event52294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56912⟩⟩) (.authority (.programFamilyFact))

def exact52295RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], []⟩, (1)⟩]

theorem exact52295RawTermsValid :
    exact52295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52295 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56912⟩⟩) exact52295RawTerms (.finite 16) 52294 .exactZero (none)

def event52296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56913⟩⟩) 0 ⟨56912⟩ 52295

def event52297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56913⟩⟩) (.identity (.predecessor 0 52296 .coefficient))

def event52298 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56913⟩⟩) (.finite 16)

def event52299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57876⟩⟩) 0 ⟨56913⟩ 52298

def event52300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57876⟩⟩) (.authority (.relationPreimageSource ⟨70⟩))

def exact52301RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩, (1)⟩]

theorem exact52301RawTermsValid :
    exact52301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52301 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57876⟩⟩) exact52301RawTerms (.finite 5647228698) 52300 .exactZero (none)

def event52302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact52303RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact52303RawTermsValid :
    exact52303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52303 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact52303RawTerms .large 52302 .exactZero (none)

def event52304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57877⟩⟩) 0 ⟨35⟩ 52303

def event52305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57877⟩⟩) 1 ⟨57876⟩ 52301

def event52306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57877⟩⟩) (.product (.predecessor 0 52304 .coefficient) (.predecessor 1 52305 .coefficient) (⟨false, false, none, none, none⟩))

def event52307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57877⟩⟩, .operator (⟨52303, 0⟩, ⟨52301, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩, (1)⟩)

def exact52308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩, (1)⟩]

theorem exact52308RawTermsValid :
    exact52308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57877⟩⟩) exact52308RawTerms .large 52306 .exactZero (none)

def event52309 : Event := .preFoldPolynomial 52308 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩, (1)⟩] .exactZero none

def exact52310RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩, (1)⟩]

def event52310 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨57877⟩⟩) 52309 exact52310RawTerms .large 52306 .exactZero (none)

def event52311 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨59165⟩⟩)

def event52312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event52313 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event52314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event52315 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event52316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event52317 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event52318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event52319 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event52320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 52319

def event52321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 52317

def event52322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 52320 .coefficient) (.value (.predecessor 1 52321 .coefficient)))

def event52323 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event52324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 52323

def event52325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 52315

def event52326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 52324 .coefficient, .predecessor 1 52325 .coefficient])

def event52327 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event52328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 52327

def event52329 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 52313

def event52330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 52329 .coefficient))

def event52331 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event52332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25106⟩⟩) 0 ⟨11173⟩ 52331

def event52333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25106⟩⟩) (.authority (.programFamilyFact))

def exact52334RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩], []⟩, (1)⟩]

theorem exact52334RawTermsValid :
    exact52334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25106⟩⟩) exact52334RawTerms (.finite 16) 52333 .exactZero (none)

def event52335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56721⟩⟩) 0 ⟨11173⟩ 52331

def event52336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56721⟩⟩) (.authority (.programFamilyFact))

def exact52337RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56721⟩⟩], []⟩, (1)⟩]

theorem exact52337RawTermsValid :
    exact52337RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52337 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56721⟩⟩) exact52337RawTerms (.finite 16) 52336 .exactZero (none)

def event52338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56722⟩⟩) 0 ⟨56721⟩ 52337

def event52339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56722⟩⟩) 1 ⟨25106⟩ 52334

def event52340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56722⟩⟩) (.product (.predecessor 0 52338 .coefficient) (.predecessor 1 52339 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52341 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56722⟩⟩, .operator (⟨52337, 0⟩, ⟨52334, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], []⟩, (1)⟩)

def exact52342RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], []⟩, (1)⟩]

theorem exact52342RawTermsValid :
    exact52342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56722⟩⟩) exact52342RawTerms (.finite 256) 52340 .exactZero (none)

def event52343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56723⟩⟩) 0 ⟨56722⟩ 52342

def event52344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56723⟩⟩) (.identity (.predecessor 0 52343 .coefficient))

def event52345 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56723⟩⟩) (.finite 256)

def event52346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56912⟩⟩) 0 ⟨56723⟩ 52345

def event52347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56912⟩⟩) (.authority (.programFamilyFact))

def exact52348RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], []⟩, (1)⟩]

theorem exact52348RawTermsValid :
    exact52348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52348 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56912⟩⟩) exact52348RawTerms (.finite 16) 52347 .exactZero (none)

def event52349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56913⟩⟩) 0 ⟨56912⟩ 52348

def event52350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56913⟩⟩) (.identity (.predecessor 0 52349 .coefficient))

def event52351 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56913⟩⟩) (.finite 16)

def event52352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58191⟩⟩) 0 ⟨56913⟩ 52351

def event52353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58191⟩⟩) (.authority (.programFamilyFact))

def event52354 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58191⟩⟩) (.finite 3720)

def event52355 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event52356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58193⟩⟩) 0 ⟨7177⟩ 52355

def event52357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58193⟩⟩) 1 ⟨58191⟩ 52354

def event52358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58193⟩⟩) (.authority (.operator))

def exact52359RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (1)⟩]

theorem exact52359RawTermsValid :
    exact52359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58193⟩⟩) exact52359RawTerms .large 52358 .exactZero (none)

def event52360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59160⟩⟩) 0 ⟨58193⟩ 52359

def event52361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59160⟩⟩) (.authority (.operator))

def exact52362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (1)⟩]

theorem exact52362RawTermsValid :
    exact52362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59160⟩⟩) exact52362RawTerms (.finite 8192) 52361 .exactZero (none)

def event52363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event52364 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event52365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58358⟩⟩) 0 ⟨56913⟩ 52351

def event52366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58358⟩⟩) 1 ⟨136⟩ 52364

def event52367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58358⟩⟩) (.sum [.predecessor 0 52365 .coefficient, .predecessor 1 52366 .coefficient])

def event52368 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58358⟩⟩) (.finite 16)

def event52369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58359⟩⟩) 0 ⟨58358⟩ 52368

def event52370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58359⟩⟩) (.identity (.predecessor 0 52369 .coefficient))

def exact52371RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], []⟩, (1)⟩]

theorem exact52371RawTermsValid :
    exact52371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52371 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58359⟩⟩) exact52371RawTerms (.finite 16) 52370 .exactZero (none)

def event52372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact52373RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52373RawTermsValid :
    exact52373RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52373 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact52373RawTerms .large 52372 .exactZero (none)

def event52374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58360⟩⟩) 0 ⟨6908⟩ 52373

def event52375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58360⟩⟩) 1 ⟨58359⟩ 52371

def event52376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58360⟩⟩) (.product (.predecessor 0 52374 .coefficient) (.predecessor 1 52375 .coefficient) (⟨false, false, none, none, none⟩))

def event52377 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58360⟩⟩, .operator (⟨52373, 0⟩, ⟨52371, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact52378RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52378RawTermsValid :
    exact52378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58360⟩⟩) exact52378RawTerms .large 52376 .exactZero (none)

def event52379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7185⟩⟩) 0 ⟨7177⟩ 52355

def event52380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7185⟩⟩) (.authority (.operator))

def exact52381RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩]

theorem exact52381RawTermsValid :
    exact52381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52381 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7185⟩⟩) exact52381RawTerms .large 52380 .exactZero (none)

def event52382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58361⟩⟩) 0 ⟨7185⟩ 52381

def event52383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58361⟩⟩) 1 ⟨58360⟩ 52378

def event52384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58361⟩⟩) (.sum [.predecessor 0 52382 .coefficient, .predecessor 1 52383 .coefficient])

def exact52385RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52385RawTermsValid :
    exact52385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58361⟩⟩) exact52385RawTerms .large 52384 .exactZero (none)

def event52386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59161⟩⟩) 0 ⟨58361⟩ 52385

def event52387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59161⟩⟩) 1 ⟨59160⟩ 52362

def event52388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59161⟩⟩) (.product (.predecessor 0 52386 .coefficient) (.predecessor 1 52387 .coefficient) (⟨false, false, none, none, none⟩))

def event52389 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59161⟩⟩, .operator (⟨52385, 0⟩, ⟨52362, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (1)⟩)

def event52390 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59161⟩⟩, .operator (⟨52385, 1⟩, ⟨52362, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (-1)⟩)

def event52391 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨59161⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59160⟩⟩) ⟨58193⟩ 52359)

def event52392 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59161⟩⟩, .relation 52391 0, ⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (-1)⟩)

def exact52393RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (-1)⟩]

theorem exact52393RawTermsValid :
    exact52393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52393 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59161⟩⟩) exact52393RawTerms .large 52388 .exactZero (none)

def event52394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57273⟩⟩) 0 ⟨56913⟩ 52351

def event52395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57273⟩⟩) (.authority (.programFamilyFact))

def exact52396RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57273⟩⟩], []⟩, (1)⟩]

theorem exact52396RawTermsValid :
    exact52396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57273⟩⟩) exact52396RawTerms (.finite 60) 52395 .exactZero (none)

def event52397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57275⟩⟩) 0 ⟨6908⟩ 52373

def event52398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57275⟩⟩) 1 ⟨57273⟩ 52396

def event52399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57275⟩⟩) (.product (.predecessor 0 52397 .coefficient) (.predecessor 1 52398 .coefficient) (⟨false, true, none, none, some 1⟩))

def event52400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57275⟩⟩, .operator (⟨52373, 0⟩, ⟨52396, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact52401RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52401RawTermsValid :
    exact52401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52401 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57275⟩⟩) exact52401RawTerms .large 52399 .exactZero (none)

def event52402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 52355

def event52403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact52404RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact52404RawTermsValid :
    exact52404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52404 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact52404RawTerms .large 52403 .exactZero (none)

def event52405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57276⟩⟩) 0 ⟨7210⟩ 52404

def event52406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57276⟩⟩) 1 ⟨57275⟩ 52401

def event52407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57276⟩⟩) (.sum [.predecessor 0 52405 .coefficient, .predecessor 1 52406 .coefficient])

def exact52408RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52408RawTermsValid :
    exact52408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52408 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57276⟩⟩) exact52408RawTerms .large 52407 .exactZero (none)

def event52409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59165⟩⟩) 0 ⟨57276⟩ 52408

def event52410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59165⟩⟩) 1 ⟨59161⟩ 52393

def event52411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59165⟩⟩) (.sum [.predecessor 0 52409 .coefficient, .predecessor 1 52410 .coefficient])

def exact52412RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52412RawTermsValid :
    exact52412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52412 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59165⟩⟩) exact52412RawTerms .large 52411 .exactZero (none)

def event52413 : Event := .preFoldPolynomial 52412 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact52414RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event52414 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨59165⟩⟩) 52413 exact52414RawTerms .large 52411 .exactZero (none)

def event52415 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨56913⟩⟩) ⟨⟨89⟩, ⟨70⟩, ⟨135⟩⟩ ⟨52257, 52415⟩

def event52416 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨57879⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩) (1) 0 2 (.universal 52415 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57876⟩⟩]⟩) (none) 52414)

def event52417 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57879⟩⟩, .relation 52416 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩)

def event52418 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57879⟩⟩, .relation 52416 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (-1)⟩)

def event52419 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57879⟩⟩, .relation 52416 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (1)⟩)

def event52420 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57879⟩⟩, .relation 52416 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact52421RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52421RawTermsValid :
    exact52421RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52421 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57879⟩⟩) exact52421RawTerms .large 52253 (.finite 202072841853861888) (some (52255))

def event52422 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59163⟩⟩) 0 ⟨57879⟩ 52421

def event52423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59163⟩⟩) 1 ⟨59162⟩ 52243

def event52424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59163⟩⟩) (.sum [.predecessor 0 52422 .coefficient, .predecessor 1 52423 .coefficient])

def event52425 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59163⟩⟩, .operator (⟨52421, 0⟩, ⟨52243, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59160⟩⟩]⟩, (1)⟩)

def event52426 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59163⟩⟩, .operator (⟨52421, 2⟩, ⟨52243, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨58193⟩⟩]⟩, (-1)⟩)

def event52427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59163⟩⟩) (.sum [.result 52421 .summary, .result 52243 .summary])

def exact52428RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52428RawTermsValid :
    exact52428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52428 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59163⟩⟩) exact52428RawTerms .large 52424 (.finite 32190182365603518530196853751808) (some (52427))

def event52429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55211⟩⟩) 0 ⟨53933⟩ 1883

def event52430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55211⟩⟩) (.authority (.programFamilyFact))

def event52431 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55211⟩⟩) (.finite 3720)

def event52432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55213⟩⟩) 0 ⟨7177⟩ 15500

def event52433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55213⟩⟩) 1 ⟨55211⟩ 52431

def event52434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55213⟩⟩) (.authority (.operator))

def exact52435RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (1)⟩]

theorem exact52435RawTermsValid :
    exact52435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52435 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55213⟩⟩) exact52435RawTerms .large 52434 .exactZero (none)

def event52436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56180⟩⟩) 0 ⟨55213⟩ 52435

def event52437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56180⟩⟩) (.authority (.operator))

def exact52438RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (1)⟩]

theorem exact52438RawTermsValid :
    exact52438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56180⟩⟩) exact52438RawTerms (.finite 8192) 52437 .exactZero (none)

def event52439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55036⟩⟩) 0 ⟨53743⟩ 1877

def event52440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55036⟩⟩) (.authority (.programFamilyFact))

def event52441 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55036⟩⟩) (.finite 3720)

def event52442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55037⟩⟩) 0 ⟨7177⟩ 15500

def event52443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55037⟩⟩) 1 ⟨55036⟩ 52441

def event52444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55037⟩⟩) (.authority (.operator))

def exact52445RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55037⟩⟩]⟩, (1)⟩]

theorem exact52445RawTermsValid :
    exact52445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52445 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55037⟩⟩) exact52445RawTerms .large 52444 .exactZero (none)

def event52446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55587⟩⟩) 0 ⟨55037⟩ 52445

def event52447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55587⟩⟩) (.authority (.operator))

def exact52448RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55587⟩⟩]⟩, (1)⟩]

theorem exact52448RawTermsValid :
    exact52448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52448 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55587⟩⟩) exact52448RawTerms (.finite 8192) 52447 .exactZero (none)

def event52449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24867⟩⟩) 0 ⟨24866⟩ 1866

def event52450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24867⟩⟩) 1 ⟨11176⟩ 46653

def event52451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24867⟩⟩) (.tensor (.predecessor 0 52449 .coefficient) (.predecessor 1 52450 .coefficient) true false)

def event52452 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24867⟩⟩, .operator (⟨1866, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact52453RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52453RawTermsValid :
    exact52453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52453 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24867⟩⟩) exact52453RawTerms .large 52451 .exactZero (none)

def event52454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11178⟩⟩) 0 ⟨11175⟩ 46523

def event52455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11178⟩⟩) 1 ⟨7272⟩ 23092

def event52456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11178⟩⟩) (.product (.predecessor 0 52454 .coefficient) (.predecessor 1 52455 .coefficient) (⟨false, false, none, none, none⟩))

def event52457 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11178⟩⟩, .operator (⟨46523, 0⟩, ⟨23092, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩)

def exact52458RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩]

theorem exact52458RawTermsValid :
    exact52458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52458 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11178⟩⟩) exact52458RawTerms .large 52456 .exactZero (none)

def event52459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24868⟩⟩) 0 ⟨11178⟩ 52458

def event52460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24868⟩⟩) 1 ⟨24867⟩ 52453

def event52461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24868⟩⟩) (.sum [.predecessor 0 52459 .coefficient, .predecessor 1 52460 .coefficient])

def exact52462RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52462RawTermsValid :
    exact52462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52462 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24868⟩⟩) exact52462RawTerms .large 52461 .exactZero (none)

def event52463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24869⟩⟩) 0 ⟨24868⟩ 52462

def event52464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24869⟩⟩) 1 ⟨98⟩ 23084

def event52465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24869⟩⟩) (.sum [.predecessor 0 52463 .coefficient, .predecessor 1 52464 .coefficient])

def event52466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24869⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨98⟩⟩]⟩) [⟨.result 23084 .coefficient, false, none⟩])

def event52467 : Event := .survivorFold (1) 52466

def exact52468RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52468RawTermsValid :
    exact52468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52468 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24869⟩⟩) exact52468RawTerms .large 52465 (.finite 26) (some (52466))

def event52469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53744⟩⟩) 0 ⟨24869⟩ 52468

def event52470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53744⟩⟩) 1 ⟨53741⟩ 1869

def event52471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53744⟩⟩) (.product (.predecessor 0 52469 .coefficient) (.predecessor 1 52470 .coefficient) (⟨false, true, none, none, some 1⟩))

def event52472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53744⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨53741⟩⟩], []⟩) [⟨.result 1869 .coefficient, true, some 1⟩])

def event52473 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53744⟩⟩) (.product (.result 52468 .summary) (.transfer 52472) (⟨false, false, none, none, none⟩))

def event52474 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53744⟩⟩, .operator (⟨52468, 1⟩, ⟨1869, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event52475 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53744⟩⟩, .operator (⟨52468, 0⟩, ⟨1869, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩)

def exact52476RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩]

theorem exact52476RawTermsValid :
    exact52476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52476 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53744⟩⟩) exact52476RawTerms .large 52471 (.finite 10223616) (some (52473))

def event52477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53745⟩⟩) 0 ⟨53741⟩ 1869

def event52478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53745⟩⟩) 1 ⟨11176⟩ 46653

def event52479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53745⟩⟩) (.tensor (.predecessor 0 52477 .coefficient) (.predecessor 1 52478 .coefficient) true false)

def eventLeaf3264 : Array AnnotatedEvent := #[
  { event := event52224
    frameStart := 0 },
  { event := event52225
    frameStart := 0 },
  { event := event52226
    frameStart := 0 },
  { event := event52227
    frameStart := 0 },
  { event := event52228
    frameStart := 0 },
  { event := event52229
    frameStart := 0 },
  { event := event52230
    frameStart := 0 },
  { event := event52231
    frameStart := 0 },
  { event := event52232
    frameStart := 0 },
  { event := event52233
    frameStart := 0 },
  { event := event52234
    frameStart := 0 },
  { event := event52235
    frameStart := 0 },
  { event := event52236
    frameStart := 0 },
  { event := event52237
    frameStart := 0 },
  { event := event52238
    frameStart := 0 },
  { event := event52239
    frameStart := 0 }
]

def eventLeaf3265 : Array AnnotatedEvent := #[
  { event := event52240
    frameStart := 0 },
  { event := event52241
    frameStart := 0 },
  { event := event52242
    frameStart := 0 },
  { event := event52243
    frameStart := 0 },
  { event := event52244
    frameStart := 0 },
  { event := event52245
    frameStart := 0 },
  { event := event52246
    frameStart := 0 },
  { event := event52247
    frameStart := 0 },
  { event := event52248
    frameStart := 0 },
  { event := event52249
    frameStart := 0 },
  { event := event52250
    frameStart := 0 },
  { event := event52251
    frameStart := 0 },
  { event := event52252
    frameStart := 0 },
  { event := event52253
    frameStart := 0 },
  { event := event52254
    frameStart := 0 },
  { event := event52255
    frameStart := 0 }
]

def eventLeaf3266 : Array AnnotatedEvent := #[
  { event := event52256
    frameStart := 0 },
  { event := event52257
    frameStart := 52257 },
  { event := event52258
    frameStart := 52257 },
  { event := event52259
    frameStart := 52257 },
  { event := event52260
    frameStart := 52257 },
  { event := event52261
    frameStart := 52257 },
  { event := event52262
    frameStart := 52257 },
  { event := event52263
    frameStart := 52257 },
  { event := event52264
    frameStart := 52257 },
  { event := event52265
    frameStart := 52257 },
  { event := event52266
    frameStart := 52257 },
  { event := event52267
    frameStart := 52257 },
  { event := event52268
    frameStart := 52257 },
  { event := event52269
    frameStart := 52257 },
  { event := event52270
    frameStart := 52257 },
  { event := event52271
    frameStart := 52257 }
]

def eventLeaf3267 : Array AnnotatedEvent := #[
  { event := event52272
    frameStart := 52257 },
  { event := event52273
    frameStart := 52257 },
  { event := event52274
    frameStart := 52257 },
  { event := event52275
    frameStart := 52257 },
  { event := event52276
    frameStart := 52257 },
  { event := event52277
    frameStart := 52257 },
  { event := event52278
    frameStart := 52257 },
  { event := event52279
    frameStart := 52257 },
  { event := event52280
    frameStart := 52257 },
  { event := event52281
    frameStart := 52257 },
  { event := event52282
    frameStart := 52257 },
  { event := event52283
    frameStart := 52257 },
  { event := event52284
    frameStart := 52257 },
  { event := event52285
    frameStart := 52257 },
  { event := event52286
    frameStart := 52257 },
  { event := event52287
    frameStart := 52257 }
]

def eventLeaf3268 : Array AnnotatedEvent := #[
  { event := event52288
    frameStart := 52257 },
  { event := event52289
    frameStart := 52257 },
  { event := event52290
    frameStart := 52257 },
  { event := event52291
    frameStart := 52257 },
  { event := event52292
    frameStart := 52257 },
  { event := event52293
    frameStart := 52257 },
  { event := event52294
    frameStart := 52257 },
  { event := event52295
    frameStart := 52257 },
  { event := event52296
    frameStart := 52257 },
  { event := event52297
    frameStart := 52257 },
  { event := event52298
    frameStart := 52257 },
  { event := event52299
    frameStart := 52257 },
  { event := event52300
    frameStart := 52257 },
  { event := event52301
    frameStart := 52257 },
  { event := event52302
    frameStart := 52257 },
  { event := event52303
    frameStart := 52257 }
]

def eventLeaf3269 : Array AnnotatedEvent := #[
  { event := event52304
    frameStart := 52257 },
  { event := event52305
    frameStart := 52257 },
  { event := event52306
    frameStart := 52257 },
  { event := event52307
    frameStart := 52257 },
  { event := event52308
    frameStart := 52257 },
  { event := event52309
    frameStart := 52257 },
  { event := event52310
    frameStart := 52257 },
  { event := event52311
    frameStart := 52311 },
  { event := event52312
    frameStart := 52311 },
  { event := event52313
    frameStart := 52311 },
  { event := event52314
    frameStart := 52311 },
  { event := event52315
    frameStart := 52311 },
  { event := event52316
    frameStart := 52311 },
  { event := event52317
    frameStart := 52311 },
  { event := event52318
    frameStart := 52311 },
  { event := event52319
    frameStart := 52311 }
]

def eventLeaf3270 : Array AnnotatedEvent := #[
  { event := event52320
    frameStart := 52311 },
  { event := event52321
    frameStart := 52311 },
  { event := event52322
    frameStart := 52311 },
  { event := event52323
    frameStart := 52311 },
  { event := event52324
    frameStart := 52311 },
  { event := event52325
    frameStart := 52311 },
  { event := event52326
    frameStart := 52311 },
  { event := event52327
    frameStart := 52311 },
  { event := event52328
    frameStart := 52311 },
  { event := event52329
    frameStart := 52311 },
  { event := event52330
    frameStart := 52311 },
  { event := event52331
    frameStart := 52311 },
  { event := event52332
    frameStart := 52311 },
  { event := event52333
    frameStart := 52311 },
  { event := event52334
    frameStart := 52311 },
  { event := event52335
    frameStart := 52311 }
]

def eventLeaf3271 : Array AnnotatedEvent := #[
  { event := event52336
    frameStart := 52311 },
  { event := event52337
    frameStart := 52311 },
  { event := event52338
    frameStart := 52311 },
  { event := event52339
    frameStart := 52311 },
  { event := event52340
    frameStart := 52311 },
  { event := event52341
    frameStart := 52311 },
  { event := event52342
    frameStart := 52311 },
  { event := event52343
    frameStart := 52311 },
  { event := event52344
    frameStart := 52311 },
  { event := event52345
    frameStart := 52311 },
  { event := event52346
    frameStart := 52311 },
  { event := event52347
    frameStart := 52311 },
  { event := event52348
    frameStart := 52311 },
  { event := event52349
    frameStart := 52311 },
  { event := event52350
    frameStart := 52311 },
  { event := event52351
    frameStart := 52311 }
]

def eventLeaf3272 : Array AnnotatedEvent := #[
  { event := event52352
    frameStart := 52311 },
  { event := event52353
    frameStart := 52311 },
  { event := event52354
    frameStart := 52311 },
  { event := event52355
    frameStart := 52311 },
  { event := event52356
    frameStart := 52311 },
  { event := event52357
    frameStart := 52311 },
  { event := event52358
    frameStart := 52311 },
  { event := event52359
    frameStart := 52311 },
  { event := event52360
    frameStart := 52311 },
  { event := event52361
    frameStart := 52311 },
  { event := event52362
    frameStart := 52311 },
  { event := event52363
    frameStart := 52311 },
  { event := event52364
    frameStart := 52311 },
  { event := event52365
    frameStart := 52311 },
  { event := event52366
    frameStart := 52311 },
  { event := event52367
    frameStart := 52311 }
]

def eventLeaf3273 : Array AnnotatedEvent := #[
  { event := event52368
    frameStart := 52311 },
  { event := event52369
    frameStart := 52311 },
  { event := event52370
    frameStart := 52311 },
  { event := event52371
    frameStart := 52311 },
  { event := event52372
    frameStart := 52311 },
  { event := event52373
    frameStart := 52311 },
  { event := event52374
    frameStart := 52311 },
  { event := event52375
    frameStart := 52311 },
  { event := event52376
    frameStart := 52311 },
  { event := event52377
    frameStart := 52311 },
  { event := event52378
    frameStart := 52311 },
  { event := event52379
    frameStart := 52311 },
  { event := event52380
    frameStart := 52311 },
  { event := event52381
    frameStart := 52311 },
  { event := event52382
    frameStart := 52311 },
  { event := event52383
    frameStart := 52311 }
]

def eventLeaf3274 : Array AnnotatedEvent := #[
  { event := event52384
    frameStart := 52311 },
  { event := event52385
    frameStart := 52311 },
  { event := event52386
    frameStart := 52311 },
  { event := event52387
    frameStart := 52311 },
  { event := event52388
    frameStart := 52311 },
  { event := event52389
    frameStart := 52311 },
  { event := event52390
    frameStart := 52311 },
  { event := event52391
    frameStart := 52311 },
  { event := event52392
    frameStart := 52311 },
  { event := event52393
    frameStart := 52311 },
  { event := event52394
    frameStart := 52311 },
  { event := event52395
    frameStart := 52311 },
  { event := event52396
    frameStart := 52311 },
  { event := event52397
    frameStart := 52311 },
  { event := event52398
    frameStart := 52311 },
  { event := event52399
    frameStart := 52311 }
]

def eventLeaf3275 : Array AnnotatedEvent := #[
  { event := event52400
    frameStart := 52311 },
  { event := event52401
    frameStart := 52311 },
  { event := event52402
    frameStart := 52311 },
  { event := event52403
    frameStart := 52311 },
  { event := event52404
    frameStart := 52311 },
  { event := event52405
    frameStart := 52311 },
  { event := event52406
    frameStart := 52311 },
  { event := event52407
    frameStart := 52311 },
  { event := event52408
    frameStart := 52311 },
  { event := event52409
    frameStart := 52311 },
  { event := event52410
    frameStart := 52311 },
  { event := event52411
    frameStart := 52311 },
  { event := event52412
    frameStart := 52311 },
  { event := event52413
    frameStart := 52311 },
  { event := event52414
    frameStart := 52311 },
  { event := event52415
    frameStart := 0 }
]

def eventLeaf3276 : Array AnnotatedEvent := #[
  { event := event52416
    frameStart := 0 },
  { event := event52417
    frameStart := 0 },
  { event := event52418
    frameStart := 0 },
  { event := event52419
    frameStart := 0 },
  { event := event52420
    frameStart := 0 },
  { event := event52421
    frameStart := 0 },
  { event := event52422
    frameStart := 0 },
  { event := event52423
    frameStart := 0 },
  { event := event52424
    frameStart := 0 },
  { event := event52425
    frameStart := 0 },
  { event := event52426
    frameStart := 0 },
  { event := event52427
    frameStart := 0 },
  { event := event52428
    frameStart := 0 },
  { event := event52429
    frameStart := 0 },
  { event := event52430
    frameStart := 0 },
  { event := event52431
    frameStart := 0 }
]

def eventLeaf3277 : Array AnnotatedEvent := #[
  { event := event52432
    frameStart := 0 },
  { event := event52433
    frameStart := 0 },
  { event := event52434
    frameStart := 0 },
  { event := event52435
    frameStart := 0 },
  { event := event52436
    frameStart := 0 },
  { event := event52437
    frameStart := 0 },
  { event := event52438
    frameStart := 0 },
  { event := event52439
    frameStart := 0 },
  { event := event52440
    frameStart := 0 },
  { event := event52441
    frameStart := 0 },
  { event := event52442
    frameStart := 0 },
  { event := event52443
    frameStart := 0 },
  { event := event52444
    frameStart := 0 },
  { event := event52445
    frameStart := 0 },
  { event := event52446
    frameStart := 0 },
  { event := event52447
    frameStart := 0 }
]

def eventLeaf3278 : Array AnnotatedEvent := #[
  { event := event52448
    frameStart := 0 },
  { event := event52449
    frameStart := 0 },
  { event := event52450
    frameStart := 0 },
  { event := event52451
    frameStart := 0 },
  { event := event52452
    frameStart := 0 },
  { event := event52453
    frameStart := 0 },
  { event := event52454
    frameStart := 0 },
  { event := event52455
    frameStart := 0 },
  { event := event52456
    frameStart := 0 },
  { event := event52457
    frameStart := 0 },
  { event := event52458
    frameStart := 0 },
  { event := event52459
    frameStart := 0 },
  { event := event52460
    frameStart := 0 },
  { event := event52461
    frameStart := 0 },
  { event := event52462
    frameStart := 0 },
  { event := event52463
    frameStart := 0 }
]

def eventLeaf3279 : Array AnnotatedEvent := #[
  { event := event52464
    frameStart := 0 },
  { event := event52465
    frameStart := 0 },
  { event := event52466
    frameStart := 0 },
  { event := event52467
    frameStart := 0 },
  { event := event52468
    frameStart := 0 },
  { event := event52469
    frameStart := 0 },
  { event := event52470
    frameStart := 0 },
  { event := event52471
    frameStart := 0 },
  { event := event52472
    frameStart := 0 },
  { event := event52473
    frameStart := 0 },
  { event := event52474
    frameStart := 0 },
  { event := event52475
    frameStart := 0 },
  { event := event52476
    frameStart := 0 },
  { event := event52477
    frameStart := 0 },
  { event := event52478
    frameStart := 0 },
  { event := event52479
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events204
