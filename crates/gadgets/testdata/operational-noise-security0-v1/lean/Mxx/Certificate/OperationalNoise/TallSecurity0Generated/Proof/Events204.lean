import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events204

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event52224 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 52223

def event52225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 52221

def event52226 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 52224 .coefficient) (.value (.predecessor 1 52225 .coefficient)))

def event52227 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event52228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 52227

def event52229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 52219

def event52230 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 52228 .coefficient, .predecessor 1 52229 .coefficient])

def event52231 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event52232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 52231

def event52233 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 52217

def event52234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 52233 .coefficient))

def event52235 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event52236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12770⟩⟩) 0 ⟨5542⟩ 52235

def event52237 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12770⟩⟩) (.authority (.programFamilyFact))

def exact52238RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩, (1)⟩]

theorem exact52238RawTermsValid :
    exact52238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52238 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12770⟩⟩) exact52238RawTerms (.finite 46) 52237 .exactZero (none)

def event52239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10035⟩⟩) 0 ⟨5542⟩ 52235

def event52240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10035⟩⟩) (.authority (.programFamilyFact))

def exact52241RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩], []⟩, (1)⟩]

theorem exact52241RawTermsValid :
    exact52241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52241 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10035⟩⟩) exact52241RawTerms (.finite 46) 52240 .exactZero (none)

def event52242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12771⟩⟩) 0 ⟨10035⟩ 52241

def event52243 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12771⟩⟩) 1 ⟨12770⟩ 52238

def event52244 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12771⟩⟩) (.product (.predecessor 0 52242 .coefficient) (.predecessor 1 52243 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52245 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12771⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩) [⟨.result 52241 .coefficient, true, some 1⟩, ⟨.result 52238 .coefficient, true, some 1⟩])

def event52246 : Event := .survivorFold (1) 52245

def exact52247RawTerms : List Term := []

theorem exact52247RawTermsValid :
    exact52247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52247 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12771⟩⟩) exact52247RawTerms (.finite 2116) 52244 (.finite 2116) (some (52245))

def event52248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12772⟩⟩) 0 ⟨12771⟩ 52247

def event52249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12772⟩⟩) (.identity (.predecessor 0 52248 .coefficient))

def event52250 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12772⟩⟩) (.finite 2116)

def event52251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20036⟩⟩) 0 ⟨12772⟩ 52250

def event52252 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20036⟩⟩) (.authority (.relationPreimageSource ⟨23⟩))

def exact52253RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩, (1)⟩]

theorem exact52253RawTermsValid :
    exact52253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52253 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20036⟩⟩) exact52253RawTerms (.finite 136065468) 52252 .exactZero (none)

def event52254 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact52255RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact52255RawTermsValid :
    exact52255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52255 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact52255RawTerms .large 52254 .exactZero (none)

def event52256 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20037⟩⟩) 0 ⟨6⟩ 52255

def event52257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20037⟩⟩) 1 ⟨20036⟩ 52253

def event52258 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20037⟩⟩) (.product (.predecessor 0 52256 .coefficient) (.predecessor 1 52257 .coefficient) (⟨false, false, none, none, none⟩))

def event52259 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20037⟩⟩, .operator (⟨52255, 0⟩, ⟨52253, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩, (1)⟩)

def exact52260RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩, (1)⟩]

theorem exact52260RawTermsValid :
    exact52260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20037⟩⟩) exact52260RawTerms .large 52258 .exactZero (none)

def event52261 : Event := .preFoldPolynomial 52260 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩, (1)⟩] .exactZero none

def exact52262RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩, (1)⟩]

def event52262 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20037⟩⟩) 52261 exact52262RawTerms .large 52258 .exactZero (none)

def event52263 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25536⟩⟩)

def event52264 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event52265 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event52266 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event52267 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event52268 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event52269 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event52270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event52271 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event52272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 52271

def event52273 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 52269

def event52274 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 52272 .coefficient) (.value (.predecessor 1 52273 .coefficient)))

def event52275 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event52276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 52275

def event52277 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 52267

def event52278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 52276 .coefficient, .predecessor 1 52277 .coefficient])

def event52279 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event52280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 52279

def event52281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 52265

def event52282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 52281 .coefficient))

def event52283 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event52284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12770⟩⟩) 0 ⟨5542⟩ 52283

def event52285 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12770⟩⟩) (.authority (.programFamilyFact))

def exact52286RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩, (1)⟩]

theorem exact52286RawTermsValid :
    exact52286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52286 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12770⟩⟩) exact52286RawTerms (.finite 46) 52285 .exactZero (none)

def event52287 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10035⟩⟩) 0 ⟨5542⟩ 52283

def event52288 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10035⟩⟩) (.authority (.programFamilyFact))

def exact52289RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩], []⟩, (1)⟩]

theorem exact52289RawTermsValid :
    exact52289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52289 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10035⟩⟩) exact52289RawTerms (.finite 46) 52288 .exactZero (none)

def event52290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12771⟩⟩) 0 ⟨10035⟩ 52289

def event52291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12771⟩⟩) 1 ⟨12770⟩ 52286

def event52292 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12771⟩⟩) (.product (.predecessor 0 52290 .coefficient) (.predecessor 1 52291 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52293 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12771⟩⟩, .operator (⟨52289, 0⟩, ⟨52286, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩, (1)⟩)

def exact52294RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩, (1)⟩]

theorem exact52294RawTermsValid :
    exact52294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52294 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12771⟩⟩) exact52294RawTerms (.finite 2116) 52292 .exactZero (none)

def event52295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12772⟩⟩) 0 ⟨12771⟩ 52294

def event52296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12772⟩⟩) (.identity (.predecessor 0 52295 .coefficient))

def event52297 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12772⟩⟩) (.finite 2116)

def event52298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23291⟩⟩) 0 ⟨12772⟩ 52297

def event52299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23291⟩⟩) (.authority (.programFamilyFact))

def event52300 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23291⟩⟩) (.finite 3720)

def event52301 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event52302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23292⟩⟩) 0 ⟨6689⟩ 52301

def event52303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23292⟩⟩) 1 ⟨23291⟩ 52300

def event52304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23292⟩⟩) (.authority (.operator))

def exact52305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (1)⟩]

theorem exact52305RawTermsValid :
    exact52305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23292⟩⟩) exact52305RawTerms .large 52304 .exactZero (none)

def event52306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25532⟩⟩) 0 ⟨23292⟩ 52305

def event52307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25532⟩⟩) (.authority (.operator))

def exact52308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (1)⟩]

theorem exact52308RawTermsValid :
    exact52308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52308 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25532⟩⟩) exact52308RawTerms (.finite 8192) 52307 .exactZero (none)

def event52309 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event52310 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event52311 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12862⟩⟩) 0 ⟨12772⟩ 52297

def event52312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12862⟩⟩) 1 ⟨110⟩ 52310

def event52313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12862⟩⟩) (.sum [.predecessor 0 52311 .coefficient, .predecessor 1 52312 .coefficient])

def event52314 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12862⟩⟩) (.finite 2116)

def event52315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12863⟩⟩) 0 ⟨12862⟩ 52314

def event52316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12863⟩⟩) (.identity (.predecessor 0 52315 .coefficient))

def exact52317RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩, (1)⟩]

theorem exact52317RawTermsValid :
    exact52317RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52317 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12863⟩⟩) exact52317RawTerms (.finite 2116) 52316 .exactZero (none)

def event52318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact52319RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact52319RawTermsValid :
    exact52319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52319 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact52319RawTerms .large 52318 .exactZero (none)

def event52320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12864⟩⟩) 0 ⟨6544⟩ 52319

def event52321 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12864⟩⟩) 1 ⟨12863⟩ 52317

def event52322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12864⟩⟩) (.product (.predecessor 0 52320 .coefficient) (.predecessor 1 52321 .coefficient) (⟨false, false, none, none, none⟩))

def event52323 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12864⟩⟩, .operator (⟨52319, 0⟩, ⟨52317, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact52324RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact52324RawTermsValid :
    exact52324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52324 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12864⟩⟩) exact52324RawTerms .large 52322 .exactZero (none)

def event52325 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event52326 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event52327 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 52301

def event52328 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact52329RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact52329RawTermsValid :
    exact52329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52329 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact52329RawTerms .large 52328 .exactZero (none)

def event52330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6787⟩⟩) 0 ⟨6757⟩ 52329

def event52331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6787⟩⟩) (.identity (.predecessor 0 52330 .coefficient))

def exact52332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩]

theorem exact52332RawTermsValid :
    exact52332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52332 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6787⟩⟩) exact52332RawTerms .large 52331 .exactZero (none)

def event52333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7873⟩⟩) 0 ⟨6787⟩ 52332

def event52334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7873⟩⟩) (.authority (.operator))

def exact52335RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩]

theorem exact52335RawTermsValid :
    exact52335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52335 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7873⟩⟩) exact52335RawTerms (.finite 8192) 52334 .exactZero (none)

def event52336 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7874⟩⟩) 0 ⟨7873⟩ 52335

def event52337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7874⟩⟩) 1 ⟨2348⟩ 52326

def event52338 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7874⟩⟩) (.scale (.predecessor 0 52336 .coefficient) (.value (.predecessor 1 52337 .coefficient)))

def exact52339RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩]

theorem exact52339RawTermsValid :
    exact52339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52339 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7874⟩⟩) exact52339RawTerms (.finite 8192) 52338 .exactZero (none)

def event52340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6767⟩⟩) 0 ⟨6757⟩ 52329

def event52341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6767⟩⟩) (.identity (.predecessor 0 52340 .coefficient))

def exact52342RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩]

theorem exact52342RawTermsValid :
    exact52342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52342 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6767⟩⟩) exact52342RawTerms .large 52341 .exactZero (none)

def event52343 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7875⟩⟩) 0 ⟨6767⟩ 52342

def event52344 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7875⟩⟩) 1 ⟨7874⟩ 52339

def event52345 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7875⟩⟩) (.product (.predecessor 0 52343 .coefficient) (.predecessor 1 52344 .coefficient) (⟨false, false, none, none, none⟩))

def event52346 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7875⟩⟩, .operator (⟨52342, 0⟩, ⟨52339, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩)

def exact52347RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩]

theorem exact52347RawTermsValid :
    exact52347RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52347 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7875⟩⟩) exact52347RawTerms .large 52345 .exactZero (none)

def event52348 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12865⟩⟩) 0 ⟨7875⟩ 52347

def event52349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12865⟩⟩) 1 ⟨12864⟩ 52324

def event52350 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12865⟩⟩) (.sum [.predecessor 0 52348 .coefficient, .predecessor 1 52349 .coefficient])

def exact52351RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52351RawTermsValid :
    exact52351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52351 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12865⟩⟩) exact52351RawTerms .large 52350 .exactZero (none)

def event52352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25535⟩⟩) 0 ⟨12865⟩ 52351

def event52353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25535⟩⟩) 1 ⟨25532⟩ 52308

def event52354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25535⟩⟩) (.product (.predecessor 0 52352 .coefficient) (.predecessor 1 52353 .coefficient) (⟨false, false, none, none, none⟩))

def event52355 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25535⟩⟩, .operator (⟨52351, 0⟩, ⟨52308, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (1)⟩)

def event52356 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25535⟩⟩, .operator (⟨52351, 1⟩, ⟨52308, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (-1)⟩)

def event52357 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25535⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25532⟩⟩) ⟨23292⟩ 52305)

def event52358 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25535⟩⟩, .relation 52357 0, ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (-1)⟩)

def exact52359RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (-1)⟩]

theorem exact52359RawTermsValid :
    exact52359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52359 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25535⟩⟩) exact52359RawTerms .large 52354 .exactZero (none)

def event52360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16637⟩⟩) 0 ⟨12772⟩ 52297

def event52361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16637⟩⟩) (.authority (.programFamilyFact))

def exact52362RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16637⟩⟩], []⟩, (1)⟩]

theorem exact52362RawTermsValid :
    exact52362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52362 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16637⟩⟩) exact52362RawTerms (.finite 46) 52361 .exactZero (none)

def event52363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16639⟩⟩) 0 ⟨6544⟩ 52319

def event52364 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16639⟩⟩) 1 ⟨16637⟩ 52362

def event52365 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16639⟩⟩) (.product (.predecessor 0 52363 .coefficient) (.predecessor 1 52364 .coefficient) (⟨false, true, none, none, some 1⟩))

def event52366 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16639⟩⟩, .operator (⟨52319, 0⟩, ⟨52362, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact52367RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact52367RawTermsValid :
    exact52367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52367 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16639⟩⟩) exact52367RawTerms .large 52365 .exactZero (none)

def event52368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6704⟩⟩) 0 ⟨6689⟩ 52301

def event52369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6704⟩⟩) (.authority (.operator))

def exact52370RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩]

theorem exact52370RawTermsValid :
    exact52370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52370 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6704⟩⟩) exact52370RawTerms .large 52369 .exactZero (none)

def event52371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16640⟩⟩) 0 ⟨6704⟩ 52370

def event52372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16640⟩⟩) 1 ⟨16639⟩ 52367

def event52373 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16640⟩⟩) (.sum [.predecessor 0 52371 .coefficient, .predecessor 1 52372 .coefficient])

def exact52374RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52374RawTermsValid :
    exact52374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52374 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16640⟩⟩) exact52374RawTerms .large 52373 .exactZero (none)

def event52375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25536⟩⟩) 0 ⟨16640⟩ 52374

def event52376 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25536⟩⟩) 1 ⟨25535⟩ 52359

def event52377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25536⟩⟩) (.sum [.predecessor 0 52375 .coefficient, .predecessor 1 52376 .coefficient])

def exact52378RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52378RawTermsValid :
    exact52378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52378 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25536⟩⟩) exact52378RawTerms .large 52377 .exactZero (none)

def event52379 : Event := .preFoldPolynomial 52378 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact52380RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event52380 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25536⟩⟩) 52379 exact52380RawTerms .large 52377 .exactZero (none)

def event52381 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12772⟩⟩) ⟨⟨117⟩, ⟨23⟩, ⟨109⟩⟩ ⟨52215, 52381⟩

def event52382 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20039⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩) (1) 0 2 (.universal 52381 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩) (none) 52380)

def event52383 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20039⟩⟩, .relation 52382 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩)

def event52384 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20039⟩⟩, .relation 52382 1, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (-1)⟩)

def event52385 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20039⟩⟩, .relation 52382 2, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (1)⟩)

def event52386 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20039⟩⟩, .relation 52382 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact52387RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52387RawTermsValid :
    exact52387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52387 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20039⟩⟩) exact52387RawTerms .large 52211 (.finite 1811303510016) (some (52213))

def event52388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25534⟩⟩) 0 ⟨20039⟩ 52387

def event52389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25534⟩⟩) 1 ⟨25533⟩ 52201

def event52390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25534⟩⟩) (.sum [.predecessor 0 52388 .coefficient, .predecessor 1 52389 .coefficient])

def event52391 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25534⟩⟩, .operator (⟨52387, 2⟩, ⟨52201, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩, (-1)⟩)

def event52392 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25534⟩⟩, .operator (⟨52387, 1⟩, ⟨52201, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩, (1)⟩)

def event52393 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25534⟩⟩) (.sum [.result 52387 .summary, .result 52201 .summary])

def exact52394RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52394RawTermsValid :
    exact52394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52394 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25534⟩⟩) exact52394RawTerms .large 52390 (.finite 352146215809024) (some (52393))

def event52395 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29400⟩⟩) 0 ⟨25534⟩ 52394

def event52396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29400⟩⟩) 1 ⟨29398⟩ 52117

def event52397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29400⟩⟩) (.product (.predecessor 0 52395 .coefficient) (.predecessor 1 52396 .coefficient) (⟨false, false, none, none, none⟩))

def event52398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29400⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29398⟩⟩]⟩) [⟨.result 52117 .coefficient, false, none⟩])

def event52399 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29400⟩⟩) (.product (.result 52394 .summary) (.transfer 52398) (⟨false, false, none, none, none⟩))

def event52400 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29400⟩⟩, .operator (⟨52394, 0⟩, ⟨52117, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29398⟩⟩]⟩, (1)⟩)

def event52401 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29400⟩⟩, .operator (⟨52394, 1⟩, ⟨52117, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29398⟩⟩]⟩, (-1)⟩)

def event52402 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29400⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29398⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29398⟩⟩) ⟨24606⟩ 52114)

def event52403 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29400⟩⟩, .relation 52402 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨24606⟩⟩]⟩, (-1)⟩)

def exact52404RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29398⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16637⟩⟩], [⟨.program ⟨214⟩, ⟨24606⟩⟩]⟩, (-1)⟩]

theorem exact52404RawTermsValid :
    exact52404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52404 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29400⟩⟩) exact52404RawTerms .large 52397 (.finite 1292382246358571024384) (some (52399))

def event52405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22412⟩⟩) 0 ⟨16638⟩ 2424

def event52406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22412⟩⟩) (.authority (.relationPreimageSource ⟨59⟩))

def exact52407RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩, (1)⟩]

theorem exact52407RawTermsValid :
    exact52407RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52407 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22412⟩⟩) exact52407RawTerms (.finite 136065468) 52406 .exactZero (none)

def event52408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22414⟩⟩) 0 ⟨22412⟩ 52407

def event52409 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22414⟩⟩) 1 ⟨2348⟩ 4

def event52410 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22414⟩⟩) (.scale (.predecessor 0 52408 .coefficient) (.value (.predecessor 1 52409 .coefficient)))

def exact52411RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩, (1)⟩]

theorem exact52411RawTermsValid :
    exact52411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52411 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22414⟩⟩) exact52411RawTerms (.finite 136065468) 52410 .exactZero (none)

def event52412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22415⟩⟩) 0 ⟨5547⟩ 50762

def event52413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22415⟩⟩) 1 ⟨22414⟩ 52411

def event52414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22415⟩⟩) (.product (.predecessor 0 52412 .coefficient) (.predecessor 1 52413 .coefficient) (⟨false, false, none, none, none⟩))

def event52415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22415⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩) [⟨.result 52407 .coefficient, false, none⟩])

def event52416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22415⟩⟩) (.product (.result 50762 .summary) (.transfer 52415) (⟨false, false, none, none, none⟩))

def event52417 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22415⟩⟩, .operator (⟨50762, 0⟩, ⟨52411, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩, (1)⟩)

def event52418 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22413⟩⟩)

def event52419 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event52420 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event52421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event52422 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event52423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event52424 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event52425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event52426 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event52427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 52426

def event52428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 52424

def event52429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 52427 .coefficient) (.value (.predecessor 1 52428 .coefficient)))

def event52430 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event52431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 52430

def event52432 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 52422

def event52433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 52431 .coefficient, .predecessor 1 52432 .coefficient])

def event52434 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event52435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 52434

def event52436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 52420

def event52437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 52436 .coefficient))

def event52438 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event52439 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12770⟩⟩) 0 ⟨5542⟩ 52438

def event52440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12770⟩⟩) (.authority (.programFamilyFact))

def exact52441RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩, (1)⟩]

theorem exact52441RawTermsValid :
    exact52441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52441 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12770⟩⟩) exact52441RawTerms (.finite 46) 52440 .exactZero (none)

def event52442 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10035⟩⟩) 0 ⟨5542⟩ 52438

def event52443 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10035⟩⟩) (.authority (.programFamilyFact))

def exact52444RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩], []⟩, (1)⟩]

theorem exact52444RawTermsValid :
    exact52444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52444 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10035⟩⟩) exact52444RawTerms (.finite 46) 52443 .exactZero (none)

def event52445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12771⟩⟩) 0 ⟨10035⟩ 52444

def event52446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12771⟩⟩) 1 ⟨12770⟩ 52441

def event52447 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12771⟩⟩) (.product (.predecessor 0 52445 .coefficient) (.predecessor 1 52446 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52448 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12771⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩) [⟨.result 52444 .coefficient, true, some 1⟩, ⟨.result 52441 .coefficient, true, some 1⟩])

def event52449 : Event := .survivorFold (1) 52448

def exact52450RawTerms : List Term := []

theorem exact52450RawTermsValid :
    exact52450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52450 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12771⟩⟩) exact52450RawTerms (.finite 2116) 52447 (.finite 2116) (some (52448))

def event52451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12772⟩⟩) 0 ⟨12771⟩ 52450

def event52452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12772⟩⟩) (.identity (.predecessor 0 52451 .coefficient))

def event52453 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12772⟩⟩) (.finite 2116)

def event52454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16637⟩⟩) 0 ⟨12772⟩ 52453

def event52455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16637⟩⟩) (.authority (.programFamilyFact))

def exact52456RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16637⟩⟩], []⟩, (1)⟩]

theorem exact52456RawTermsValid :
    exact52456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52456 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16637⟩⟩) exact52456RawTerms (.finite 46) 52455 .exactZero (none)

def event52457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16638⟩⟩) 0 ⟨16637⟩ 52456

def event52458 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16638⟩⟩) (.identity (.predecessor 0 52457 .coefficient))

def event52459 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16638⟩⟩) (.finite 46)

def event52460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22412⟩⟩) 0 ⟨16638⟩ 52459

def event52461 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22412⟩⟩) (.authority (.relationPreimageSource ⟨59⟩))

def exact52462RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩, (1)⟩]

theorem exact52462RawTermsValid :
    exact52462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52462 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22412⟩⟩) exact52462RawTerms (.finite 136065468) 52461 .exactZero (none)

def event52463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact52464RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact52464RawTermsValid :
    exact52464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52464 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact52464RawTerms .large 52463 .exactZero (none)

def event52465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22413⟩⟩) 0 ⟨6⟩ 52464

def event52466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22413⟩⟩) 1 ⟨22412⟩ 52462

def event52467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22413⟩⟩) (.product (.predecessor 0 52465 .coefficient) (.predecessor 1 52466 .coefficient) (⟨false, false, none, none, none⟩))

def event52468 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22413⟩⟩, .operator (⟨52464, 0⟩, ⟨52462, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩, (1)⟩)

def exact52469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩, (1)⟩]

theorem exact52469RawTermsValid :
    exact52469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52469 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22413⟩⟩) exact52469RawTerms .large 52467 .exactZero (none)

def event52470 : Event := .preFoldPolynomial 52469 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩, (1)⟩] .exactZero none

def exact52471RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22412⟩⟩]⟩, (1)⟩]

def event52471 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22413⟩⟩) 52470 exact52471RawTerms .large 52467 .exactZero (none)

def event52472 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29403⟩⟩)

def event52473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event52474 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event52475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event52476 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event52477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event52478 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event52479 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def eventLeaf3264 : Array AnnotatedEvent := #[
  { event := event52224
    frameStart := 52215 },
  { event := event52225
    frameStart := 52215 },
  { event := event52226
    frameStart := 52215 },
  { event := event52227
    frameStart := 52215 },
  { event := event52228
    frameStart := 52215 },
  { event := event52229
    frameStart := 52215 },
  { event := event52230
    frameStart := 52215 },
  { event := event52231
    frameStart := 52215 },
  { event := event52232
    frameStart := 52215 },
  { event := event52233
    frameStart := 52215 },
  { event := event52234
    frameStart := 52215 },
  { event := event52235
    frameStart := 52215 },
  { event := event52236
    frameStart := 52215 },
  { event := event52237
    frameStart := 52215 },
  { event := event52238
    frameStart := 52215 },
  { event := event52239
    frameStart := 52215 }
]

def eventLeaf3265 : Array AnnotatedEvent := #[
  { event := event52240
    frameStart := 52215 },
  { event := event52241
    frameStart := 52215 },
  { event := event52242
    frameStart := 52215 },
  { event := event52243
    frameStart := 52215 },
  { event := event52244
    frameStart := 52215 },
  { event := event52245
    frameStart := 52215 },
  { event := event52246
    frameStart := 52215 },
  { event := event52247
    frameStart := 52215 },
  { event := event52248
    frameStart := 52215 },
  { event := event52249
    frameStart := 52215 },
  { event := event52250
    frameStart := 52215 },
  { event := event52251
    frameStart := 52215 },
  { event := event52252
    frameStart := 52215 },
  { event := event52253
    frameStart := 52215 },
  { event := event52254
    frameStart := 52215 },
  { event := event52255
    frameStart := 52215 }
]

def eventLeaf3266 : Array AnnotatedEvent := #[
  { event := event52256
    frameStart := 52215 },
  { event := event52257
    frameStart := 52215 },
  { event := event52258
    frameStart := 52215 },
  { event := event52259
    frameStart := 52215 },
  { event := event52260
    frameStart := 52215 },
  { event := event52261
    frameStart := 52215 },
  { event := event52262
    frameStart := 52215 },
  { event := event52263
    frameStart := 52263 },
  { event := event52264
    frameStart := 52263 },
  { event := event52265
    frameStart := 52263 },
  { event := event52266
    frameStart := 52263 },
  { event := event52267
    frameStart := 52263 },
  { event := event52268
    frameStart := 52263 },
  { event := event52269
    frameStart := 52263 },
  { event := event52270
    frameStart := 52263 },
  { event := event52271
    frameStart := 52263 }
]

def eventLeaf3267 : Array AnnotatedEvent := #[
  { event := event52272
    frameStart := 52263 },
  { event := event52273
    frameStart := 52263 },
  { event := event52274
    frameStart := 52263 },
  { event := event52275
    frameStart := 52263 },
  { event := event52276
    frameStart := 52263 },
  { event := event52277
    frameStart := 52263 },
  { event := event52278
    frameStart := 52263 },
  { event := event52279
    frameStart := 52263 },
  { event := event52280
    frameStart := 52263 },
  { event := event52281
    frameStart := 52263 },
  { event := event52282
    frameStart := 52263 },
  { event := event52283
    frameStart := 52263 },
  { event := event52284
    frameStart := 52263 },
  { event := event52285
    frameStart := 52263 },
  { event := event52286
    frameStart := 52263 },
  { event := event52287
    frameStart := 52263 }
]

def eventLeaf3268 : Array AnnotatedEvent := #[
  { event := event52288
    frameStart := 52263 },
  { event := event52289
    frameStart := 52263 },
  { event := event52290
    frameStart := 52263 },
  { event := event52291
    frameStart := 52263 },
  { event := event52292
    frameStart := 52263 },
  { event := event52293
    frameStart := 52263 },
  { event := event52294
    frameStart := 52263 },
  { event := event52295
    frameStart := 52263 },
  { event := event52296
    frameStart := 52263 },
  { event := event52297
    frameStart := 52263 },
  { event := event52298
    frameStart := 52263 },
  { event := event52299
    frameStart := 52263 },
  { event := event52300
    frameStart := 52263 },
  { event := event52301
    frameStart := 52263 },
  { event := event52302
    frameStart := 52263 },
  { event := event52303
    frameStart := 52263 }
]

def eventLeaf3269 : Array AnnotatedEvent := #[
  { event := event52304
    frameStart := 52263 },
  { event := event52305
    frameStart := 52263 },
  { event := event52306
    frameStart := 52263 },
  { event := event52307
    frameStart := 52263 },
  { event := event52308
    frameStart := 52263 },
  { event := event52309
    frameStart := 52263 },
  { event := event52310
    frameStart := 52263 },
  { event := event52311
    frameStart := 52263 },
  { event := event52312
    frameStart := 52263 },
  { event := event52313
    frameStart := 52263 },
  { event := event52314
    frameStart := 52263 },
  { event := event52315
    frameStart := 52263 },
  { event := event52316
    frameStart := 52263 },
  { event := event52317
    frameStart := 52263 },
  { event := event52318
    frameStart := 52263 },
  { event := event52319
    frameStart := 52263 }
]

def eventLeaf3270 : Array AnnotatedEvent := #[
  { event := event52320
    frameStart := 52263 },
  { event := event52321
    frameStart := 52263 },
  { event := event52322
    frameStart := 52263 },
  { event := event52323
    frameStart := 52263 },
  { event := event52324
    frameStart := 52263 },
  { event := event52325
    frameStart := 52263 },
  { event := event52326
    frameStart := 52263 },
  { event := event52327
    frameStart := 52263 },
  { event := event52328
    frameStart := 52263 },
  { event := event52329
    frameStart := 52263 },
  { event := event52330
    frameStart := 52263 },
  { event := event52331
    frameStart := 52263 },
  { event := event52332
    frameStart := 52263 },
  { event := event52333
    frameStart := 52263 },
  { event := event52334
    frameStart := 52263 },
  { event := event52335
    frameStart := 52263 }
]

def eventLeaf3271 : Array AnnotatedEvent := #[
  { event := event52336
    frameStart := 52263 },
  { event := event52337
    frameStart := 52263 },
  { event := event52338
    frameStart := 52263 },
  { event := event52339
    frameStart := 52263 },
  { event := event52340
    frameStart := 52263 },
  { event := event52341
    frameStart := 52263 },
  { event := event52342
    frameStart := 52263 },
  { event := event52343
    frameStart := 52263 },
  { event := event52344
    frameStart := 52263 },
  { event := event52345
    frameStart := 52263 },
  { event := event52346
    frameStart := 52263 },
  { event := event52347
    frameStart := 52263 },
  { event := event52348
    frameStart := 52263 },
  { event := event52349
    frameStart := 52263 },
  { event := event52350
    frameStart := 52263 },
  { event := event52351
    frameStart := 52263 }
]

def eventLeaf3272 : Array AnnotatedEvent := #[
  { event := event52352
    frameStart := 52263 },
  { event := event52353
    frameStart := 52263 },
  { event := event52354
    frameStart := 52263 },
  { event := event52355
    frameStart := 52263 },
  { event := event52356
    frameStart := 52263 },
  { event := event52357
    frameStart := 52263 },
  { event := event52358
    frameStart := 52263 },
  { event := event52359
    frameStart := 52263 },
  { event := event52360
    frameStart := 52263 },
  { event := event52361
    frameStart := 52263 },
  { event := event52362
    frameStart := 52263 },
  { event := event52363
    frameStart := 52263 },
  { event := event52364
    frameStart := 52263 },
  { event := event52365
    frameStart := 52263 },
  { event := event52366
    frameStart := 52263 },
  { event := event52367
    frameStart := 52263 }
]

def eventLeaf3273 : Array AnnotatedEvent := #[
  { event := event52368
    frameStart := 52263 },
  { event := event52369
    frameStart := 52263 },
  { event := event52370
    frameStart := 52263 },
  { event := event52371
    frameStart := 52263 },
  { event := event52372
    frameStart := 52263 },
  { event := event52373
    frameStart := 52263 },
  { event := event52374
    frameStart := 52263 },
  { event := event52375
    frameStart := 52263 },
  { event := event52376
    frameStart := 52263 },
  { event := event52377
    frameStart := 52263 },
  { event := event52378
    frameStart := 52263 },
  { event := event52379
    frameStart := 52263 },
  { event := event52380
    frameStart := 52263 },
  { event := event52381
    frameStart := 0 },
  { event := event52382
    frameStart := 0 },
  { event := event52383
    frameStart := 0 }
]

def eventLeaf3274 : Array AnnotatedEvent := #[
  { event := event52384
    frameStart := 0 },
  { event := event52385
    frameStart := 0 },
  { event := event52386
    frameStart := 0 },
  { event := event52387
    frameStart := 0 },
  { event := event52388
    frameStart := 0 },
  { event := event52389
    frameStart := 0 },
  { event := event52390
    frameStart := 0 },
  { event := event52391
    frameStart := 0 },
  { event := event52392
    frameStart := 0 },
  { event := event52393
    frameStart := 0 },
  { event := event52394
    frameStart := 0 },
  { event := event52395
    frameStart := 0 },
  { event := event52396
    frameStart := 0 },
  { event := event52397
    frameStart := 0 },
  { event := event52398
    frameStart := 0 },
  { event := event52399
    frameStart := 0 }
]

def eventLeaf3275 : Array AnnotatedEvent := #[
  { event := event52400
    frameStart := 0 },
  { event := event52401
    frameStart := 0 },
  { event := event52402
    frameStart := 0 },
  { event := event52403
    frameStart := 0 },
  { event := event52404
    frameStart := 0 },
  { event := event52405
    frameStart := 0 },
  { event := event52406
    frameStart := 0 },
  { event := event52407
    frameStart := 0 },
  { event := event52408
    frameStart := 0 },
  { event := event52409
    frameStart := 0 },
  { event := event52410
    frameStart := 0 },
  { event := event52411
    frameStart := 0 },
  { event := event52412
    frameStart := 0 },
  { event := event52413
    frameStart := 0 },
  { event := event52414
    frameStart := 0 },
  { event := event52415
    frameStart := 0 }
]

def eventLeaf3276 : Array AnnotatedEvent := #[
  { event := event52416
    frameStart := 0 },
  { event := event52417
    frameStart := 0 },
  { event := event52418
    frameStart := 52418 },
  { event := event52419
    frameStart := 52418 },
  { event := event52420
    frameStart := 52418 },
  { event := event52421
    frameStart := 52418 },
  { event := event52422
    frameStart := 52418 },
  { event := event52423
    frameStart := 52418 },
  { event := event52424
    frameStart := 52418 },
  { event := event52425
    frameStart := 52418 },
  { event := event52426
    frameStart := 52418 },
  { event := event52427
    frameStart := 52418 },
  { event := event52428
    frameStart := 52418 },
  { event := event52429
    frameStart := 52418 },
  { event := event52430
    frameStart := 52418 },
  { event := event52431
    frameStart := 52418 }
]

def eventLeaf3277 : Array AnnotatedEvent := #[
  { event := event52432
    frameStart := 52418 },
  { event := event52433
    frameStart := 52418 },
  { event := event52434
    frameStart := 52418 },
  { event := event52435
    frameStart := 52418 },
  { event := event52436
    frameStart := 52418 },
  { event := event52437
    frameStart := 52418 },
  { event := event52438
    frameStart := 52418 },
  { event := event52439
    frameStart := 52418 },
  { event := event52440
    frameStart := 52418 },
  { event := event52441
    frameStart := 52418 },
  { event := event52442
    frameStart := 52418 },
  { event := event52443
    frameStart := 52418 },
  { event := event52444
    frameStart := 52418 },
  { event := event52445
    frameStart := 52418 },
  { event := event52446
    frameStart := 52418 },
  { event := event52447
    frameStart := 52418 }
]

def eventLeaf3278 : Array AnnotatedEvent := #[
  { event := event52448
    frameStart := 52418 },
  { event := event52449
    frameStart := 52418 },
  { event := event52450
    frameStart := 52418 },
  { event := event52451
    frameStart := 52418 },
  { event := event52452
    frameStart := 52418 },
  { event := event52453
    frameStart := 52418 },
  { event := event52454
    frameStart := 52418 },
  { event := event52455
    frameStart := 52418 },
  { event := event52456
    frameStart := 52418 },
  { event := event52457
    frameStart := 52418 },
  { event := event52458
    frameStart := 52418 },
  { event := event52459
    frameStart := 52418 },
  { event := event52460
    frameStart := 52418 },
  { event := event52461
    frameStart := 52418 },
  { event := event52462
    frameStart := 52418 },
  { event := event52463
    frameStart := 52418 }
]

def eventLeaf3279 : Array AnnotatedEvent := #[
  { event := event52464
    frameStart := 52418 },
  { event := event52465
    frameStart := 52418 },
  { event := event52466
    frameStart := 52418 },
  { event := event52467
    frameStart := 52418 },
  { event := event52468
    frameStart := 52418 },
  { event := event52469
    frameStart := 52418 },
  { event := event52470
    frameStart := 52418 },
  { event := event52471
    frameStart := 52418 },
  { event := event52472
    frameStart := 52472 },
  { event := event52473
    frameStart := 52472 },
  { event := event52474
    frameStart := 52472 },
  { event := event52475
    frameStart := 52472 },
  { event := event52476
    frameStart := 52472 },
  { event := event52477
    frameStart := 52472 },
  { event := event52478
    frameStart := 52472 },
  { event := event52479
    frameStart := 52472 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events204
