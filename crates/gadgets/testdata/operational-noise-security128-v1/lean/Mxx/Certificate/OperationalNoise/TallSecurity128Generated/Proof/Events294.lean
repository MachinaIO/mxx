import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events294

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event75264 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17956⟩⟩)

def event75265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event75266 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event75267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event75268 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event75269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event75270 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event75271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event75272 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event75273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 75272

def event75274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 75270

def event75275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 75273 .coefficient) (.value (.predecessor 1 75274 .coefficient)))

def event75276 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event75277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 75276

def event75278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 75268

def event75279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 75277 .coefficient, .predecessor 1 75278 .coefficient])

def event75280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event75281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 75280

def event75282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 75266

def event75283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 75282 .coefficient))

def event75284 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event75285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15642⟩⟩) 0 ⟨10749⟩ 75284

def event75286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15642⟩⟩) (.authority (.programFamilyFact))

def exact75287RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15642⟩⟩], []⟩, (1)⟩]

theorem exact75287RawTermsValid :
    exact75287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75287 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15642⟩⟩) exact75287RawTerms (.finite 2) 75286 .exactZero (none)

def event75288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12486⟩⟩) 0 ⟨10749⟩ 75284

def event75289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12486⟩⟩) (.authority (.programFamilyFact))

def exact75290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12486⟩⟩], []⟩, (1)⟩]

theorem exact75290RawTermsValid :
    exact75290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12486⟩⟩) exact75290RawTerms (.finite 2) 75289 .exactZero (none)

def event75291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15643⟩⟩) 0 ⟨12486⟩ 75290

def event75292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15643⟩⟩) 1 ⟨15642⟩ 75287

def event75293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15643⟩⟩) (.product (.predecessor 0 75291 .coefficient) (.predecessor 1 75292 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event75294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15643⟩⟩, .operator (⟨75290, 0⟩, ⟨75287, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], []⟩, (1)⟩)

def exact75295RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], []⟩, (1)⟩]

theorem exact75295RawTermsValid :
    exact75295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75295 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15643⟩⟩) exact75295RawTerms (.finite 4) 75293 .exactZero (none)

def event75296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15644⟩⟩) 0 ⟨15643⟩ 75295

def event75297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15644⟩⟩) (.identity (.predecessor 0 75296 .coefficient))

def event75298 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15644⟩⟩) (.finite 4)

def event75299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15844⟩⟩) 0 ⟨15644⟩ 75298

def event75300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15844⟩⟩) (.authority (.programFamilyFact))

def exact75301RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], []⟩, (1)⟩]

theorem exact75301RawTermsValid :
    exact75301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75301 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15844⟩⟩) exact75301RawTerms (.finite 2) 75300 .exactZero (none)

def event75302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15845⟩⟩) 0 ⟨15844⟩ 75301

def event75303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15845⟩⟩) (.identity (.predecessor 0 75302 .coefficient))

def event75304 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15845⟩⟩) (.finite 2)

def event75305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17062⟩⟩) 0 ⟨15845⟩ 75304

def event75306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17062⟩⟩) (.authority (.programFamilyFact))

def event75307 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17062⟩⟩) (.finite 3720)

def event75308 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event75309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17063⟩⟩) 0 ⟨7177⟩ 75308

def event75310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17063⟩⟩) 1 ⟨17062⟩ 75307

def event75311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17063⟩⟩) (.authority (.operator))

def exact75312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (1)⟩]

theorem exact75312RawTermsValid :
    exact75312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75312 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17063⟩⟩) exact75312RawTerms .large 75311 .exactZero (none)

def event75313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17950⟩⟩) 0 ⟨17063⟩ 75312

def event75314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17950⟩⟩) (.authority (.operator))

def exact75315RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (1)⟩]

theorem exact75315RawTermsValid :
    exact75315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75315 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17950⟩⟩) exact75315RawTerms (.finite 8192) 75314 .exactZero (none)

def event75316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event75317 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event75318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17234⟩⟩) 0 ⟨15845⟩ 75304

def event75319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17234⟩⟩) 1 ⟨136⟩ 75317

def event75320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17234⟩⟩) (.sum [.predecessor 0 75318 .coefficient, .predecessor 1 75319 .coefficient])

def event75321 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17234⟩⟩) (.finite 2)

def event75322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17235⟩⟩) 0 ⟨17234⟩ 75321

def event75323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17235⟩⟩) (.identity (.predecessor 0 75322 .coefficient))

def exact75324RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], []⟩, (1)⟩]

theorem exact75324RawTermsValid :
    exact75324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75324 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17235⟩⟩) exact75324RawTerms (.finite 2) 75323 .exactZero (none)

def event75325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact75326RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact75326RawTermsValid :
    exact75326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact75326RawTerms .large 75325 .exactZero (none)

def event75327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17236⟩⟩) 0 ⟨6908⟩ 75326

def event75328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17236⟩⟩) 1 ⟨17235⟩ 75324

def event75329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17236⟩⟩) (.product (.predecessor 0 75327 .coefficient) (.predecessor 1 75328 .coefficient) (⟨false, false, none, none, none⟩))

def event75330 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17236⟩⟩, .operator (⟨75326, 0⟩, ⟨75324, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact75331RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact75331RawTermsValid :
    exact75331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75331 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17236⟩⟩) exact75331RawTerms .large 75329 .exactZero (none)

def event75332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 75308

def event75333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact75334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact75334RawTermsValid :
    exact75334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact75334RawTerms .large 75333 .exactZero (none)

def event75335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17237⟩⟩) 0 ⟨7179⟩ 75334

def event75336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17237⟩⟩) 1 ⟨17236⟩ 75331

def event75337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17237⟩⟩) (.sum [.predecessor 0 75335 .coefficient, .predecessor 1 75336 .coefficient])

def exact75338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75338RawTermsValid :
    exact75338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17237⟩⟩) exact75338RawTerms .large 75337 .exactZero (none)

def event75339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17951⟩⟩) 0 ⟨17237⟩ 75338

def event75340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17951⟩⟩) 1 ⟨17950⟩ 75315

def event75341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17951⟩⟩) (.product (.predecessor 0 75339 .coefficient) (.predecessor 1 75340 .coefficient) (⟨false, false, none, none, none⟩))

def event75342 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17951⟩⟩, .operator (⟨75338, 0⟩, ⟨75315, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (1)⟩)

def event75343 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17951⟩⟩, .operator (⟨75338, 1⟩, ⟨75315, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (-1)⟩)

def event75344 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17951⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17950⟩⟩) ⟨17063⟩ 75312)

def event75345 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17951⟩⟩, .relation 75344 0, ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (-1)⟩)

def exact75346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (-1)⟩]

theorem exact75346RawTermsValid :
    exact75346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17951⟩⟩) exact75346RawTerms .large 75341 .exactZero (none)

def event75347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16142⟩⟩) 0 ⟨15845⟩ 75304

def event75348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16142⟩⟩) (.authority (.programFamilyFact))

def exact75349RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact75349RawTermsValid :
    exact75349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16142⟩⟩) exact75349RawTerms (.finite 2) 75348 .exactZero (none)

def event75350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16145⟩⟩) 0 ⟨6908⟩ 75326

def event75351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16145⟩⟩) 1 ⟨16142⟩ 75349

def event75352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16145⟩⟩) (.product (.predecessor 0 75350 .coefficient) (.predecessor 1 75351 .coefficient) (⟨false, true, none, none, some 1⟩))

def event75353 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16145⟩⟩, .operator (⟨75326, 0⟩, ⟨75349, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact75354RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact75354RawTermsValid :
    exact75354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16145⟩⟩) exact75354RawTerms .large 75352 .exactZero (none)

def event75355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 75308

def event75356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact75357RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact75357RawTermsValid :
    exact75357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75357 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact75357RawTerms .large 75356 .exactZero (none)

def event75358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16146⟩⟩) 0 ⟨7197⟩ 75357

def event75359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16146⟩⟩) 1 ⟨16145⟩ 75354

def event75360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16146⟩⟩) (.sum [.predecessor 0 75358 .coefficient, .predecessor 1 75359 .coefficient])

def exact75361RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75361RawTermsValid :
    exact75361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75361 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16146⟩⟩) exact75361RawTerms .large 75360 .exactZero (none)

def event75362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17956⟩⟩) 0 ⟨16146⟩ 75361

def event75363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17956⟩⟩) 1 ⟨17951⟩ 75346

def event75364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17956⟩⟩) (.sum [.predecessor 0 75362 .coefficient, .predecessor 1 75363 .coefficient])

def exact75365RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75365RawTermsValid :
    exact75365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17956⟩⟩) exact75365RawTerms .large 75364 .exactZero (none)

def event75366 : Event := .preFoldPolynomial 75365 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact75367RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event75367 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17956⟩⟩) 75366 exact75367RawTerms .large 75364 .exactZero (none)

def event75368 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15845⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨75210, 75368⟩

def event75369 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16735⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16732⟩⟩]⟩) (1) 0 2 (.universal 75368 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16732⟩⟩]⟩) (none) 75367)

def event75370 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16735⟩⟩, .relation 75369 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event75371 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16735⟩⟩, .relation 75369 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (-1)⟩)

def event75372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16735⟩⟩, .relation 75369 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (1)⟩)

def event75373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16735⟩⟩, .relation 75369 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact75374RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75374RawTermsValid :
    exact75374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75374 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16735⟩⟩) exact75374RawTerms .large 75206 (.finite 202072841853861888) (some (75208))

def event75375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17953⟩⟩) 0 ⟨16735⟩ 75374

def event75376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17953⟩⟩) 1 ⟨17952⟩ 75196

def event75377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17953⟩⟩) (.sum [.predecessor 0 75375 .coefficient, .predecessor 1 75376 .coefficient])

def event75378 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17953⟩⟩, .operator (⟨75374, 0⟩, ⟨75196, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17950⟩⟩]⟩, (1)⟩)

def event75379 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17953⟩⟩, .operator (⟨75374, 2⟩, ⟨75196, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17063⟩⟩]⟩, (-1)⟩)

def event75380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17953⟩⟩) (.sum [.result 75374 .summary, .result 75196 .summary])

def exact75381RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75381RawTermsValid :
    exact75381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75381 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17953⟩⟩) exact75381RawTerms .large 75377 (.finite 32188807212483706889510625476608) (some (75380))

def event75382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17954⟩⟩) 0 ⟨17953⟩ 75381

def event75383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17954⟩⟩) 1 ⟨7172⟩ 15882

def event75384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17954⟩⟩) (.product (.predecessor 0 75382 .coefficient) (.predecessor 1 75383 .coefficient) (⟨false, false, none, none, none⟩))

def event75385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17954⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event75386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17954⟩⟩) (.product (.result 75381 .summary) (.transfer 75385) (⟨false, false, none, none, none⟩))

def event75387 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17954⟩⟩, .operator (⟨75381, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event75388 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17954⟩⟩, .operator (⟨75381, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event75389 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17954⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event75390 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17954⟩⟩, .relation 75389 0, ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact75391RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩]

theorem exact75391RawTermsValid :
    exact75391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75391 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17954⟩⟩) exact75391RawTerms .large 75384 (.finite 345624685687166110058245054666339432529920) (some (75386))

def event75392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10796⟩⟩) 0 ⟨6727⟩ 723

def event75393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10796⟩⟩) 1 ⟨10752⟩ 61278

def event75394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10796⟩⟩) (.tensor (.predecessor 0 75392 .coefficient) (.predecessor 1 75393 .coefficient) true false)

def event75395 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10796⟩⟩, .operator (⟨723, 0⟩, ⟨61278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact75396RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact75396RawTermsValid :
    exact75396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10796⟩⟩) exact75396RawTerms .large 75394 .exactZero (none)

def event75397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10774⟩⟩) 0 ⟨10751⟩ 61148

def event75398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10774⟩⟩) 1 ⟨7292⟩ 15896

def event75399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10774⟩⟩) (.product (.predecessor 0 75397 .coefficient) (.predecessor 1 75398 .coefficient) (⟨false, false, none, none, none⟩))

def event75400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10774⟩⟩, .operator (⟨61148, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact75401RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact75401RawTermsValid :
    exact75401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75401 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10774⟩⟩) exact75401RawTerms .large 75399 .exactZero (none)

def event75402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10797⟩⟩) 0 ⟨10774⟩ 75401

def event75403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10797⟩⟩) 1 ⟨10796⟩ 75396

def event75404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10797⟩⟩) (.sum [.predecessor 0 75402 .coefficient, .predecessor 1 75403 .coefficient])

def exact75405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact75405RawTermsValid :
    exact75405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10797⟩⟩) exact75405RawTerms .large 75404 .exactZero (none)

def event75406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10798⟩⟩) 0 ⟨10797⟩ 75405

def event75407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10798⟩⟩) 1 ⟨118⟩ 31516

def event75408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10798⟩⟩) (.sum [.predecessor 0 75406 .coefficient, .predecessor 1 75407 .coefficient])

def event75409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10798⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event75410 : Event := .survivorFold (1) 75409

def exact75411RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact75411RawTermsValid :
    exact75411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10798⟩⟩) exact75411RawTerms .large 75408 (.finite 26) (some (75409))

def event75412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10799⟩⟩) 0 ⟨10798⟩ 75411

def event75413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10799⟩⟩) 1 ⟨10798⟩ 75411

def event75414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10799⟩⟩) (.sum [.predecessor 0 75412 .coefficient, .predecessor 1 75413 .coefficient])

def event75415 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10799⟩⟩, .operator (⟨75411, 0⟩, ⟨75411, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event75416 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10799⟩⟩, .operator (⟨75411, 1⟩, ⟨75411, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event75417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10799⟩⟩) (.sum [.result 75411 .summary, .result 75411 .summary])

def exact75418RawTerms : List Term := []

theorem exact75418RawTermsValid :
    exact75418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75418 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10799⟩⟩) exact75418RawTerms .large 75414 (.finite 52) (some (75417))

def event75419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17955⟩⟩) 0 ⟨10799⟩ 75418

def event75420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17955⟩⟩) 1 ⟨17954⟩ 75391

def event75421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17955⟩⟩) (.sum [.predecessor 0 75419 .coefficient, .predecessor 1 75420 .coefficient])

def event75422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17955⟩⟩) (.sum [.result 75418 .summary, .result 75391 .summary])

def exact75423RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩]

theorem exact75423RawTermsValid :
    exact75423RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75423 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17955⟩⟩) exact75423RawTerms .large 75421 (.finite 345624685687166110058245054666339432529972) (some (75422))

def event75424 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20867⟩⟩) 0 ⟨17955⟩ 75423

def event75425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20867⟩⟩) 1 ⟨20866⟩ 75179

def event75426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20867⟩⟩) (.sum [.predecessor 0 75424 .coefficient, .predecessor 1 75425 .coefficient])

def event75427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20867⟩⟩) (.sum [.result 75423 .summary, .result 75179 .summary])

def exact75428RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩]

theorem exact75428RawTermsValid :
    exact75428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75428 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20867⟩⟩) exact75428RawTerms .large 75426 (.finite 691250426059631610003352154589745737891892) (some (75427))

def event75429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24087⟩⟩) 0 ⟨20867⟩ 75428

def event75430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24087⟩⟩) 1 ⟨24086⟩ 74967

def event75431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24087⟩⟩) (.sum [.predecessor 0 75429 .coefficient, .predecessor 1 75430 .coefficient])

def event75432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24087⟩⟩) (.sum [.result 75428 .summary, .result 74967 .summary])

def exact75433RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩]

theorem exact75433RawTermsValid :
    exact75433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24087⟩⟩) exact75433RawTerms .large 75431 (.finite 1036877221117396499835321299770218916085812) (some (75432))

def event75434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34107⟩⟩) 0 ⟨24087⟩ 75433

def event75435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34107⟩⟩) 1 ⟨34106⟩ 74755

def event75436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34107⟩⟩) (.sum [.predecessor 0 75434 .coefficient, .predecessor 1 75435 .coefficient])

def event75437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34107⟩⟩) (.sum [.result 75433 .summary, .result 74755 .summary])

def exact75438RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩]

theorem exact75438RawTermsValid :
    exact75438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34107⟩⟩) exact75438RawTerms .large 75436 (.finite 1382506125545760169441014535464825839943732) (some (75437))

def event75439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53167⟩⟩) 0 ⟨34107⟩ 75438

def event75440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53167⟩⟩) 1 ⟨53166⟩ 74543

def event75441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53167⟩⟩) (.sum [.predecessor 0 75439 .coefficient, .predecessor 1 75440 .coefficient])

def event75442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53167⟩⟩) (.sum [.result 75438 .summary, .result 74543 .summary])

def exact75443RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩]

theorem exact75443RawTermsValid :
    exact75443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75443 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53167⟩⟩) exact75443RawTerms .large 75441 (.finite 1728139248715321398594155952187700255129652) (some (75442))

def event75444 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56147⟩⟩) 0 ⟨53167⟩ 75443

def event75445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56147⟩⟩) 1 ⟨56146⟩ 74331

def event75446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56147⟩⟩) (.sum [.predecessor 0 75444 .coefficient, .predecessor 1 75445 .coefficient])

def event75447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56147⟩⟩) (.sum [.result 75443 .summary, .result 74331 .summary])

def exact75448RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩]

theorem exact75448RawTermsValid :
    exact75448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75448 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56147⟩⟩) exact75448RawTerms .large 75446 (.finite 2073774481255481407521021459424708415979572) (some (75447))

def event75449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59127⟩⟩) 0 ⟨56147⟩ 75448

def event75450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59127⟩⟩) 1 ⟨59126⟩ 74119

def event75451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59127⟩⟩) (.sum [.predecessor 0 75449 .coefficient, .predecessor 1 75450 .coefficient])

def event75452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59127⟩⟩) (.sum [.result 75448 .summary, .result 74119 .summary])

def exact75453RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩]

theorem exact75453RawTermsValid :
    exact75453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75453 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59127⟩⟩) exact75453RawTerms .large 75451 (.finite 2419413932536838975995335147689984068157492) (some (75452))

def event75454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62107⟩⟩) 0 ⟨59127⟩ 75453

def event75455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62107⟩⟩) 1 ⟨62106⟩ 73907

def event75456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62107⟩⟩) (.sum [.predecessor 0 75454 .coefficient, .predecessor 1 75455 .coefficient])

def event75457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62107⟩⟩) (.sum [.result 75453 .summary, .result 73907 .summary])

def exact75458RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩]

theorem exact75458RawTermsValid :
    exact75458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75458 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62107⟩⟩) exact75458RawTerms .large 75456 (.finite 2765055493188795324243372926469393465999412) (some (75457))

def event75459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65087⟩⟩) 0 ⟨62107⟩ 75458

def event75460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65087⟩⟩) 1 ⟨65086⟩ 73695

def event75461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65087⟩⟩) (.sum [.predecessor 0 75459 .coefficient, .predecessor 1 75460 .coefficient])

def event75462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65087⟩⟩) (.sum [.result 75458 .summary, .result 73695 .summary])

def exact75463RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩]

theorem exact75463RawTermsValid :
    exact75463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75463 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65087⟩⟩) exact75463RawTerms .large 75461 (.finite 3110701272581949232038858886277070355169332) (some (75462))

def event75464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70720⟩⟩) 0 ⟨65087⟩ 75463

def event75465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70720⟩⟩) 1 ⟨70719⟩ 73483

def event75466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70720⟩⟩) (.sum [.predecessor 0 75464 .coefficient, .predecessor 1 75465 .coefficient])

def event75467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70720⟩⟩) (.sum [.result 75463 .summary, .result 73483 .summary])

def exact75468RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩]

theorem exact75468RawTermsValid :
    exact75468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75468 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70720⟩⟩) exact75468RawTerms .large 75466 (.finite 3456353380086899479155517117627148481331252) (some (75467))

def event75469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70721⟩⟩) 0 ⟨70720⟩ 75468

def event75470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70721⟩⟩) 1 ⟨28462⟩ 73271

def event75471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70721⟩⟩) (.sum [.predecessor 0 75469 .coefficient, .predecessor 1 75470 .coefficient])

def event75472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70721⟩⟩) (.sum [.result 75468 .summary, .result 73271 .summary])

def exact75473RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩]

theorem exact75473RawTermsValid :
    exact75473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75473 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70721⟩⟩) exact75473RawTerms .large 75471 (.finite 3802007596962448506045899439491360353157172) (some (75472))

def event75474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70722⟩⟩) 0 ⟨70721⟩ 75473

def event75475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70722⟩⟩) 1 ⟨31142⟩ 73059

def event75476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70722⟩⟩) (.sum [.predecessor 0 75474 .coefficient, .predecessor 1 75475 .coefficient])

def event75477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70722⟩⟩) (.sum [.result 75473 .summary, .result 73059 .summary])

def exact75478RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩]

theorem exact75478RawTermsValid :
    exact75478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75478 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70722⟩⟩) exact75478RawTerms .large 75476 (.finite 4147668141949793872257454032897973461975092) (some (75477))

def event75479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70723⟩⟩) 0 ⟨70722⟩ 75478

def event75480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70723⟩⟩) 1 ⟨36802⟩ 72847

def event75481 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70723⟩⟩) (.sum [.predecessor 0 75479 .coefficient, .predecessor 1 75480 .coefficient])

def event75482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70723⟩⟩) (.sum [.result 75478 .summary, .result 72847 .summary])

def exact75483RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩]

theorem exact75483RawTermsValid :
    exact75483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75483 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70723⟩⟩) exact75483RawTerms .large 75481 (.finite 4493332905678336798016456807332854062121012) (some (75482))

def event75484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70724⟩⟩) 0 ⟨70723⟩ 75483

def event75485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70724⟩⟩) 1 ⟨39482⟩ 72635

def event75486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70724⟩⟩) (.sum [.predecessor 0 75484 .coefficient, .predecessor 1 75485 .coefficient])

def event75487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70724⟩⟩) (.sum [.result 75483 .summary, .result 72635 .summary])

def exact75488RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩]

theorem exact75488RawTermsValid :
    exact75488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75488 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70724⟩⟩) exact75488RawTerms .large 75486 (.finite 4838999778777478503549183672281868407930932) (some (75487))

def event75489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70725⟩⟩) 0 ⟨70724⟩ 75488

def event75490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70725⟩⟩) 1 ⟨42162⟩ 72423

def event75491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70725⟩⟩) (.sum [.predecessor 0 75489 .coefficient, .predecessor 1 75490 .coefficient])

def event75492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70725⟩⟩) (.sum [.result 75488 .summary, .result 72423 .summary])

def exact75493RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩]

theorem exact75493RawTermsValid :
    exact75493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70725⟩⟩) exact75493RawTerms .large 75491 (.finite 5184670870617817768629358718259150245068852) (some (75492))

def event75494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70726⟩⟩) 0 ⟨70725⟩ 75493

def event75495 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70726⟩⟩) 1 ⟨44842⟩ 72211

def event75496 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70726⟩⟩) (.sum [.predecessor 0 75494 .coefficient, .predecessor 1 75495 .coefficient])

def event75497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70726⟩⟩) (.sum [.result 75493 .summary, .result 72211 .summary])

def exact75498RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩]

theorem exact75498RawTermsValid :
    exact75498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75498 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70726⟩⟩) exact75498RawTerms .large 75496 (.finite 5530348290569953373030706035778833319198772) (some (75497))

def event75499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70727⟩⟩) 0 ⟨70726⟩ 75498

def event75500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70727⟩⟩) 1 ⟨47522⟩ 71999

def event75501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70727⟩⟩) (.sum [.predecessor 0 75499 .coefficient, .predecessor 1 75500 .coefficient])

def event75502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70727⟩⟩) (.sum [.result 75498 .summary, .result 71999 .summary])

def exact75503RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩]

theorem exact75503RawTermsValid :
    exact75503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70727⟩⟩) exact75503RawTerms .large 75501 (.finite 5876032038633885316753225624840917630320692) (some (75502))

def event75504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70728⟩⟩) 0 ⟨70727⟩ 75503

def event75505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70728⟩⟩) 1 ⟨50202⟩ 71787

def event75506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70728⟩⟩) (.sum [.predecessor 0 75504 .coefficient, .predecessor 1 75505 .coefficient])

def event75507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70728⟩⟩) (.sum [.result 75503 .summary, .result 71787 .summary])

def exact75508RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩]

theorem exact75508RawTermsValid :
    exact75508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70728⟩⟩) exact75508RawTerms .large 75506 (.finite 6221717896068416040249469304417135687106612) (some (75507))

def event75509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71475⟩⟩) 0 ⟨70728⟩ 75508

def event75510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71475⟩⟩) 1 ⟨71473⟩ 71575

def event75511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71475⟩⟩) (.sum [.predecessor 0 75509 .coefficient, .predecessor 1 75510 .coefficient])

def event75512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71475⟩⟩) (.sum [.result 75508 .summary, .result 71575 .summary])

def exact75513RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩]

theorem exact75513RawTermsValid :
    exact75513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75513 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71475⟩⟩) exact75513RawTerms .large 75511 (.finite 66805187227601152574551644069558752530002096506798132) (some (75512))

def event75514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26⟩⟩) (.authority (.operator))

def exact75515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨26⟩⟩]⟩, (1)⟩]

theorem exact75515RawTermsValid :
    exact75515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75515 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26⟩⟩) exact75515RawTerms (.finite 26) 75514 .exactZero (none)

def event75516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7404⟩⟩) 0 ⟨2377⟩ 27

def event75517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7404⟩⟩) 1 ⟨7240⟩ 16107

def event75518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7404⟩⟩) (.product (.predecessor 0 75516 .coefficient) (.predecessor 1 75517 .coefficient) (⟨false, false, none, none, none⟩))

def event75519 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7404⟩⟩, .operator (⟨27, 0⟩, ⟨16107, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩]⟩, (1)⟩)

def eventLeaf4704 : Array AnnotatedEvent := #[
  { event := event75264
    frameStart := 75264 },
  { event := event75265
    frameStart := 75264 },
  { event := event75266
    frameStart := 75264 },
  { event := event75267
    frameStart := 75264 },
  { event := event75268
    frameStart := 75264 },
  { event := event75269
    frameStart := 75264 },
  { event := event75270
    frameStart := 75264 },
  { event := event75271
    frameStart := 75264 },
  { event := event75272
    frameStart := 75264 },
  { event := event75273
    frameStart := 75264 },
  { event := event75274
    frameStart := 75264 },
  { event := event75275
    frameStart := 75264 },
  { event := event75276
    frameStart := 75264 },
  { event := event75277
    frameStart := 75264 },
  { event := event75278
    frameStart := 75264 },
  { event := event75279
    frameStart := 75264 }
]

def eventLeaf4705 : Array AnnotatedEvent := #[
  { event := event75280
    frameStart := 75264 },
  { event := event75281
    frameStart := 75264 },
  { event := event75282
    frameStart := 75264 },
  { event := event75283
    frameStart := 75264 },
  { event := event75284
    frameStart := 75264 },
  { event := event75285
    frameStart := 75264 },
  { event := event75286
    frameStart := 75264 },
  { event := event75287
    frameStart := 75264 },
  { event := event75288
    frameStart := 75264 },
  { event := event75289
    frameStart := 75264 },
  { event := event75290
    frameStart := 75264 },
  { event := event75291
    frameStart := 75264 },
  { event := event75292
    frameStart := 75264 },
  { event := event75293
    frameStart := 75264 },
  { event := event75294
    frameStart := 75264 },
  { event := event75295
    frameStart := 75264 }
]

def eventLeaf4706 : Array AnnotatedEvent := #[
  { event := event75296
    frameStart := 75264 },
  { event := event75297
    frameStart := 75264 },
  { event := event75298
    frameStart := 75264 },
  { event := event75299
    frameStart := 75264 },
  { event := event75300
    frameStart := 75264 },
  { event := event75301
    frameStart := 75264 },
  { event := event75302
    frameStart := 75264 },
  { event := event75303
    frameStart := 75264 },
  { event := event75304
    frameStart := 75264 },
  { event := event75305
    frameStart := 75264 },
  { event := event75306
    frameStart := 75264 },
  { event := event75307
    frameStart := 75264 },
  { event := event75308
    frameStart := 75264 },
  { event := event75309
    frameStart := 75264 },
  { event := event75310
    frameStart := 75264 },
  { event := event75311
    frameStart := 75264 }
]

def eventLeaf4707 : Array AnnotatedEvent := #[
  { event := event75312
    frameStart := 75264 },
  { event := event75313
    frameStart := 75264 },
  { event := event75314
    frameStart := 75264 },
  { event := event75315
    frameStart := 75264 },
  { event := event75316
    frameStart := 75264 },
  { event := event75317
    frameStart := 75264 },
  { event := event75318
    frameStart := 75264 },
  { event := event75319
    frameStart := 75264 },
  { event := event75320
    frameStart := 75264 },
  { event := event75321
    frameStart := 75264 },
  { event := event75322
    frameStart := 75264 },
  { event := event75323
    frameStart := 75264 },
  { event := event75324
    frameStart := 75264 },
  { event := event75325
    frameStart := 75264 },
  { event := event75326
    frameStart := 75264 },
  { event := event75327
    frameStart := 75264 }
]

def eventLeaf4708 : Array AnnotatedEvent := #[
  { event := event75328
    frameStart := 75264 },
  { event := event75329
    frameStart := 75264 },
  { event := event75330
    frameStart := 75264 },
  { event := event75331
    frameStart := 75264 },
  { event := event75332
    frameStart := 75264 },
  { event := event75333
    frameStart := 75264 },
  { event := event75334
    frameStart := 75264 },
  { event := event75335
    frameStart := 75264 },
  { event := event75336
    frameStart := 75264 },
  { event := event75337
    frameStart := 75264 },
  { event := event75338
    frameStart := 75264 },
  { event := event75339
    frameStart := 75264 },
  { event := event75340
    frameStart := 75264 },
  { event := event75341
    frameStart := 75264 },
  { event := event75342
    frameStart := 75264 },
  { event := event75343
    frameStart := 75264 }
]

def eventLeaf4709 : Array AnnotatedEvent := #[
  { event := event75344
    frameStart := 75264 },
  { event := event75345
    frameStart := 75264 },
  { event := event75346
    frameStart := 75264 },
  { event := event75347
    frameStart := 75264 },
  { event := event75348
    frameStart := 75264 },
  { event := event75349
    frameStart := 75264 },
  { event := event75350
    frameStart := 75264 },
  { event := event75351
    frameStart := 75264 },
  { event := event75352
    frameStart := 75264 },
  { event := event75353
    frameStart := 75264 },
  { event := event75354
    frameStart := 75264 },
  { event := event75355
    frameStart := 75264 },
  { event := event75356
    frameStart := 75264 },
  { event := event75357
    frameStart := 75264 },
  { event := event75358
    frameStart := 75264 },
  { event := event75359
    frameStart := 75264 }
]

def eventLeaf4710 : Array AnnotatedEvent := #[
  { event := event75360
    frameStart := 75264 },
  { event := event75361
    frameStart := 75264 },
  { event := event75362
    frameStart := 75264 },
  { event := event75363
    frameStart := 75264 },
  { event := event75364
    frameStart := 75264 },
  { event := event75365
    frameStart := 75264 },
  { event := event75366
    frameStart := 75264 },
  { event := event75367
    frameStart := 75264 },
  { event := event75368
    frameStart := 0 },
  { event := event75369
    frameStart := 0 },
  { event := event75370
    frameStart := 0 },
  { event := event75371
    frameStart := 0 },
  { event := event75372
    frameStart := 0 },
  { event := event75373
    frameStart := 0 },
  { event := event75374
    frameStart := 0 },
  { event := event75375
    frameStart := 0 }
]

def eventLeaf4711 : Array AnnotatedEvent := #[
  { event := event75376
    frameStart := 0 },
  { event := event75377
    frameStart := 0 },
  { event := event75378
    frameStart := 0 },
  { event := event75379
    frameStart := 0 },
  { event := event75380
    frameStart := 0 },
  { event := event75381
    frameStart := 0 },
  { event := event75382
    frameStart := 0 },
  { event := event75383
    frameStart := 0 },
  { event := event75384
    frameStart := 0 },
  { event := event75385
    frameStart := 0 },
  { event := event75386
    frameStart := 0 },
  { event := event75387
    frameStart := 0 },
  { event := event75388
    frameStart := 0 },
  { event := event75389
    frameStart := 0 },
  { event := event75390
    frameStart := 0 },
  { event := event75391
    frameStart := 0 }
]

def eventLeaf4712 : Array AnnotatedEvent := #[
  { event := event75392
    frameStart := 0 },
  { event := event75393
    frameStart := 0 },
  { event := event75394
    frameStart := 0 },
  { event := event75395
    frameStart := 0 },
  { event := event75396
    frameStart := 0 },
  { event := event75397
    frameStart := 0 },
  { event := event75398
    frameStart := 0 },
  { event := event75399
    frameStart := 0 },
  { event := event75400
    frameStart := 0 },
  { event := event75401
    frameStart := 0 },
  { event := event75402
    frameStart := 0 },
  { event := event75403
    frameStart := 0 },
  { event := event75404
    frameStart := 0 },
  { event := event75405
    frameStart := 0 },
  { event := event75406
    frameStart := 0 },
  { event := event75407
    frameStart := 0 }
]

def eventLeaf4713 : Array AnnotatedEvent := #[
  { event := event75408
    frameStart := 0 },
  { event := event75409
    frameStart := 0 },
  { event := event75410
    frameStart := 0 },
  { event := event75411
    frameStart := 0 },
  { event := event75412
    frameStart := 0 },
  { event := event75413
    frameStart := 0 },
  { event := event75414
    frameStart := 0 },
  { event := event75415
    frameStart := 0 },
  { event := event75416
    frameStart := 0 },
  { event := event75417
    frameStart := 0 },
  { event := event75418
    frameStart := 0 },
  { event := event75419
    frameStart := 0 },
  { event := event75420
    frameStart := 0 },
  { event := event75421
    frameStart := 0 },
  { event := event75422
    frameStart := 0 },
  { event := event75423
    frameStart := 0 }
]

def eventLeaf4714 : Array AnnotatedEvent := #[
  { event := event75424
    frameStart := 0 },
  { event := event75425
    frameStart := 0 },
  { event := event75426
    frameStart := 0 },
  { event := event75427
    frameStart := 0 },
  { event := event75428
    frameStart := 0 },
  { event := event75429
    frameStart := 0 },
  { event := event75430
    frameStart := 0 },
  { event := event75431
    frameStart := 0 },
  { event := event75432
    frameStart := 0 },
  { event := event75433
    frameStart := 0 },
  { event := event75434
    frameStart := 0 },
  { event := event75435
    frameStart := 0 },
  { event := event75436
    frameStart := 0 },
  { event := event75437
    frameStart := 0 },
  { event := event75438
    frameStart := 0 },
  { event := event75439
    frameStart := 0 }
]

def eventLeaf4715 : Array AnnotatedEvent := #[
  { event := event75440
    frameStart := 0 },
  { event := event75441
    frameStart := 0 },
  { event := event75442
    frameStart := 0 },
  { event := event75443
    frameStart := 0 },
  { event := event75444
    frameStart := 0 },
  { event := event75445
    frameStart := 0 },
  { event := event75446
    frameStart := 0 },
  { event := event75447
    frameStart := 0 },
  { event := event75448
    frameStart := 0 },
  { event := event75449
    frameStart := 0 },
  { event := event75450
    frameStart := 0 },
  { event := event75451
    frameStart := 0 },
  { event := event75452
    frameStart := 0 },
  { event := event75453
    frameStart := 0 },
  { event := event75454
    frameStart := 0 },
  { event := event75455
    frameStart := 0 }
]

def eventLeaf4716 : Array AnnotatedEvent := #[
  { event := event75456
    frameStart := 0 },
  { event := event75457
    frameStart := 0 },
  { event := event75458
    frameStart := 0 },
  { event := event75459
    frameStart := 0 },
  { event := event75460
    frameStart := 0 },
  { event := event75461
    frameStart := 0 },
  { event := event75462
    frameStart := 0 },
  { event := event75463
    frameStart := 0 },
  { event := event75464
    frameStart := 0 },
  { event := event75465
    frameStart := 0 },
  { event := event75466
    frameStart := 0 },
  { event := event75467
    frameStart := 0 },
  { event := event75468
    frameStart := 0 },
  { event := event75469
    frameStart := 0 },
  { event := event75470
    frameStart := 0 },
  { event := event75471
    frameStart := 0 }
]

def eventLeaf4717 : Array AnnotatedEvent := #[
  { event := event75472
    frameStart := 0 },
  { event := event75473
    frameStart := 0 },
  { event := event75474
    frameStart := 0 },
  { event := event75475
    frameStart := 0 },
  { event := event75476
    frameStart := 0 },
  { event := event75477
    frameStart := 0 },
  { event := event75478
    frameStart := 0 },
  { event := event75479
    frameStart := 0 },
  { event := event75480
    frameStart := 0 },
  { event := event75481
    frameStart := 0 },
  { event := event75482
    frameStart := 0 },
  { event := event75483
    frameStart := 0 },
  { event := event75484
    frameStart := 0 },
  { event := event75485
    frameStart := 0 },
  { event := event75486
    frameStart := 0 },
  { event := event75487
    frameStart := 0 }
]

def eventLeaf4718 : Array AnnotatedEvent := #[
  { event := event75488
    frameStart := 0 },
  { event := event75489
    frameStart := 0 },
  { event := event75490
    frameStart := 0 },
  { event := event75491
    frameStart := 0 },
  { event := event75492
    frameStart := 0 },
  { event := event75493
    frameStart := 0 },
  { event := event75494
    frameStart := 0 },
  { event := event75495
    frameStart := 0 },
  { event := event75496
    frameStart := 0 },
  { event := event75497
    frameStart := 0 },
  { event := event75498
    frameStart := 0 },
  { event := event75499
    frameStart := 0 },
  { event := event75500
    frameStart := 0 },
  { event := event75501
    frameStart := 0 },
  { event := event75502
    frameStart := 0 },
  { event := event75503
    frameStart := 0 }
]

def eventLeaf4719 : Array AnnotatedEvent := #[
  { event := event75504
    frameStart := 0 },
  { event := event75505
    frameStart := 0 },
  { event := event75506
    frameStart := 0 },
  { event := event75507
    frameStart := 0 },
  { event := event75508
    frameStart := 0 },
  { event := event75509
    frameStart := 0 },
  { event := event75510
    frameStart := 0 },
  { event := event75511
    frameStart := 0 },
  { event := event75512
    frameStart := 0 },
  { event := event75513
    frameStart := 0 },
  { event := event75514
    frameStart := 0 },
  { event := event75515
    frameStart := 0 },
  { event := event75516
    frameStart := 0 },
  { event := event75517
    frameStart := 0 },
  { event := event75518
    frameStart := 0 },
  { event := event75519
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events294
