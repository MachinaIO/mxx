import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1130

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event289280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17596⟩⟩) (.sum [.predecessor 0 289278 .coefficient, .predecessor 1 289279 .coefficient])

def event289281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17596⟩⟩, .operator (⟨289277, 0⟩, ⟨289099, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17593⟩⟩]⟩, (1)⟩)

def event289282 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17596⟩⟩, .operator (⟨289277, 2⟩, ⟨289099, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], [⟨.program ⟨257⟩, ⟨16947⟩⟩]⟩, (-1)⟩)

def event289283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17596⟩⟩) (.sum [.result 289277 .summary, .result 289099 .summary])

def exact289284RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289284RawTermsValid :
    exact289284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17596⟩⟩) exact289284RawTerms .large 289280 (.finite 32188807212483706889510625476608) (some (289283))

def event289285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20470⟩⟩) 0 ⟨17596⟩ 289284

def event289286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20470⟩⟩) 1 ⟨20469⟩ 288804

def event289287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20470⟩⟩) (.sum [.predecessor 0 289285 .coefficient, .predecessor 1 289286 .coefficient])

def event289288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20470⟩⟩) (.sum [.result 289284 .summary, .result 288804 .summary])

def exact289289RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289289RawTermsValid :
    exact289289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20470⟩⟩) exact289289RawTerms .large 289287 (.finite 64377712650190257467641695830016) (some (289288))

def event289290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23690⟩⟩) 0 ⟨20470⟩ 289289

def event289291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23690⟩⟩) 1 ⟨23689⟩ 288324

def event289292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23690⟩⟩) (.sum [.predecessor 0 289290 .coefficient, .predecessor 1 289291 .coefficient])

def event289293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23690⟩⟩) (.sum [.result 289289 .summary, .result 288324 .summary])

def exact289294RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289294RawTermsValid :
    exact289294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289294 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23690⟩⟩) exact289294RawTerms .large 289292 (.finite 96566716313119651734393211060224) (some (289293))

def event289295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33710⟩⟩) 0 ⟨23690⟩ 289294

def event289296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33710⟩⟩) 1 ⟨33709⟩ 287844

def event289297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33710⟩⟩) (.sum [.predecessor 0 289295 .coefficient, .predecessor 1 289296 .coefficient])

def event289298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33710⟩⟩) (.sum [.result 289294 .summary, .result 287844 .summary])

def exact289299RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289299RawTermsValid :
    exact289299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33710⟩⟩) exact289299RawTerms .large 289297 (.finite 128755916426494733378385616044032) (some (289298))

def event289300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52770⟩⟩) 0 ⟨33710⟩ 289299

def event289301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52770⟩⟩) 1 ⟨52769⟩ 287364

def event289302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52770⟩⟩) (.sum [.predecessor 0 289300 .coefficient, .predecessor 1 289301 .coefficient])

def event289303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52770⟩⟩) (.sum [.result 289299 .summary, .result 287364 .summary])

def exact289304RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289304RawTermsValid :
    exact289304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52770⟩⟩) exact289304RawTerms .large 289302 (.finite 160945509440761189776859800535040) (some (289303))

def event289305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55750⟩⟩) 0 ⟨52770⟩ 289304

def event289306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55750⟩⟩) 1 ⟨55749⟩ 286884

def event289307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55750⟩⟩) (.sum [.predecessor 0 289305 .coefficient, .predecessor 1 289306 .coefficient])

def event289308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55750⟩⟩) (.sum [.result 289304 .summary, .result 286884 .summary])

def exact289309RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289309RawTermsValid :
    exact289309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55750⟩⟩) exact289309RawTerms .large 289307 (.finite 193135298905473333552574874779648) (some (289308))

def event289310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58730⟩⟩) 0 ⟨55750⟩ 289309

def event289311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58730⟩⟩) 1 ⟨58729⟩ 286404

def event289312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58730⟩⟩) (.sum [.predecessor 0 289310 .coefficient, .predecessor 1 289311 .coefficient])

def event289313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58730⟩⟩) (.sum [.result 289309 .summary, .result 286404 .summary])

def exact289314RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289314RawTermsValid :
    exact289314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58730⟩⟩) exact289314RawTerms .large 289312 (.finite 225325481271076852082771728531456) (some (289313))

def event289315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61710⟩⟩) 0 ⟨58730⟩ 289314

def event289316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61710⟩⟩) 1 ⟨61709⟩ 285924

def event289317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61710⟩⟩) (.sum [.predecessor 0 289315 .coefficient, .predecessor 1 289316 .coefficient])

def event289318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61710⟩⟩) (.sum [.result 289314 .summary, .result 285924 .summary])

def exact289319RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289319RawTermsValid :
    exact289319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289319 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61710⟩⟩) exact289319RawTerms .large 289317 (.finite 257515860087126057990209472036864) (some (289318))

def event289320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64690⟩⟩) 0 ⟨61710⟩ 289319

def event289321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64690⟩⟩) 1 ⟨64689⟩ 285444

def event289322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64690⟩⟩) (.sum [.predecessor 0 289320 .coefficient, .predecessor 1 289321 .coefficient])

def event289323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64690⟩⟩) (.sum [.result 289319 .summary, .result 285444 .summary])

def exact289324RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289324RawTermsValid :
    exact289324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289324 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64690⟩⟩) exact289324RawTerms .large 289322 (.finite 289706631804066638652128995049472) (some (289323))

def event289325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69707⟩⟩) 0 ⟨64690⟩ 289324

def event289326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69707⟩⟩) 1 ⟨69706⟩ 284964

def event289327 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69707⟩⟩) (.sum [.predecessor 0 289325 .coefficient, .predecessor 1 289326 .coefficient])

def event289328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69707⟩⟩) (.sum [.result 289324 .summary, .result 284964 .summary])

def exact289329RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289329RawTermsValid :
    exact289329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69707⟩⟩) exact289329RawTerms .large 289327 (.finite 321897992872344281445771187322880) (some (289328))

def event289330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69708⟩⟩) 0 ⟨69707⟩ 289329

def event289331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69708⟩⟩) 1 ⟨28142⟩ 284484

def event289332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69708⟩⟩) (.sum [.predecessor 0 289330 .coefficient, .predecessor 1 289331 .coefficient])

def event289333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69708⟩⟩) (.sum [.result 289329 .summary, .result 284484 .summary])

def exact289334RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289334RawTermsValid :
    exact289334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69708⟩⟩) exact289334RawTerms .large 289332 (.finite 354089550391067611616654269349888) (some (289333))

def event289335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69709⟩⟩) 0 ⟨69708⟩ 289334

def event289336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69709⟩⟩) 1 ⟨30822⟩ 284004

def event289337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69709⟩⟩) (.sum [.predecessor 0 289335 .coefficient, .predecessor 1 289336 .coefficient])

def event289338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69709⟩⟩) (.sum [.result 289334 .summary, .result 284004 .summary])

def exact289339RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289339RawTermsValid :
    exact289339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289339 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69709⟩⟩) exact289339RawTerms .large 289337 (.finite 386281697261128003919260020637696) (some (289338))

def event289340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69710⟩⟩) 0 ⟨69709⟩ 289339

def event289341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69710⟩⟩) 1 ⟨36482⟩ 283524

def event289342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69710⟩⟩) (.sum [.predecessor 0 289340 .coefficient, .predecessor 1 289341 .coefficient])

def event289343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69710⟩⟩) (.sum [.result 289339 .summary, .result 283524 .summary])

def exact289344RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289344RawTermsValid :
    exact289344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289344 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69710⟩⟩) exact289344RawTerms .large 289342 (.finite 418474237032079770976347551432704) (some (289343))

def event289345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69711⟩⟩) 0 ⟨69710⟩ 289344

def event289346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69711⟩⟩) 1 ⟨39162⟩ 283044

def event289347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69711⟩⟩) (.sum [.predecessor 0 289345 .coefficient, .predecessor 1 289346 .coefficient])

def event289348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69711⟩⟩) (.sum [.result 289344 .summary, .result 283044 .summary])

def exact289349RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289349RawTermsValid :
    exact289349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69711⟩⟩) exact289349RawTerms .large 289347 (.finite 450666973253477225410675971981312) (some (289348))

def event289350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69712⟩⟩) 0 ⟨69711⟩ 289349

def event289351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69712⟩⟩) 1 ⟨41842⟩ 282564

def event289352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69712⟩⟩) (.sum [.predecessor 0 289350 .coefficient, .predecessor 1 289351 .coefficient])

def event289353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69712⟩⟩) (.sum [.result 289349 .summary, .result 282564 .summary])

def exact289354RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289354RawTermsValid :
    exact289354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69712⟩⟩) exact289354RawTerms .large 289352 (.finite 482860102375766054599486172037120) (some (289353))

def event289355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69713⟩⟩) 0 ⟨69712⟩ 289354

def event289356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69713⟩⟩) 1 ⟨44522⟩ 282084

def event289357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69713⟩⟩) (.sum [.predecessor 0 289355 .coefficient, .predecessor 1 289356 .coefficient])

def event289358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69713⟩⟩) (.sum [.result 289354 .summary, .result 282084 .summary])

def exact289359RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289359RawTermsValid :
    exact289359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69713⟩⟩) exact289359RawTerms .large 289357 (.finite 515053820849391945920019041353728) (some (289358))

def event289360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69714⟩⟩) 0 ⟨69713⟩ 289359

def event289361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69714⟩⟩) 1 ⟨47202⟩ 281604

def event289362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69714⟩⟩) (.sum [.predecessor 0 289360 .coefficient, .predecessor 1 289361 .coefficient])

def event289363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69714⟩⟩) (.sum [.result 289359 .summary, .result 281604 .summary])

def exact289364RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289364RawTermsValid :
    exact289364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289364 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69714⟩⟩) exact289364RawTerms .large 289362 (.finite 547248128674354899372274579931136) (some (289363))

def event289365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69715⟩⟩) 0 ⟨69714⟩ 289364

def event289366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69715⟩⟩) 1 ⟨49882⟩ 281124

def event289367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69715⟩⟩) (.sum [.predecessor 0 289365 .coefficient, .predecessor 1 289366 .coefficient])

def event289368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69715⟩⟩) (.sum [.result 289364 .summary, .result 281124 .summary])

def exact289369RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact289369RawTermsValid :
    exact289369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289369 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69715⟩⟩) exact289369RawTerms .large 289367 (.finite 579442632949763540201771008262144) (some (289368))

def event289370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71050⟩⟩) 0 ⟨69715⟩ 289369

def event289371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71050⟩⟩) 1 ⟨71048⟩ 280628

def event289372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71050⟩⟩) (.product (.predecessor 0 289370 .coefficient) (.predecessor 1 289371 .coefficient) (⟨false, false, none, none, none⟩))

def event289373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71050⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) [⟨.result 280628 .coefficient, false, none⟩])

def event289374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71050⟩⟩) (.product (.result 289369 .summary) (.transfer 289373) (⟨false, false, none, none, none⟩))

def event289375 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 17⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289376 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 29⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289377 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289378 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289377 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289379 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 16⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289380 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 28⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289381 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289382 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289381 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289383 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 15⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289384 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 27⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289385 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289386 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289385 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289387 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 14⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289388 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 26⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289389 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289390 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289389 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289391 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 13⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289392 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 25⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289393 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289394 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289393 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289395 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 12⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289396 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 24⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289397 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289398 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289397 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289399 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 11⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 22⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289401 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289401 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289403 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 10⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289404 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 21⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289405 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289406 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289405 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289407 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 9⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289408 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 35⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289409 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289410 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289409 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 8⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289412 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 34⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289413 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289414 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289413 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289415 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 7⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289416 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 33⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289417 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289418 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289417 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289419 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 6⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289420 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 32⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289421 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289422 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289421 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289423 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 5⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289424 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 31⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289425 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289426 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289425 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289427 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 4⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289428 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 30⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289429 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289430 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289429 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289431 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 3⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289432 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 23⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289433 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289434 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289433 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289435 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 2⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289436 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 20⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289437 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289438 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289437 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289439 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 1⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289440 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 19⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289441 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289442 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289441 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event289443 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 0⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event289444 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .operator (⟨289369, 18⟩, ⟨280628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event289445 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71050⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 280625)

def event289446 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71050⟩⟩, .relation 289445 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def exact289447RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩]

theorem exact289447RawTermsValid :
    exact289447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289447 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71050⟩⟩) exact289447RawTerms .large 289372 (.finite 6221717896068416040249469304417135687106560) (some (289374))

def event289448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68310⟩⟩) 0 ⟨66191⟩ 14034

def event289449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68310⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact289450RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68310⟩⟩]⟩, (1)⟩]

theorem exact289450RawTermsValid :
    exact289450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289450 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68310⟩⟩) exact289450RawTerms (.finite 5647228698) 289449 .exactZero (none)

def event289451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68312⟩⟩) 0 ⟨68310⟩ 289450

def event289452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68312⟩⟩) 1 ⟨2370⟩ 4

def event289453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68312⟩⟩) (.scale (.predecessor 0 289451 .coefficient) (.value (.predecessor 1 289452 .coefficient)))

def exact289454RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68310⟩⟩]⟩, (1)⟩]

theorem exact289454RawTermsValid :
    exact289454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68312⟩⟩) exact289454RawTerms (.finite 5647228698) 289453 .exactZero (none)

def event289455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68313⟩⟩) 0 ⟨5491⟩ 280745

def event289456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68313⟩⟩) 1 ⟨68312⟩ 289454

def event289457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68313⟩⟩) (.product (.predecessor 0 289455 .coefficient) (.predecessor 1 289456 .coefficient) (⟨false, false, none, none, none⟩))

def event289458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68313⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68310⟩⟩]⟩) [⟨.result 289450 .coefficient, false, none⟩])

def event289459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68313⟩⟩) (.product (.result 280745 .summary) (.transfer 289458) (⟨false, false, none, none, none⟩))

def event289460 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .operator (⟨280745, 0⟩, ⟨289454, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68310⟩⟩]⟩, (1)⟩)

def event289461 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68311⟩⟩)

def event289462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event289463 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event289464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event289465 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event289466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event289467 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event289468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event289469 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event289470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 289469

def event289471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 289467

def event289472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 289470 .coefficient) (.value (.predecessor 1 289471 .coefficient)))

def event289473 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event289474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 289473

def event289475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 289465

def event289476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 289474 .coefficient, .predecessor 1 289475 .coefficient])

def event289477 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event289478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 289477

def event289479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 289463

def event289480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 289479 .coefficient))

def event289481 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event289482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47690⟩⟩) 0 ⟨5487⟩ 289481

def event289483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47690⟩⟩) (.authority (.programFamilyFact))

def exact289484RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47690⟩⟩], []⟩, (1)⟩]

theorem exact289484RawTermsValid :
    exact289484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47690⟩⟩) exact289484RawTerms (.finite 60) 289483 .exactZero (none)

def event289485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14991⟩⟩) 0 ⟨5487⟩ 289481

def event289486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14991⟩⟩) (.authority (.programFamilyFact))

def exact289487RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩], []⟩, (1)⟩]

theorem exact289487RawTermsValid :
    exact289487RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289487 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14991⟩⟩) exact289487RawTerms (.finite 60) 289486 .exactZero (none)

def event289488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47691⟩⟩) 0 ⟨14991⟩ 289487

def event289489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47691⟩⟩) 1 ⟨47690⟩ 289484

def event289490 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47691⟩⟩) (.product (.predecessor 0 289488 .coefficient) (.predecessor 1 289489 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event289491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47691⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], []⟩) [⟨.result 289487 .coefficient, true, some 1⟩, ⟨.result 289484 .coefficient, true, some 1⟩])

def event289492 : Event := .survivorFold (1) 289491

def exact289493RawTerms : List Term := []

theorem exact289493RawTermsValid :
    exact289493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47691⟩⟩) exact289493RawTerms (.finite 3600) 289490 (.finite 3600) (some (289491))

def event289494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47692⟩⟩) 0 ⟨47691⟩ 289493

def event289495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47692⟩⟩) (.identity (.predecessor 0 289494 .coefficient))

def event289496 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47692⟩⟩) (.finite 3600)

def event289497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48100⟩⟩) 0 ⟨47692⟩ 289496

def event289498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48100⟩⟩) (.authority (.programFamilyFact))

def exact289499RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], []⟩, (1)⟩]

theorem exact289499RawTermsValid :
    exact289499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48100⟩⟩) exact289499RawTerms (.finite 60) 289498 .exactZero (none)

def event289500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48101⟩⟩) 0 ⟨48100⟩ 289499

def event289501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48101⟩⟩) (.identity (.predecessor 0 289500 .coefficient))

def event289502 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48101⟩⟩) (.finite 60)

def event289503 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48285⟩⟩) 0 ⟨48101⟩ 289502

def event289504 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48285⟩⟩) (.authority (.programFamilyFact))

def exact289505RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], []⟩, (1)⟩]

theorem exact289505RawTermsValid :
    exact289505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289505 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48285⟩⟩) exact289505RawTerms (.finite 63) 289504 .exactZero (none)

def event289506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45010⟩⟩) 0 ⟨5487⟩ 289481

def event289507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45010⟩⟩) (.authority (.programFamilyFact))

def exact289508RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45010⟩⟩], []⟩, (1)⟩]

theorem exact289508RawTermsValid :
    exact289508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45010⟩⟩) exact289508RawTerms (.finite 58) 289507 .exactZero (none)

def event289509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14691⟩⟩) 0 ⟨5487⟩ 289481

def event289510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14691⟩⟩) (.authority (.programFamilyFact))

def exact289511RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14691⟩⟩], []⟩, (1)⟩]

theorem exact289511RawTermsValid :
    exact289511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289511 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14691⟩⟩) exact289511RawTerms (.finite 58) 289510 .exactZero (none)

def event289512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45011⟩⟩) 0 ⟨14691⟩ 289511

def event289513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45011⟩⟩) 1 ⟨45010⟩ 289508

def event289514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45011⟩⟩) (.product (.predecessor 0 289512 .coefficient) (.predecessor 1 289513 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event289515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45011⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14691⟩⟩, ⟨.program ⟨257⟩, ⟨45010⟩⟩], []⟩) [⟨.result 289511 .coefficient, true, some 1⟩, ⟨.result 289508 .coefficient, true, some 1⟩])

def event289516 : Event := .survivorFold (1) 289515

def exact289517RawTerms : List Term := []

theorem exact289517RawTermsValid :
    exact289517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289517 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45011⟩⟩) exact289517RawTerms (.finite 3364) 289514 (.finite 3364) (some (289515))

def event289518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45012⟩⟩) 0 ⟨45011⟩ 289517

def event289519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45012⟩⟩) (.identity (.predecessor 0 289518 .coefficient))

def event289520 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45012⟩⟩) (.finite 3364)

def event289521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45420⟩⟩) 0 ⟨45012⟩ 289520

def event289522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45420⟩⟩) (.authority (.programFamilyFact))

def exact289523RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45420⟩⟩], []⟩, (1)⟩]

theorem exact289523RawTermsValid :
    exact289523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45420⟩⟩) exact289523RawTerms (.finite 58) 289522 .exactZero (none)

def event289524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45421⟩⟩) 0 ⟨45420⟩ 289523

def event289525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45421⟩⟩) (.identity (.predecessor 0 289524 .coefficient))

def event289526 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45421⟩⟩) (.finite 58)

def event289527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45605⟩⟩) 0 ⟨45421⟩ 289526

def event289528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45605⟩⟩) (.authority (.programFamilyFact))

def exact289529RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], []⟩, (1)⟩]

theorem exact289529RawTermsValid :
    exact289529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45605⟩⟩) exact289529RawTerms (.finite 63) 289528 .exactZero (none)

def event289530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42330⟩⟩) 0 ⟨5487⟩ 289481

def event289531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42330⟩⟩) (.authority (.programFamilyFact))

def exact289532RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42330⟩⟩], []⟩, (1)⟩]

theorem exact289532RawTermsValid :
    exact289532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289532 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42330⟩⟩) exact289532RawTerms (.finite 52) 289531 .exactZero (none)

def event289533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14391⟩⟩) 0 ⟨5487⟩ 289481

def event289534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14391⟩⟩) (.authority (.programFamilyFact))

def exact289535RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14391⟩⟩], []⟩, (1)⟩]

theorem exact289535RawTermsValid :
    exact289535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14391⟩⟩) exact289535RawTerms (.finite 52) 289534 .exactZero (none)

def eventLeaf18080 : Array AnnotatedEvent := #[
  { event := event289280
    frameStart := 0 },
  { event := event289281
    frameStart := 0 },
  { event := event289282
    frameStart := 0 },
  { event := event289283
    frameStart := 0 },
  { event := event289284
    frameStart := 0 },
  { event := event289285
    frameStart := 0 },
  { event := event289286
    frameStart := 0 },
  { event := event289287
    frameStart := 0 },
  { event := event289288
    frameStart := 0 },
  { event := event289289
    frameStart := 0 },
  { event := event289290
    frameStart := 0 },
  { event := event289291
    frameStart := 0 },
  { event := event289292
    frameStart := 0 },
  { event := event289293
    frameStart := 0 },
  { event := event289294
    frameStart := 0 },
  { event := event289295
    frameStart := 0 }
]

def eventLeaf18081 : Array AnnotatedEvent := #[
  { event := event289296
    frameStart := 0 },
  { event := event289297
    frameStart := 0 },
  { event := event289298
    frameStart := 0 },
  { event := event289299
    frameStart := 0 },
  { event := event289300
    frameStart := 0 },
  { event := event289301
    frameStart := 0 },
  { event := event289302
    frameStart := 0 },
  { event := event289303
    frameStart := 0 },
  { event := event289304
    frameStart := 0 },
  { event := event289305
    frameStart := 0 },
  { event := event289306
    frameStart := 0 },
  { event := event289307
    frameStart := 0 },
  { event := event289308
    frameStart := 0 },
  { event := event289309
    frameStart := 0 },
  { event := event289310
    frameStart := 0 },
  { event := event289311
    frameStart := 0 }
]

def eventLeaf18082 : Array AnnotatedEvent := #[
  { event := event289312
    frameStart := 0 },
  { event := event289313
    frameStart := 0 },
  { event := event289314
    frameStart := 0 },
  { event := event289315
    frameStart := 0 },
  { event := event289316
    frameStart := 0 },
  { event := event289317
    frameStart := 0 },
  { event := event289318
    frameStart := 0 },
  { event := event289319
    frameStart := 0 },
  { event := event289320
    frameStart := 0 },
  { event := event289321
    frameStart := 0 },
  { event := event289322
    frameStart := 0 },
  { event := event289323
    frameStart := 0 },
  { event := event289324
    frameStart := 0 },
  { event := event289325
    frameStart := 0 },
  { event := event289326
    frameStart := 0 },
  { event := event289327
    frameStart := 0 }
]

def eventLeaf18083 : Array AnnotatedEvent := #[
  { event := event289328
    frameStart := 0 },
  { event := event289329
    frameStart := 0 },
  { event := event289330
    frameStart := 0 },
  { event := event289331
    frameStart := 0 },
  { event := event289332
    frameStart := 0 },
  { event := event289333
    frameStart := 0 },
  { event := event289334
    frameStart := 0 },
  { event := event289335
    frameStart := 0 },
  { event := event289336
    frameStart := 0 },
  { event := event289337
    frameStart := 0 },
  { event := event289338
    frameStart := 0 },
  { event := event289339
    frameStart := 0 },
  { event := event289340
    frameStart := 0 },
  { event := event289341
    frameStart := 0 },
  { event := event289342
    frameStart := 0 },
  { event := event289343
    frameStart := 0 }
]

def eventLeaf18084 : Array AnnotatedEvent := #[
  { event := event289344
    frameStart := 0 },
  { event := event289345
    frameStart := 0 },
  { event := event289346
    frameStart := 0 },
  { event := event289347
    frameStart := 0 },
  { event := event289348
    frameStart := 0 },
  { event := event289349
    frameStart := 0 },
  { event := event289350
    frameStart := 0 },
  { event := event289351
    frameStart := 0 },
  { event := event289352
    frameStart := 0 },
  { event := event289353
    frameStart := 0 },
  { event := event289354
    frameStart := 0 },
  { event := event289355
    frameStart := 0 },
  { event := event289356
    frameStart := 0 },
  { event := event289357
    frameStart := 0 },
  { event := event289358
    frameStart := 0 },
  { event := event289359
    frameStart := 0 }
]

def eventLeaf18085 : Array AnnotatedEvent := #[
  { event := event289360
    frameStart := 0 },
  { event := event289361
    frameStart := 0 },
  { event := event289362
    frameStart := 0 },
  { event := event289363
    frameStart := 0 },
  { event := event289364
    frameStart := 0 },
  { event := event289365
    frameStart := 0 },
  { event := event289366
    frameStart := 0 },
  { event := event289367
    frameStart := 0 },
  { event := event289368
    frameStart := 0 },
  { event := event289369
    frameStart := 0 },
  { event := event289370
    frameStart := 0 },
  { event := event289371
    frameStart := 0 },
  { event := event289372
    frameStart := 0 },
  { event := event289373
    frameStart := 0 },
  { event := event289374
    frameStart := 0 },
  { event := event289375
    frameStart := 0 }
]

def eventLeaf18086 : Array AnnotatedEvent := #[
  { event := event289376
    frameStart := 0 },
  { event := event289377
    frameStart := 0 },
  { event := event289378
    frameStart := 0 },
  { event := event289379
    frameStart := 0 },
  { event := event289380
    frameStart := 0 },
  { event := event289381
    frameStart := 0 },
  { event := event289382
    frameStart := 0 },
  { event := event289383
    frameStart := 0 },
  { event := event289384
    frameStart := 0 },
  { event := event289385
    frameStart := 0 },
  { event := event289386
    frameStart := 0 },
  { event := event289387
    frameStart := 0 },
  { event := event289388
    frameStart := 0 },
  { event := event289389
    frameStart := 0 },
  { event := event289390
    frameStart := 0 },
  { event := event289391
    frameStart := 0 }
]

def eventLeaf18087 : Array AnnotatedEvent := #[
  { event := event289392
    frameStart := 0 },
  { event := event289393
    frameStart := 0 },
  { event := event289394
    frameStart := 0 },
  { event := event289395
    frameStart := 0 },
  { event := event289396
    frameStart := 0 },
  { event := event289397
    frameStart := 0 },
  { event := event289398
    frameStart := 0 },
  { event := event289399
    frameStart := 0 },
  { event := event289400
    frameStart := 0 },
  { event := event289401
    frameStart := 0 },
  { event := event289402
    frameStart := 0 },
  { event := event289403
    frameStart := 0 },
  { event := event289404
    frameStart := 0 },
  { event := event289405
    frameStart := 0 },
  { event := event289406
    frameStart := 0 },
  { event := event289407
    frameStart := 0 }
]

def eventLeaf18088 : Array AnnotatedEvent := #[
  { event := event289408
    frameStart := 0 },
  { event := event289409
    frameStart := 0 },
  { event := event289410
    frameStart := 0 },
  { event := event289411
    frameStart := 0 },
  { event := event289412
    frameStart := 0 },
  { event := event289413
    frameStart := 0 },
  { event := event289414
    frameStart := 0 },
  { event := event289415
    frameStart := 0 },
  { event := event289416
    frameStart := 0 },
  { event := event289417
    frameStart := 0 },
  { event := event289418
    frameStart := 0 },
  { event := event289419
    frameStart := 0 },
  { event := event289420
    frameStart := 0 },
  { event := event289421
    frameStart := 0 },
  { event := event289422
    frameStart := 0 },
  { event := event289423
    frameStart := 0 }
]

def eventLeaf18089 : Array AnnotatedEvent := #[
  { event := event289424
    frameStart := 0 },
  { event := event289425
    frameStart := 0 },
  { event := event289426
    frameStart := 0 },
  { event := event289427
    frameStart := 0 },
  { event := event289428
    frameStart := 0 },
  { event := event289429
    frameStart := 0 },
  { event := event289430
    frameStart := 0 },
  { event := event289431
    frameStart := 0 },
  { event := event289432
    frameStart := 0 },
  { event := event289433
    frameStart := 0 },
  { event := event289434
    frameStart := 0 },
  { event := event289435
    frameStart := 0 },
  { event := event289436
    frameStart := 0 },
  { event := event289437
    frameStart := 0 },
  { event := event289438
    frameStart := 0 },
  { event := event289439
    frameStart := 0 }
]

def eventLeaf18090 : Array AnnotatedEvent := #[
  { event := event289440
    frameStart := 0 },
  { event := event289441
    frameStart := 0 },
  { event := event289442
    frameStart := 0 },
  { event := event289443
    frameStart := 0 },
  { event := event289444
    frameStart := 0 },
  { event := event289445
    frameStart := 0 },
  { event := event289446
    frameStart := 0 },
  { event := event289447
    frameStart := 0 },
  { event := event289448
    frameStart := 0 },
  { event := event289449
    frameStart := 0 },
  { event := event289450
    frameStart := 0 },
  { event := event289451
    frameStart := 0 },
  { event := event289452
    frameStart := 0 },
  { event := event289453
    frameStart := 0 },
  { event := event289454
    frameStart := 0 },
  { event := event289455
    frameStart := 0 }
]

def eventLeaf18091 : Array AnnotatedEvent := #[
  { event := event289456
    frameStart := 0 },
  { event := event289457
    frameStart := 0 },
  { event := event289458
    frameStart := 0 },
  { event := event289459
    frameStart := 0 },
  { event := event289460
    frameStart := 0 },
  { event := event289461
    frameStart := 289461 },
  { event := event289462
    frameStart := 289461 },
  { event := event289463
    frameStart := 289461 },
  { event := event289464
    frameStart := 289461 },
  { event := event289465
    frameStart := 289461 },
  { event := event289466
    frameStart := 289461 },
  { event := event289467
    frameStart := 289461 },
  { event := event289468
    frameStart := 289461 },
  { event := event289469
    frameStart := 289461 },
  { event := event289470
    frameStart := 289461 },
  { event := event289471
    frameStart := 289461 }
]

def eventLeaf18092 : Array AnnotatedEvent := #[
  { event := event289472
    frameStart := 289461 },
  { event := event289473
    frameStart := 289461 },
  { event := event289474
    frameStart := 289461 },
  { event := event289475
    frameStart := 289461 },
  { event := event289476
    frameStart := 289461 },
  { event := event289477
    frameStart := 289461 },
  { event := event289478
    frameStart := 289461 },
  { event := event289479
    frameStart := 289461 },
  { event := event289480
    frameStart := 289461 },
  { event := event289481
    frameStart := 289461 },
  { event := event289482
    frameStart := 289461 },
  { event := event289483
    frameStart := 289461 },
  { event := event289484
    frameStart := 289461 },
  { event := event289485
    frameStart := 289461 },
  { event := event289486
    frameStart := 289461 },
  { event := event289487
    frameStart := 289461 }
]

def eventLeaf18093 : Array AnnotatedEvent := #[
  { event := event289488
    frameStart := 289461 },
  { event := event289489
    frameStart := 289461 },
  { event := event289490
    frameStart := 289461 },
  { event := event289491
    frameStart := 289461 },
  { event := event289492
    frameStart := 289461 },
  { event := event289493
    frameStart := 289461 },
  { event := event289494
    frameStart := 289461 },
  { event := event289495
    frameStart := 289461 },
  { event := event289496
    frameStart := 289461 },
  { event := event289497
    frameStart := 289461 },
  { event := event289498
    frameStart := 289461 },
  { event := event289499
    frameStart := 289461 },
  { event := event289500
    frameStart := 289461 },
  { event := event289501
    frameStart := 289461 },
  { event := event289502
    frameStart := 289461 },
  { event := event289503
    frameStart := 289461 }
]

def eventLeaf18094 : Array AnnotatedEvent := #[
  { event := event289504
    frameStart := 289461 },
  { event := event289505
    frameStart := 289461 },
  { event := event289506
    frameStart := 289461 },
  { event := event289507
    frameStart := 289461 },
  { event := event289508
    frameStart := 289461 },
  { event := event289509
    frameStart := 289461 },
  { event := event289510
    frameStart := 289461 },
  { event := event289511
    frameStart := 289461 },
  { event := event289512
    frameStart := 289461 },
  { event := event289513
    frameStart := 289461 },
  { event := event289514
    frameStart := 289461 },
  { event := event289515
    frameStart := 289461 },
  { event := event289516
    frameStart := 289461 },
  { event := event289517
    frameStart := 289461 },
  { event := event289518
    frameStart := 289461 },
  { event := event289519
    frameStart := 289461 }
]

def eventLeaf18095 : Array AnnotatedEvent := #[
  { event := event289520
    frameStart := 289461 },
  { event := event289521
    frameStart := 289461 },
  { event := event289522
    frameStart := 289461 },
  { event := event289523
    frameStart := 289461 },
  { event := event289524
    frameStart := 289461 },
  { event := event289525
    frameStart := 289461 },
  { event := event289526
    frameStart := 289461 },
  { event := event289527
    frameStart := 289461 },
  { event := event289528
    frameStart := 289461 },
  { event := event289529
    frameStart := 289461 },
  { event := event289530
    frameStart := 289461 },
  { event := event289531
    frameStart := 289461 },
  { event := event289532
    frameStart := 289461 },
  { event := event289533
    frameStart := 289461 },
  { event := event289534
    frameStart := 289461 },
  { event := event289535
    frameStart := 289461 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1130
