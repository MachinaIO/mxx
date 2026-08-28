import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events616

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event157696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20563⟩⟩) 0 ⟨17680⟩ 157695

def event157697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20563⟩⟩) 1 ⟨20562⟩ 157213

def event157698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20563⟩⟩) (.sum [.predecessor 0 157696 .coefficient, .predecessor 1 157697 .coefficient])

def event157699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20563⟩⟩) (.sum [.result 157695 .summary, .result 157213 .summary])

def exact157700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157700RawTermsValid :
    exact157700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20563⟩⟩) exact157700RawTerms .large 157698 (.finite 64377712650190257467641695830016) (some (157699))

def event157701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23783⟩⟩) 0 ⟨20563⟩ 157700

def event157702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23783⟩⟩) 1 ⟨23782⟩ 156731

def event157703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23783⟩⟩) (.sum [.predecessor 0 157701 .coefficient, .predecessor 1 157702 .coefficient])

def event157704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23783⟩⟩) (.sum [.result 157700 .summary, .result 156731 .summary])

def exact157705RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157705RawTermsValid :
    exact157705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157705 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23783⟩⟩) exact157705RawTerms .large 157703 (.finite 96566716313119651734393211060224) (some (157704))

def event157706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33803⟩⟩) 0 ⟨23783⟩ 157705

def event157707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33803⟩⟩) 1 ⟨33802⟩ 156249

def event157708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33803⟩⟩) (.sum [.predecessor 0 157706 .coefficient, .predecessor 1 157707 .coefficient])

def event157709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33803⟩⟩) (.sum [.result 157705 .summary, .result 156249 .summary])

def exact157710RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157710RawTermsValid :
    exact157710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157710 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33803⟩⟩) exact157710RawTerms .large 157708 (.finite 128755916426494733378385616044032) (some (157709))

def event157711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52863⟩⟩) 0 ⟨33803⟩ 157710

def event157712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52863⟩⟩) 1 ⟨52862⟩ 155767

def event157713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52863⟩⟩) (.sum [.predecessor 0 157711 .coefficient, .predecessor 1 157712 .coefficient])

def event157714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52863⟩⟩) (.sum [.result 157710 .summary, .result 155767 .summary])

def exact157715RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157715RawTermsValid :
    exact157715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52863⟩⟩) exact157715RawTerms .large 157713 (.finite 160945509440761189776859800535040) (some (157714))

def event157716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55843⟩⟩) 0 ⟨52863⟩ 157715

def event157717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55843⟩⟩) 1 ⟨55842⟩ 155285

def event157718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55843⟩⟩) (.sum [.predecessor 0 157716 .coefficient, .predecessor 1 157717 .coefficient])

def event157719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55843⟩⟩) (.sum [.result 157715 .summary, .result 155285 .summary])

def exact157720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157720RawTermsValid :
    exact157720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55843⟩⟩) exact157720RawTerms .large 157718 (.finite 193135298905473333552574874779648) (some (157719))

def event157721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58823⟩⟩) 0 ⟨55843⟩ 157720

def event157722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58823⟩⟩) 1 ⟨58822⟩ 154803

def event157723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58823⟩⟩) (.sum [.predecessor 0 157721 .coefficient, .predecessor 1 157722 .coefficient])

def event157724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58823⟩⟩) (.sum [.result 157720 .summary, .result 154803 .summary])

def exact157725RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157725RawTermsValid :
    exact157725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58823⟩⟩) exact157725RawTerms .large 157723 (.finite 225325481271076852082771728531456) (some (157724))

def event157726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61803⟩⟩) 0 ⟨58823⟩ 157725

def event157727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61803⟩⟩) 1 ⟨61802⟩ 154321

def event157728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61803⟩⟩) (.sum [.predecessor 0 157726 .coefficient, .predecessor 1 157727 .coefficient])

def event157729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61803⟩⟩) (.sum [.result 157725 .summary, .result 154321 .summary])

def exact157730RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157730RawTermsValid :
    exact157730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157730 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61803⟩⟩) exact157730RawTerms .large 157728 (.finite 257515860087126057990209472036864) (some (157729))

def event157731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64783⟩⟩) 0 ⟨61803⟩ 157730

def event157732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64783⟩⟩) 1 ⟨64782⟩ 153839

def event157733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64783⟩⟩) (.sum [.predecessor 0 157731 .coefficient, .predecessor 1 157732 .coefficient])

def event157734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64783⟩⟩) (.sum [.result 157730 .summary, .result 153839 .summary])

def exact157735RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157735RawTermsValid :
    exact157735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157735 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64783⟩⟩) exact157735RawTerms .large 157733 (.finite 289706631804066638652128995049472) (some (157734))

def event157736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69944⟩⟩) 0 ⟨64783⟩ 157735

def event157737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69944⟩⟩) 1 ⟨69943⟩ 153357

def event157738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69944⟩⟩) (.sum [.predecessor 0 157736 .coefficient, .predecessor 1 157737 .coefficient])

def event157739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69944⟩⟩) (.sum [.result 157735 .summary, .result 153357 .summary])

def exact157740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157740RawTermsValid :
    exact157740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69944⟩⟩) exact157740RawTerms .large 157738 (.finite 321897992872344281445771187322880) (some (157739))

def event157741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69945⟩⟩) 0 ⟨69944⟩ 157740

def event157742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69945⟩⟩) 1 ⟨28217⟩ 152875

def event157743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69945⟩⟩) (.sum [.predecessor 0 157741 .coefficient, .predecessor 1 157742 .coefficient])

def event157744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69945⟩⟩) (.sum [.result 157740 .summary, .result 152875 .summary])

def exact157745RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157745RawTermsValid :
    exact157745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69945⟩⟩) exact157745RawTerms .large 157743 (.finite 354089550391067611616654269349888) (some (157744))

def event157746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69946⟩⟩) 0 ⟨69945⟩ 157745

def event157747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69946⟩⟩) 1 ⟨30897⟩ 152393

def event157748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69946⟩⟩) (.sum [.predecessor 0 157746 .coefficient, .predecessor 1 157747 .coefficient])

def event157749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69946⟩⟩) (.sum [.result 157745 .summary, .result 152393 .summary])

def exact157750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157750RawTermsValid :
    exact157750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69946⟩⟩) exact157750RawTerms .large 157748 (.finite 386281697261128003919260020637696) (some (157749))

def event157751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69947⟩⟩) 0 ⟨69946⟩ 157750

def event157752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69947⟩⟩) 1 ⟨36557⟩ 151911

def event157753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69947⟩⟩) (.sum [.predecessor 0 157751 .coefficient, .predecessor 1 157752 .coefficient])

def event157754 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69947⟩⟩) (.sum [.result 157750 .summary, .result 151911 .summary])

def exact157755RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157755RawTermsValid :
    exact157755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157755 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69947⟩⟩) exact157755RawTerms .large 157753 (.finite 418474237032079770976347551432704) (some (157754))

def event157756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69948⟩⟩) 0 ⟨69947⟩ 157755

def event157757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69948⟩⟩) 1 ⟨39237⟩ 151429

def event157758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69948⟩⟩) (.sum [.predecessor 0 157756 .coefficient, .predecessor 1 157757 .coefficient])

def event157759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69948⟩⟩) (.sum [.result 157755 .summary, .result 151429 .summary])

def exact157760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157760RawTermsValid :
    exact157760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69948⟩⟩) exact157760RawTerms .large 157758 (.finite 450666973253477225410675971981312) (some (157759))

def event157761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69949⟩⟩) 0 ⟨69948⟩ 157760

def event157762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69949⟩⟩) 1 ⟨41917⟩ 150947

def event157763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69949⟩⟩) (.sum [.predecessor 0 157761 .coefficient, .predecessor 1 157762 .coefficient])

def event157764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69949⟩⟩) (.sum [.result 157760 .summary, .result 150947 .summary])

def exact157765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157765RawTermsValid :
    exact157765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69949⟩⟩) exact157765RawTerms .large 157763 (.finite 482860102375766054599486172037120) (some (157764))

def event157766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69950⟩⟩) 0 ⟨69949⟩ 157765

def event157767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69950⟩⟩) 1 ⟨44597⟩ 150465

def event157768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69950⟩⟩) (.sum [.predecessor 0 157766 .coefficient, .predecessor 1 157767 .coefficient])

def event157769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69950⟩⟩) (.sum [.result 157765 .summary, .result 150465 .summary])

def exact157770RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157770RawTermsValid :
    exact157770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69950⟩⟩) exact157770RawTerms .large 157768 (.finite 515053820849391945920019041353728) (some (157769))

def event157771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69951⟩⟩) 0 ⟨69950⟩ 157770

def event157772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69951⟩⟩) 1 ⟨47277⟩ 149983

def event157773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69951⟩⟩) (.sum [.predecessor 0 157771 .coefficient, .predecessor 1 157772 .coefficient])

def event157774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69951⟩⟩) (.sum [.result 157770 .summary, .result 149983 .summary])

def exact157775RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨45644⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157775RawTermsValid :
    exact157775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69951⟩⟩) exact157775RawTerms .large 157773 (.finite 547248128674354899372274579931136) (some (157774))

def event157776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69952⟩⟩) 0 ⟨69951⟩ 157775

def event157777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69952⟩⟩) 1 ⟨49957⟩ 149501

def event157778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69952⟩⟩) (.sum [.predecessor 0 157776 .coefficient, .predecessor 1 157777 .coefficient])

def event157779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69952⟩⟩) (.sum [.result 157775 .summary, .result 149501 .summary])

def exact157780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨45644⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48324⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact157780RawTermsValid :
    exact157780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69952⟩⟩) exact157780RawTerms .large 157778 (.finite 579442632949763540201771008262144) (some (157779))

def event157781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71144⟩⟩) 0 ⟨69952⟩ 157780

def event157782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71144⟩⟩) 1 ⟨71142⟩ 149003

def event157783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71144⟩⟩) (.product (.predecessor 0 157781 .coefficient) (.predecessor 1 157782 .coefficient) (⟨false, false, none, none, none⟩))

def event157784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71144⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) [⟨.result 149003 .coefficient, false, none⟩])

def event157785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71144⟩⟩) (.product (.result 157780 .summary) (.transfer 157784) (⟨false, false, none, none, none⟩))

def event157786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 17⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 29⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48324⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157788 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48324⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157788 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48324⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 16⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 28⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨45644⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157792 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨45644⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157792 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨45644⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 15⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 27⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157796 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157796 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42960⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157798 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 14⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157799 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 26⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157800 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157800 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40280⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 13⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 25⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157804 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157805 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157804 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 12⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 24⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157808 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157808 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 11⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 22⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157812 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157813 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157812 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157814 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 10⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157815 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 21⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157816 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157817 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157816 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 9⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157819 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 35⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157820 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157820 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157822 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 8⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157823 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 34⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157824 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157825 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157824 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157826 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 7⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 33⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157828 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157829 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157828 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 6⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 32⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157832 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157832 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157834 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 5⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 31⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157836 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157836 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 4⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 30⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157840 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157840 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 3⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 23⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157844 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157844 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157846 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 2⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 20⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157848 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157848 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 1⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 19⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157852 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157853 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157852 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def event157854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 0⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩)

def event157855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .operator (⟨157780, 18⟩, ⟨149003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (-1)⟩)

def event157856 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71144⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 149000)

def event157857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71144⟩⟩, .relation 157856 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩)

def exact157858RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29260⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34924⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40280⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨42960⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨45644⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48324⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩, (-1)⟩]

theorem exact157858RawTermsValid :
    exact157858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71144⟩⟩) exact157858RawTerms .large 157783 (.finite 6221717896068416040249469304417135687106560) (some (157785))

def event157859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68340⟩⟩) 0 ⟨66401⟩ 7308

def event157860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68340⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact157861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68340⟩⟩]⟩, (1)⟩]

theorem exact157861RawTermsValid :
    exact157861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68340⟩⟩) exact157861RawTerms (.finite 5647228698) 157860 .exactZero (none)

def event157862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68342⟩⟩) 0 ⟨68340⟩ 157861

def event157863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68342⟩⟩) 1 ⟨2370⟩ 4

def event157864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68342⟩⟩) (.scale (.predecessor 0 157862 .coefficient) (.value (.predecessor 1 157863 .coefficient)))

def exact157865RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68340⟩⟩]⟩, (1)⟩]

theorem exact157865RawTermsValid :
    exact157865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68342⟩⟩) exact157865RawTerms (.finite 5647228698) 157864 .exactZero (none)

def event157866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68343⟩⟩) 0 ⟨5545⟩ 149120

def event157867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68343⟩⟩) 1 ⟨68342⟩ 157865

def event157868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68343⟩⟩) (.product (.predecessor 0 157866 .coefficient) (.predecessor 1 157867 .coefficient) (⟨false, false, none, none, none⟩))

def event157869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68343⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68340⟩⟩]⟩) [⟨.result 157861 .coefficient, false, none⟩])

def event157870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68343⟩⟩) (.product (.result 149120 .summary) (.transfer 157869) (⟨false, false, none, none, none⟩))

def event157871 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68343⟩⟩, .operator (⟨149120, 0⟩, ⟨157865, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68340⟩⟩]⟩, (1)⟩)

def event157872 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68341⟩⟩)

def event157873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event157874 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event157875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4614⟩⟩) (.authority (.operator))

def event157876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4614⟩⟩) (.finite 10)

def event157877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event157878 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event157879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event157880 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event157881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 157880

def event157882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 157878

def event157883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 157881 .coefficient) (.value (.predecessor 1 157882 .coefficient)))

def event157884 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event157885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4616⟩⟩) 0 ⟨392⟩ 157884

def event157886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4616⟩⟩) 1 ⟨4614⟩ 157876

def event157887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4616⟩⟩) (.sum [.predecessor 0 157885 .coefficient, .predecessor 1 157886 .coefficient])

def event157888 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4616⟩⟩) (.finite 655350)

def event157889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5541⟩⟩) 0 ⟨4616⟩ 157888

def event157890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5541⟩⟩) 1 ⟨5426⟩ 157874

def event157891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5541⟩⟩) (.identity (.predecessor 1 157890 .coefficient))

def event157892 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5541⟩⟩) (.finite 655360)

def event157893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47762⟩⟩) 0 ⟨5541⟩ 157892

def event157894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47762⟩⟩) (.authority (.programFamilyFact))

def exact157895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47762⟩⟩], []⟩, (1)⟩]

theorem exact157895RawTermsValid :
    exact157895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47762⟩⟩) exact157895RawTerms (.finite 60) 157894 .exactZero (none)

def event157896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15036⟩⟩) 0 ⟨5541⟩ 157892

def event157897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15036⟩⟩) (.authority (.programFamilyFact))

def exact157898RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩], []⟩, (1)⟩]

theorem exact157898RawTermsValid :
    exact157898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15036⟩⟩) exact157898RawTerms (.finite 60) 157897 .exactZero (none)

def event157899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47763⟩⟩) 0 ⟨15036⟩ 157898

def event157900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47763⟩⟩) 1 ⟨47762⟩ 157895

def event157901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47763⟩⟩) (.product (.predecessor 0 157899 .coefficient) (.predecessor 1 157900 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event157902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47763⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], []⟩) [⟨.result 157898 .coefficient, true, some 1⟩, ⟨.result 157895 .coefficient, true, some 1⟩])

def event157903 : Event := .survivorFold (1) 157902

def exact157904RawTerms : List Term := []

theorem exact157904RawTermsValid :
    exact157904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47763⟩⟩) exact157904RawTerms (.finite 3600) 157901 (.finite 3600) (some (157902))

def event157905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47764⟩⟩) 0 ⟨47763⟩ 157904

def event157906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47764⟩⟩) (.identity (.predecessor 0 157905 .coefficient))

def event157907 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47764⟩⟩) (.finite 3600)

def event157908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48124⟩⟩) 0 ⟨47764⟩ 157907

def event157909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48124⟩⟩) (.authority (.programFamilyFact))

def exact157910RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48124⟩⟩], []⟩, (1)⟩]

theorem exact157910RawTermsValid :
    exact157910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48124⟩⟩) exact157910RawTerms (.finite 60) 157909 .exactZero (none)

def event157911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48125⟩⟩) 0 ⟨48124⟩ 157910

def event157912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48125⟩⟩) (.identity (.predecessor 0 157911 .coefficient))

def event157913 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48125⟩⟩) (.finite 60)

def event157914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48324⟩⟩) 0 ⟨48125⟩ 157913

def event157915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48324⟩⟩) (.authority (.programFamilyFact))

def exact157916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48324⟩⟩], []⟩, (1)⟩]

theorem exact157916RawTermsValid :
    exact157916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48324⟩⟩) exact157916RawTerms (.finite 63) 157915 .exactZero (none)

def event157917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45082⟩⟩) 0 ⟨5541⟩ 157892

def event157918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45082⟩⟩) (.authority (.programFamilyFact))

def exact157919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45082⟩⟩], []⟩, (1)⟩]

theorem exact157919RawTermsValid :
    exact157919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45082⟩⟩) exact157919RawTerms (.finite 58) 157918 .exactZero (none)

def event157920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14736⟩⟩) 0 ⟨5541⟩ 157892

def event157921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14736⟩⟩) (.authority (.programFamilyFact))

def exact157922RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩], []⟩, (1)⟩]

theorem exact157922RawTermsValid :
    exact157922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14736⟩⟩) exact157922RawTerms (.finite 58) 157921 .exactZero (none)

def event157923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45083⟩⟩) 0 ⟨14736⟩ 157922

def event157924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45083⟩⟩) 1 ⟨45082⟩ 157919

def event157925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45083⟩⟩) (.product (.predecessor 0 157923 .coefficient) (.predecessor 1 157924 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event157926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45083⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], []⟩) [⟨.result 157922 .coefficient, true, some 1⟩, ⟨.result 157919 .coefficient, true, some 1⟩])

def event157927 : Event := .survivorFold (1) 157926

def exact157928RawTerms : List Term := []

theorem exact157928RawTermsValid :
    exact157928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157928 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45083⟩⟩) exact157928RawTerms (.finite 3364) 157925 (.finite 3364) (some (157926))

def event157929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45084⟩⟩) 0 ⟨45083⟩ 157928

def event157930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45084⟩⟩) (.identity (.predecessor 0 157929 .coefficient))

def event157931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45084⟩⟩) (.finite 3364)

def event157932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45444⟩⟩) 0 ⟨45084⟩ 157931

def event157933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45444⟩⟩) (.authority (.programFamilyFact))

def exact157934RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45444⟩⟩], []⟩, (1)⟩]

theorem exact157934RawTermsValid :
    exact157934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45444⟩⟩) exact157934RawTerms (.finite 58) 157933 .exactZero (none)

def event157935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45445⟩⟩) 0 ⟨45444⟩ 157934

def event157936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45445⟩⟩) (.identity (.predecessor 0 157935 .coefficient))

def event157937 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45445⟩⟩) (.finite 58)

def event157938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45644⟩⟩) 0 ⟨45445⟩ 157937

def event157939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45644⟩⟩) (.authority (.programFamilyFact))

def exact157940RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45644⟩⟩], []⟩, (1)⟩]

theorem exact157940RawTermsValid :
    exact157940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45644⟩⟩) exact157940RawTerms (.finite 63) 157939 .exactZero (none)

def event157941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42402⟩⟩) 0 ⟨5541⟩ 157892

def event157942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42402⟩⟩) (.authority (.programFamilyFact))

def exact157943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42402⟩⟩], []⟩, (1)⟩]

theorem exact157943RawTermsValid :
    exact157943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42402⟩⟩) exact157943RawTerms (.finite 52) 157942 .exactZero (none)

def event157944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14436⟩⟩) 0 ⟨5541⟩ 157892

def event157945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14436⟩⟩) (.authority (.programFamilyFact))

def exact157946RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩], []⟩, (1)⟩]

theorem exact157946RawTermsValid :
    exact157946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event157946 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14436⟩⟩) exact157946RawTerms (.finite 52) 157945 .exactZero (none)

def event157947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42403⟩⟩) 0 ⟨14436⟩ 157946

def event157948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42403⟩⟩) 1 ⟨42402⟩ 157943

def event157949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42403⟩⟩) (.product (.predecessor 0 157947 .coefficient) (.predecessor 1 157948 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event157950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42403⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], []⟩) [⟨.result 157946 .coefficient, true, some 1⟩, ⟨.result 157943 .coefficient, true, some 1⟩])

def event157951 : Event := .survivorFold (1) 157950

def eventLeaf9856 : Array AnnotatedEvent := #[
  { event := event157696
    frameStart := 0 },
  { event := event157697
    frameStart := 0 },
  { event := event157698
    frameStart := 0 },
  { event := event157699
    frameStart := 0 },
  { event := event157700
    frameStart := 0 },
  { event := event157701
    frameStart := 0 },
  { event := event157702
    frameStart := 0 },
  { event := event157703
    frameStart := 0 },
  { event := event157704
    frameStart := 0 },
  { event := event157705
    frameStart := 0 },
  { event := event157706
    frameStart := 0 },
  { event := event157707
    frameStart := 0 },
  { event := event157708
    frameStart := 0 },
  { event := event157709
    frameStart := 0 },
  { event := event157710
    frameStart := 0 },
  { event := event157711
    frameStart := 0 }
]

def eventLeaf9857 : Array AnnotatedEvent := #[
  { event := event157712
    frameStart := 0 },
  { event := event157713
    frameStart := 0 },
  { event := event157714
    frameStart := 0 },
  { event := event157715
    frameStart := 0 },
  { event := event157716
    frameStart := 0 },
  { event := event157717
    frameStart := 0 },
  { event := event157718
    frameStart := 0 },
  { event := event157719
    frameStart := 0 },
  { event := event157720
    frameStart := 0 },
  { event := event157721
    frameStart := 0 },
  { event := event157722
    frameStart := 0 },
  { event := event157723
    frameStart := 0 },
  { event := event157724
    frameStart := 0 },
  { event := event157725
    frameStart := 0 },
  { event := event157726
    frameStart := 0 },
  { event := event157727
    frameStart := 0 }
]

def eventLeaf9858 : Array AnnotatedEvent := #[
  { event := event157728
    frameStart := 0 },
  { event := event157729
    frameStart := 0 },
  { event := event157730
    frameStart := 0 },
  { event := event157731
    frameStart := 0 },
  { event := event157732
    frameStart := 0 },
  { event := event157733
    frameStart := 0 },
  { event := event157734
    frameStart := 0 },
  { event := event157735
    frameStart := 0 },
  { event := event157736
    frameStart := 0 },
  { event := event157737
    frameStart := 0 },
  { event := event157738
    frameStart := 0 },
  { event := event157739
    frameStart := 0 },
  { event := event157740
    frameStart := 0 },
  { event := event157741
    frameStart := 0 },
  { event := event157742
    frameStart := 0 },
  { event := event157743
    frameStart := 0 }
]

def eventLeaf9859 : Array AnnotatedEvent := #[
  { event := event157744
    frameStart := 0 },
  { event := event157745
    frameStart := 0 },
  { event := event157746
    frameStart := 0 },
  { event := event157747
    frameStart := 0 },
  { event := event157748
    frameStart := 0 },
  { event := event157749
    frameStart := 0 },
  { event := event157750
    frameStart := 0 },
  { event := event157751
    frameStart := 0 },
  { event := event157752
    frameStart := 0 },
  { event := event157753
    frameStart := 0 },
  { event := event157754
    frameStart := 0 },
  { event := event157755
    frameStart := 0 },
  { event := event157756
    frameStart := 0 },
  { event := event157757
    frameStart := 0 },
  { event := event157758
    frameStart := 0 },
  { event := event157759
    frameStart := 0 }
]

def eventLeaf9860 : Array AnnotatedEvent := #[
  { event := event157760
    frameStart := 0 },
  { event := event157761
    frameStart := 0 },
  { event := event157762
    frameStart := 0 },
  { event := event157763
    frameStart := 0 },
  { event := event157764
    frameStart := 0 },
  { event := event157765
    frameStart := 0 },
  { event := event157766
    frameStart := 0 },
  { event := event157767
    frameStart := 0 },
  { event := event157768
    frameStart := 0 },
  { event := event157769
    frameStart := 0 },
  { event := event157770
    frameStart := 0 },
  { event := event157771
    frameStart := 0 },
  { event := event157772
    frameStart := 0 },
  { event := event157773
    frameStart := 0 },
  { event := event157774
    frameStart := 0 },
  { event := event157775
    frameStart := 0 }
]

def eventLeaf9861 : Array AnnotatedEvent := #[
  { event := event157776
    frameStart := 0 },
  { event := event157777
    frameStart := 0 },
  { event := event157778
    frameStart := 0 },
  { event := event157779
    frameStart := 0 },
  { event := event157780
    frameStart := 0 },
  { event := event157781
    frameStart := 0 },
  { event := event157782
    frameStart := 0 },
  { event := event157783
    frameStart := 0 },
  { event := event157784
    frameStart := 0 },
  { event := event157785
    frameStart := 0 },
  { event := event157786
    frameStart := 0 },
  { event := event157787
    frameStart := 0 },
  { event := event157788
    frameStart := 0 },
  { event := event157789
    frameStart := 0 },
  { event := event157790
    frameStart := 0 },
  { event := event157791
    frameStart := 0 }
]

def eventLeaf9862 : Array AnnotatedEvent := #[
  { event := event157792
    frameStart := 0 },
  { event := event157793
    frameStart := 0 },
  { event := event157794
    frameStart := 0 },
  { event := event157795
    frameStart := 0 },
  { event := event157796
    frameStart := 0 },
  { event := event157797
    frameStart := 0 },
  { event := event157798
    frameStart := 0 },
  { event := event157799
    frameStart := 0 },
  { event := event157800
    frameStart := 0 },
  { event := event157801
    frameStart := 0 },
  { event := event157802
    frameStart := 0 },
  { event := event157803
    frameStart := 0 },
  { event := event157804
    frameStart := 0 },
  { event := event157805
    frameStart := 0 },
  { event := event157806
    frameStart := 0 },
  { event := event157807
    frameStart := 0 }
]

def eventLeaf9863 : Array AnnotatedEvent := #[
  { event := event157808
    frameStart := 0 },
  { event := event157809
    frameStart := 0 },
  { event := event157810
    frameStart := 0 },
  { event := event157811
    frameStart := 0 },
  { event := event157812
    frameStart := 0 },
  { event := event157813
    frameStart := 0 },
  { event := event157814
    frameStart := 0 },
  { event := event157815
    frameStart := 0 },
  { event := event157816
    frameStart := 0 },
  { event := event157817
    frameStart := 0 },
  { event := event157818
    frameStart := 0 },
  { event := event157819
    frameStart := 0 },
  { event := event157820
    frameStart := 0 },
  { event := event157821
    frameStart := 0 },
  { event := event157822
    frameStart := 0 },
  { event := event157823
    frameStart := 0 }
]

def eventLeaf9864 : Array AnnotatedEvent := #[
  { event := event157824
    frameStart := 0 },
  { event := event157825
    frameStart := 0 },
  { event := event157826
    frameStart := 0 },
  { event := event157827
    frameStart := 0 },
  { event := event157828
    frameStart := 0 },
  { event := event157829
    frameStart := 0 },
  { event := event157830
    frameStart := 0 },
  { event := event157831
    frameStart := 0 },
  { event := event157832
    frameStart := 0 },
  { event := event157833
    frameStart := 0 },
  { event := event157834
    frameStart := 0 },
  { event := event157835
    frameStart := 0 },
  { event := event157836
    frameStart := 0 },
  { event := event157837
    frameStart := 0 },
  { event := event157838
    frameStart := 0 },
  { event := event157839
    frameStart := 0 }
]

def eventLeaf9865 : Array AnnotatedEvent := #[
  { event := event157840
    frameStart := 0 },
  { event := event157841
    frameStart := 0 },
  { event := event157842
    frameStart := 0 },
  { event := event157843
    frameStart := 0 },
  { event := event157844
    frameStart := 0 },
  { event := event157845
    frameStart := 0 },
  { event := event157846
    frameStart := 0 },
  { event := event157847
    frameStart := 0 },
  { event := event157848
    frameStart := 0 },
  { event := event157849
    frameStart := 0 },
  { event := event157850
    frameStart := 0 },
  { event := event157851
    frameStart := 0 },
  { event := event157852
    frameStart := 0 },
  { event := event157853
    frameStart := 0 },
  { event := event157854
    frameStart := 0 },
  { event := event157855
    frameStart := 0 }
]

def eventLeaf9866 : Array AnnotatedEvent := #[
  { event := event157856
    frameStart := 0 },
  { event := event157857
    frameStart := 0 },
  { event := event157858
    frameStart := 0 },
  { event := event157859
    frameStart := 0 },
  { event := event157860
    frameStart := 0 },
  { event := event157861
    frameStart := 0 },
  { event := event157862
    frameStart := 0 },
  { event := event157863
    frameStart := 0 },
  { event := event157864
    frameStart := 0 },
  { event := event157865
    frameStart := 0 },
  { event := event157866
    frameStart := 0 },
  { event := event157867
    frameStart := 0 },
  { event := event157868
    frameStart := 0 },
  { event := event157869
    frameStart := 0 },
  { event := event157870
    frameStart := 0 },
  { event := event157871
    frameStart := 0 }
]

def eventLeaf9867 : Array AnnotatedEvent := #[
  { event := event157872
    frameStart := 157872 },
  { event := event157873
    frameStart := 157872 },
  { event := event157874
    frameStart := 157872 },
  { event := event157875
    frameStart := 157872 },
  { event := event157876
    frameStart := 157872 },
  { event := event157877
    frameStart := 157872 },
  { event := event157878
    frameStart := 157872 },
  { event := event157879
    frameStart := 157872 },
  { event := event157880
    frameStart := 157872 },
  { event := event157881
    frameStart := 157872 },
  { event := event157882
    frameStart := 157872 },
  { event := event157883
    frameStart := 157872 },
  { event := event157884
    frameStart := 157872 },
  { event := event157885
    frameStart := 157872 },
  { event := event157886
    frameStart := 157872 },
  { event := event157887
    frameStart := 157872 }
]

def eventLeaf9868 : Array AnnotatedEvent := #[
  { event := event157888
    frameStart := 157872 },
  { event := event157889
    frameStart := 157872 },
  { event := event157890
    frameStart := 157872 },
  { event := event157891
    frameStart := 157872 },
  { event := event157892
    frameStart := 157872 },
  { event := event157893
    frameStart := 157872 },
  { event := event157894
    frameStart := 157872 },
  { event := event157895
    frameStart := 157872 },
  { event := event157896
    frameStart := 157872 },
  { event := event157897
    frameStart := 157872 },
  { event := event157898
    frameStart := 157872 },
  { event := event157899
    frameStart := 157872 },
  { event := event157900
    frameStart := 157872 },
  { event := event157901
    frameStart := 157872 },
  { event := event157902
    frameStart := 157872 },
  { event := event157903
    frameStart := 157872 }
]

def eventLeaf9869 : Array AnnotatedEvent := #[
  { event := event157904
    frameStart := 157872 },
  { event := event157905
    frameStart := 157872 },
  { event := event157906
    frameStart := 157872 },
  { event := event157907
    frameStart := 157872 },
  { event := event157908
    frameStart := 157872 },
  { event := event157909
    frameStart := 157872 },
  { event := event157910
    frameStart := 157872 },
  { event := event157911
    frameStart := 157872 },
  { event := event157912
    frameStart := 157872 },
  { event := event157913
    frameStart := 157872 },
  { event := event157914
    frameStart := 157872 },
  { event := event157915
    frameStart := 157872 },
  { event := event157916
    frameStart := 157872 },
  { event := event157917
    frameStart := 157872 },
  { event := event157918
    frameStart := 157872 },
  { event := event157919
    frameStart := 157872 }
]

def eventLeaf9870 : Array AnnotatedEvent := #[
  { event := event157920
    frameStart := 157872 },
  { event := event157921
    frameStart := 157872 },
  { event := event157922
    frameStart := 157872 },
  { event := event157923
    frameStart := 157872 },
  { event := event157924
    frameStart := 157872 },
  { event := event157925
    frameStart := 157872 },
  { event := event157926
    frameStart := 157872 },
  { event := event157927
    frameStart := 157872 },
  { event := event157928
    frameStart := 157872 },
  { event := event157929
    frameStart := 157872 },
  { event := event157930
    frameStart := 157872 },
  { event := event157931
    frameStart := 157872 },
  { event := event157932
    frameStart := 157872 },
  { event := event157933
    frameStart := 157872 },
  { event := event157934
    frameStart := 157872 },
  { event := event157935
    frameStart := 157872 }
]

def eventLeaf9871 : Array AnnotatedEvent := #[
  { event := event157936
    frameStart := 157872 },
  { event := event157937
    frameStart := 157872 },
  { event := event157938
    frameStart := 157872 },
  { event := event157939
    frameStart := 157872 },
  { event := event157940
    frameStart := 157872 },
  { event := event157941
    frameStart := 157872 },
  { event := event157942
    frameStart := 157872 },
  { event := event157943
    frameStart := 157872 },
  { event := event157944
    frameStart := 157872 },
  { event := event157945
    frameStart := 157872 },
  { event := event157946
    frameStart := 157872 },
  { event := event157947
    frameStart := 157872 },
  { event := event157948
    frameStart := 157872 },
  { event := event157949
    frameStart := 157872 },
  { event := event157950
    frameStart := 157872 },
  { event := event157951
    frameStart := 157872 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events616
