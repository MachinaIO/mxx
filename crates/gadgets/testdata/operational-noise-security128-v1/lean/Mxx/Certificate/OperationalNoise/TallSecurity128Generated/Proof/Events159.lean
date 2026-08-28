import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events159

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event40704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24155⟩⟩) (.sum [.result 40700 .summary, .result 39731 .summary])

def exact40705RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40705RawTermsValid :
    exact40705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40705 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24155⟩⟩) exact40705RawTerms .large 40703 (.finite 96566716313119651734393211060224) (some (40704))

def event40706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34175⟩⟩) 0 ⟨24155⟩ 40705

def event40707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34175⟩⟩) 1 ⟨34174⟩ 39249

def event40708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34175⟩⟩) (.sum [.predecessor 0 40706 .coefficient, .predecessor 1 40707 .coefficient])

def event40709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34175⟩⟩) (.sum [.result 40705 .summary, .result 39249 .summary])

def exact40710RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40710RawTermsValid :
    exact40710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40710 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34175⟩⟩) exact40710RawTerms .large 40708 (.finite 128755916426494733378385616044032) (some (40709))

def event40711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53235⟩⟩) 0 ⟨34175⟩ 40710

def event40712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53235⟩⟩) 1 ⟨53234⟩ 38767

def event40713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53235⟩⟩) (.sum [.predecessor 0 40711 .coefficient, .predecessor 1 40712 .coefficient])

def event40714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53235⟩⟩) (.sum [.result 40710 .summary, .result 38767 .summary])

def exact40715RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40715RawTermsValid :
    exact40715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53235⟩⟩) exact40715RawTerms .large 40713 (.finite 160945509440761189776859800535040) (some (40714))

def event40716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56215⟩⟩) 0 ⟨53235⟩ 40715

def event40717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56215⟩⟩) 1 ⟨56214⟩ 38285

def event40718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56215⟩⟩) (.sum [.predecessor 0 40716 .coefficient, .predecessor 1 40717 .coefficient])

def event40719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56215⟩⟩) (.sum [.result 40715 .summary, .result 38285 .summary])

def exact40720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40720RawTermsValid :
    exact40720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56215⟩⟩) exact40720RawTerms .large 40718 (.finite 193135298905473333552574874779648) (some (40719))

def event40721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59195⟩⟩) 0 ⟨56215⟩ 40720

def event40722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59195⟩⟩) 1 ⟨59194⟩ 37803

def event40723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59195⟩⟩) (.sum [.predecessor 0 40721 .coefficient, .predecessor 1 40722 .coefficient])

def event40724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59195⟩⟩) (.sum [.result 40720 .summary, .result 37803 .summary])

def exact40725RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40725RawTermsValid :
    exact40725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59195⟩⟩) exact40725RawTerms .large 40723 (.finite 225325481271076852082771728531456) (some (40724))

def event40726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62175⟩⟩) 0 ⟨59195⟩ 40725

def event40727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62175⟩⟩) 1 ⟨62174⟩ 37321

def event40728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62175⟩⟩) (.sum [.predecessor 0 40726 .coefficient, .predecessor 1 40727 .coefficient])

def event40729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62175⟩⟩) (.sum [.result 40725 .summary, .result 37321 .summary])

def exact40730RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40730RawTermsValid :
    exact40730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40730 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62175⟩⟩) exact40730RawTerms .large 40728 (.finite 257515860087126057990209472036864) (some (40729))

def event40731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65155⟩⟩) 0 ⟨62175⟩ 40730

def event40732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65155⟩⟩) 1 ⟨65154⟩ 36839

def event40733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65155⟩⟩) (.sum [.predecessor 0 40731 .coefficient, .predecessor 1 40732 .coefficient])

def event40734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65155⟩⟩) (.sum [.result 40730 .summary, .result 36839 .summary])

def exact40735RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40735RawTermsValid :
    exact40735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40735 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65155⟩⟩) exact40735RawTerms .large 40733 (.finite 289706631804066638652128995049472) (some (40734))

def event40736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70892⟩⟩) 0 ⟨65155⟩ 40735

def event40737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70892⟩⟩) 1 ⟨70891⟩ 36357

def event40738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70892⟩⟩) (.sum [.predecessor 0 40736 .coefficient, .predecessor 1 40737 .coefficient])

def event40739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70892⟩⟩) (.sum [.result 40735 .summary, .result 36357 .summary])

def exact40740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40740RawTermsValid :
    exact40740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70892⟩⟩) exact40740RawTerms .large 40738 (.finite 321897992872344281445771187322880) (some (40739))

def event40741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70893⟩⟩) 0 ⟨70892⟩ 40740

def event40742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70893⟩⟩) 1 ⟨28517⟩ 35875

def event40743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70893⟩⟩) (.sum [.predecessor 0 40741 .coefficient, .predecessor 1 40742 .coefficient])

def event40744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70893⟩⟩) (.sum [.result 40740 .summary, .result 35875 .summary])

def exact40745RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40745RawTermsValid :
    exact40745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70893⟩⟩) exact40745RawTerms .large 40743 (.finite 354089550391067611616654269349888) (some (40744))

def event40746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70894⟩⟩) 0 ⟨70893⟩ 40745

def event40747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70894⟩⟩) 1 ⟨31197⟩ 35393

def event40748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70894⟩⟩) (.sum [.predecessor 0 40746 .coefficient, .predecessor 1 40747 .coefficient])

def event40749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70894⟩⟩) (.sum [.result 40745 .summary, .result 35393 .summary])

def exact40750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40750RawTermsValid :
    exact40750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70894⟩⟩) exact40750RawTerms .large 40748 (.finite 386281697261128003919260020637696) (some (40749))

def event40751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70895⟩⟩) 0 ⟨70894⟩ 40750

def event40752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70895⟩⟩) 1 ⟨36857⟩ 34911

def event40753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70895⟩⟩) (.sum [.predecessor 0 40751 .coefficient, .predecessor 1 40752 .coefficient])

def event40754 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70895⟩⟩) (.sum [.result 40750 .summary, .result 34911 .summary])

def exact40755RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40755RawTermsValid :
    exact40755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40755 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70895⟩⟩) exact40755RawTerms .large 40753 (.finite 418474237032079770976347551432704) (some (40754))

def event40756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70896⟩⟩) 0 ⟨70895⟩ 40755

def event40757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70896⟩⟩) 1 ⟨39537⟩ 34429

def event40758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70896⟩⟩) (.sum [.predecessor 0 40756 .coefficient, .predecessor 1 40757 .coefficient])

def event40759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70896⟩⟩) (.sum [.result 40755 .summary, .result 34429 .summary])

def exact40760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40760RawTermsValid :
    exact40760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70896⟩⟩) exact40760RawTerms .large 40758 (.finite 450666973253477225410675971981312) (some (40759))

def event40761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70897⟩⟩) 0 ⟨70896⟩ 40760

def event40762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70897⟩⟩) 1 ⟨42217⟩ 33947

def event40763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70897⟩⟩) (.sum [.predecessor 0 40761 .coefficient, .predecessor 1 40762 .coefficient])

def event40764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70897⟩⟩) (.sum [.result 40760 .summary, .result 33947 .summary])

def exact40765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40765RawTermsValid :
    exact40765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70897⟩⟩) exact40765RawTerms .large 40763 (.finite 482860102375766054599486172037120) (some (40764))

def event40766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70898⟩⟩) 0 ⟨70897⟩ 40765

def event40767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70898⟩⟩) 1 ⟨44897⟩ 33465

def event40768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70898⟩⟩) (.sum [.predecessor 0 40766 .coefficient, .predecessor 1 40767 .coefficient])

def event40769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70898⟩⟩) (.sum [.result 40765 .summary, .result 33465 .summary])

def exact40770RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40770RawTermsValid :
    exact40770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70898⟩⟩) exact40770RawTerms .large 40768 (.finite 515053820849391945920019041353728) (some (40769))

def event40771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70899⟩⟩) 0 ⟨70898⟩ 40770

def event40772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70899⟩⟩) 1 ⟨47577⟩ 32983

def event40773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70899⟩⟩) (.sum [.predecessor 0 40771 .coefficient, .predecessor 1 40772 .coefficient])

def event40774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70899⟩⟩) (.sum [.result 40770 .summary, .result 32983 .summary])

def exact40775RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40775RawTermsValid :
    exact40775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70899⟩⟩) exact40775RawTerms .large 40773 (.finite 547248128674354899372274579931136) (some (40774))

def event40776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70900⟩⟩) 0 ⟨70899⟩ 40775

def event40777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70900⟩⟩) 1 ⟨50257⟩ 32501

def event40778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70900⟩⟩) (.sum [.predecessor 0 40776 .coefficient, .predecessor 1 40777 .coefficient])

def event40779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70900⟩⟩) (.sum [.result 40775 .summary, .result 32501 .summary])

def exact40780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact40780RawTermsValid :
    exact40780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70900⟩⟩) exact40780RawTerms .large 40778 (.finite 579442632949763540201771008262144) (some (40779))

def event40781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71536⟩⟩) 0 ⟨70900⟩ 40780

def event40782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71536⟩⟩) 1 ⟨71534⟩ 32003

def event40783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71536⟩⟩) (.product (.predecessor 0 40781 .coefficient) (.predecessor 1 40782 .coefficient) (⟨false, false, none, none, none⟩))

def event40784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71536⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) [⟨.result 32003 .coefficient, false, none⟩])

def event40785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71536⟩⟩) (.product (.result 40780 .summary) (.transfer 40784) (⟨false, false, none, none, none⟩))

def event40786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 17⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 29⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40788 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40788 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 16⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 28⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40792 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40792 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 15⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 27⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40796 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40796 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40798 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 14⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40799 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 26⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40800 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40800 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 13⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 25⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40804 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40805 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40804 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 12⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 24⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40808 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40808 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 11⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 22⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40812 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40813 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40812 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40814 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 10⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40815 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 21⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40816 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40817 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40816 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 9⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40819 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 35⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40820 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40820 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40822 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 8⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40823 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 34⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40824 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40825 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40824 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40826 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 7⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 33⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40828 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40829 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40828 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 6⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 32⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40832 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40832 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40834 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 5⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 31⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40836 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40836 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 4⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 30⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40840 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40840 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 3⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 23⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40844 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40844 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40846 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 2⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 20⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40848 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40848 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 1⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 19⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40852 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40853 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40852 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def event40854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 0⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩)

def event40855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .operator (⟨40780, 18⟩, ⟨32003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (-1)⟩)

def event40856 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000)

def event40857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71536⟩⟩, .relation 40856 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩)

def exact40858RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26736⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29416⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨60272⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨67231⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩, (-1)⟩]

theorem exact40858RawTermsValid :
    exact40858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71536⟩⟩) exact40858RawTerms .large 40783 (.finite 6221717896068416040249469304417135687106560) (some (40785))

def event40859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68460⟩⟩) 0 ⟨67241⟩ 1324

def event40860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68460⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact40861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68460⟩⟩]⟩, (1)⟩]

theorem exact40861RawTermsValid :
    exact40861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68460⟩⟩) exact40861RawTerms (.finite 5647228698) 40860 .exactZero (none)

def event40862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68462⟩⟩) 0 ⟨68460⟩ 40861

def event40863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68462⟩⟩) 1 ⟨2370⟩ 4

def event40864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68462⟩⟩) (.scale (.predecessor 0 40862 .coefficient) (.value (.predecessor 1 40863 .coefficient)))

def exact40865RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68460⟩⟩]⟩, (1)⟩]

theorem exact40865RawTermsValid :
    exact40865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68462⟩⟩) exact40865RawTerms (.finite 5647228698) 40864 .exactZero (none)

def event40866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68463⟩⟩) 0 ⟨11643⟩ 32120

def event40867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68463⟩⟩) 1 ⟨68462⟩ 40865

def event40868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68463⟩⟩) (.product (.predecessor 0 40866 .coefficient) (.predecessor 1 40867 .coefficient) (⟨false, false, none, none, none⟩))

def event40869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68463⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68460⟩⟩]⟩) [⟨.result 40861 .coefficient, false, none⟩])

def event40870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68463⟩⟩) (.product (.result 32120 .summary) (.transfer 40869) (⟨false, false, none, none, none⟩))

def event40871 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68463⟩⟩, .operator (⟨32120, 0⟩, ⟨40865, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68460⟩⟩]⟩, (1)⟩)

def event40872 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68461⟩⟩)

def event40873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event40874 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event40875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event40876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event40877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event40878 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event40879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event40880 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event40881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 40880

def event40882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 40878

def event40883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 40881 .coefficient) (.value (.predecessor 1 40882 .coefficient)))

def event40884 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event40885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 40884

def event40886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 40876

def event40887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 40885 .coefficient, .predecessor 1 40886 .coefficient])

def event40888 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event40889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 40888

def event40890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 40874

def event40891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 40890 .coefficient))

def event40892 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event40893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48050⟩⟩) 0 ⟨11600⟩ 40892

def event40894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48050⟩⟩) (.authority (.programFamilyFact))

def exact40895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48050⟩⟩], []⟩, (1)⟩]

theorem exact40895RawTermsValid :
    exact40895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48050⟩⟩) exact40895RawTerms (.finite 60) 40894 .exactZero (none)

def event40896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15216⟩⟩) 0 ⟨11600⟩ 40892

def event40897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15216⟩⟩) (.authority (.programFamilyFact))

def exact40898RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15216⟩⟩], []⟩, (1)⟩]

theorem exact40898RawTermsValid :
    exact40898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15216⟩⟩) exact40898RawTerms (.finite 60) 40897 .exactZero (none)

def event40899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48051⟩⟩) 0 ⟨15216⟩ 40898

def event40900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48051⟩⟩) 1 ⟨48050⟩ 40895

def event40901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48051⟩⟩) (.product (.predecessor 0 40899 .coefficient) (.predecessor 1 40900 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event40902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48051⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], []⟩) [⟨.result 40898 .coefficient, true, some 1⟩, ⟨.result 40895 .coefficient, true, some 1⟩])

def event40903 : Event := .survivorFold (1) 40902

def exact40904RawTerms : List Term := []

theorem exact40904RawTermsValid :
    exact40904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48051⟩⟩) exact40904RawTerms (.finite 3600) 40901 (.finite 3600) (some (40902))

def event40905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48052⟩⟩) 0 ⟨48051⟩ 40904

def event40906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48052⟩⟩) (.identity (.predecessor 0 40905 .coefficient))

def event40907 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48052⟩⟩) (.finite 3600)

def event40908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48220⟩⟩) 0 ⟨48052⟩ 40907

def event40909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48220⟩⟩) (.authority (.programFamilyFact))

def exact40910RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48220⟩⟩], []⟩, (1)⟩]

theorem exact40910RawTermsValid :
    exact40910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48220⟩⟩) exact40910RawTerms (.finite 60) 40909 .exactZero (none)

def event40911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48221⟩⟩) 0 ⟨48220⟩ 40910

def event40912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48221⟩⟩) (.identity (.predecessor 0 40911 .coefficient))

def event40913 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48221⟩⟩) (.finite 60)

def event40914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48480⟩⟩) 0 ⟨48221⟩ 40913

def event40915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48480⟩⟩) (.authority (.programFamilyFact))

def exact40916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48480⟩⟩], []⟩, (1)⟩]

theorem exact40916RawTermsValid :
    exact40916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48480⟩⟩) exact40916RawTerms (.finite 63) 40915 .exactZero (none)

def event40917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45370⟩⟩) 0 ⟨11600⟩ 40892

def event40918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45370⟩⟩) (.authority (.programFamilyFact))

def exact40919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45370⟩⟩], []⟩, (1)⟩]

theorem exact40919RawTermsValid :
    exact40919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45370⟩⟩) exact40919RawTerms (.finite 58) 40918 .exactZero (none)

def event40920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14916⟩⟩) 0 ⟨11600⟩ 40892

def event40921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14916⟩⟩) (.authority (.programFamilyFact))

def exact40922RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14916⟩⟩], []⟩, (1)⟩]

theorem exact40922RawTermsValid :
    exact40922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14916⟩⟩) exact40922RawTerms (.finite 58) 40921 .exactZero (none)

def event40923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45371⟩⟩) 0 ⟨14916⟩ 40922

def event40924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45371⟩⟩) 1 ⟨45370⟩ 40919

def event40925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45371⟩⟩) (.product (.predecessor 0 40923 .coefficient) (.predecessor 1 40924 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event40926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45371⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], []⟩) [⟨.result 40922 .coefficient, true, some 1⟩, ⟨.result 40919 .coefficient, true, some 1⟩])

def event40927 : Event := .survivorFold (1) 40926

def exact40928RawTerms : List Term := []

theorem exact40928RawTermsValid :
    exact40928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40928 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45371⟩⟩) exact40928RawTerms (.finite 3364) 40925 (.finite 3364) (some (40926))

def event40929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45372⟩⟩) 0 ⟨45371⟩ 40928

def event40930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45372⟩⟩) (.identity (.predecessor 0 40929 .coefficient))

def event40931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45372⟩⟩) (.finite 3364)

def event40932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45540⟩⟩) 0 ⟨45372⟩ 40931

def event40933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45540⟩⟩) (.authority (.programFamilyFact))

def exact40934RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45540⟩⟩], []⟩, (1)⟩]

theorem exact40934RawTermsValid :
    exact40934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45540⟩⟩) exact40934RawTerms (.finite 58) 40933 .exactZero (none)

def event40935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45541⟩⟩) 0 ⟨45540⟩ 40934

def event40936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45541⟩⟩) (.identity (.predecessor 0 40935 .coefficient))

def event40937 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45541⟩⟩) (.finite 58)

def event40938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45800⟩⟩) 0 ⟨45541⟩ 40937

def event40939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45800⟩⟩) (.authority (.programFamilyFact))

def exact40940RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45800⟩⟩], []⟩, (1)⟩]

theorem exact40940RawTermsValid :
    exact40940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45800⟩⟩) exact40940RawTerms (.finite 63) 40939 .exactZero (none)

def event40941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42690⟩⟩) 0 ⟨11600⟩ 40892

def event40942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42690⟩⟩) (.authority (.programFamilyFact))

def exact40943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42690⟩⟩], []⟩, (1)⟩]

theorem exact40943RawTermsValid :
    exact40943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42690⟩⟩) exact40943RawTerms (.finite 52) 40942 .exactZero (none)

def event40944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14616⟩⟩) 0 ⟨11600⟩ 40892

def event40945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14616⟩⟩) (.authority (.programFamilyFact))

def exact40946RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14616⟩⟩], []⟩, (1)⟩]

theorem exact40946RawTermsValid :
    exact40946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40946 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14616⟩⟩) exact40946RawTerms (.finite 52) 40945 .exactZero (none)

def event40947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42691⟩⟩) 0 ⟨14616⟩ 40946

def event40948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42691⟩⟩) 1 ⟨42690⟩ 40943

def event40949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42691⟩⟩) (.product (.predecessor 0 40947 .coefficient) (.predecessor 1 40948 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event40950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42691⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14616⟩⟩, ⟨.program ⟨257⟩, ⟨42690⟩⟩], []⟩) [⟨.result 40946 .coefficient, true, some 1⟩, ⟨.result 40943 .coefficient, true, some 1⟩])

def event40951 : Event := .survivorFold (1) 40950

def exact40952RawTerms : List Term := []

theorem exact40952RawTermsValid :
    exact40952RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40952 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42691⟩⟩) exact40952RawTerms (.finite 2704) 40949 (.finite 2704) (some (40950))

def event40953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42692⟩⟩) 0 ⟨42691⟩ 40952

def event40954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42692⟩⟩) (.identity (.predecessor 0 40953 .coefficient))

def event40955 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42692⟩⟩) (.finite 2704)

def event40956 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42860⟩⟩) 0 ⟨42692⟩ 40955

def event40957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42860⟩⟩) (.authority (.programFamilyFact))

def exact40958RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], []⟩, (1)⟩]

theorem exact40958RawTermsValid :
    exact40958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event40958 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42860⟩⟩) exact40958RawTerms (.finite 52) 40957 .exactZero (none)

def event40959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42861⟩⟩) 0 ⟨42860⟩ 40958

def eventLeaf2544 : Array AnnotatedEvent := #[
  { event := event40704
    frameStart := 0 },
  { event := event40705
    frameStart := 0 },
  { event := event40706
    frameStart := 0 },
  { event := event40707
    frameStart := 0 },
  { event := event40708
    frameStart := 0 },
  { event := event40709
    frameStart := 0 },
  { event := event40710
    frameStart := 0 },
  { event := event40711
    frameStart := 0 },
  { event := event40712
    frameStart := 0 },
  { event := event40713
    frameStart := 0 },
  { event := event40714
    frameStart := 0 },
  { event := event40715
    frameStart := 0 },
  { event := event40716
    frameStart := 0 },
  { event := event40717
    frameStart := 0 },
  { event := event40718
    frameStart := 0 },
  { event := event40719
    frameStart := 0 }
]

def eventLeaf2545 : Array AnnotatedEvent := #[
  { event := event40720
    frameStart := 0 },
  { event := event40721
    frameStart := 0 },
  { event := event40722
    frameStart := 0 },
  { event := event40723
    frameStart := 0 },
  { event := event40724
    frameStart := 0 },
  { event := event40725
    frameStart := 0 },
  { event := event40726
    frameStart := 0 },
  { event := event40727
    frameStart := 0 },
  { event := event40728
    frameStart := 0 },
  { event := event40729
    frameStart := 0 },
  { event := event40730
    frameStart := 0 },
  { event := event40731
    frameStart := 0 },
  { event := event40732
    frameStart := 0 },
  { event := event40733
    frameStart := 0 },
  { event := event40734
    frameStart := 0 },
  { event := event40735
    frameStart := 0 }
]

def eventLeaf2546 : Array AnnotatedEvent := #[
  { event := event40736
    frameStart := 0 },
  { event := event40737
    frameStart := 0 },
  { event := event40738
    frameStart := 0 },
  { event := event40739
    frameStart := 0 },
  { event := event40740
    frameStart := 0 },
  { event := event40741
    frameStart := 0 },
  { event := event40742
    frameStart := 0 },
  { event := event40743
    frameStart := 0 },
  { event := event40744
    frameStart := 0 },
  { event := event40745
    frameStart := 0 },
  { event := event40746
    frameStart := 0 },
  { event := event40747
    frameStart := 0 },
  { event := event40748
    frameStart := 0 },
  { event := event40749
    frameStart := 0 },
  { event := event40750
    frameStart := 0 },
  { event := event40751
    frameStart := 0 }
]

def eventLeaf2547 : Array AnnotatedEvent := #[
  { event := event40752
    frameStart := 0 },
  { event := event40753
    frameStart := 0 },
  { event := event40754
    frameStart := 0 },
  { event := event40755
    frameStart := 0 },
  { event := event40756
    frameStart := 0 },
  { event := event40757
    frameStart := 0 },
  { event := event40758
    frameStart := 0 },
  { event := event40759
    frameStart := 0 },
  { event := event40760
    frameStart := 0 },
  { event := event40761
    frameStart := 0 },
  { event := event40762
    frameStart := 0 },
  { event := event40763
    frameStart := 0 },
  { event := event40764
    frameStart := 0 },
  { event := event40765
    frameStart := 0 },
  { event := event40766
    frameStart := 0 },
  { event := event40767
    frameStart := 0 }
]

def eventLeaf2548 : Array AnnotatedEvent := #[
  { event := event40768
    frameStart := 0 },
  { event := event40769
    frameStart := 0 },
  { event := event40770
    frameStart := 0 },
  { event := event40771
    frameStart := 0 },
  { event := event40772
    frameStart := 0 },
  { event := event40773
    frameStart := 0 },
  { event := event40774
    frameStart := 0 },
  { event := event40775
    frameStart := 0 },
  { event := event40776
    frameStart := 0 },
  { event := event40777
    frameStart := 0 },
  { event := event40778
    frameStart := 0 },
  { event := event40779
    frameStart := 0 },
  { event := event40780
    frameStart := 0 },
  { event := event40781
    frameStart := 0 },
  { event := event40782
    frameStart := 0 },
  { event := event40783
    frameStart := 0 }
]

def eventLeaf2549 : Array AnnotatedEvent := #[
  { event := event40784
    frameStart := 0 },
  { event := event40785
    frameStart := 0 },
  { event := event40786
    frameStart := 0 },
  { event := event40787
    frameStart := 0 },
  { event := event40788
    frameStart := 0 },
  { event := event40789
    frameStart := 0 },
  { event := event40790
    frameStart := 0 },
  { event := event40791
    frameStart := 0 },
  { event := event40792
    frameStart := 0 },
  { event := event40793
    frameStart := 0 },
  { event := event40794
    frameStart := 0 },
  { event := event40795
    frameStart := 0 },
  { event := event40796
    frameStart := 0 },
  { event := event40797
    frameStart := 0 },
  { event := event40798
    frameStart := 0 },
  { event := event40799
    frameStart := 0 }
]

def eventLeaf2550 : Array AnnotatedEvent := #[
  { event := event40800
    frameStart := 0 },
  { event := event40801
    frameStart := 0 },
  { event := event40802
    frameStart := 0 },
  { event := event40803
    frameStart := 0 },
  { event := event40804
    frameStart := 0 },
  { event := event40805
    frameStart := 0 },
  { event := event40806
    frameStart := 0 },
  { event := event40807
    frameStart := 0 },
  { event := event40808
    frameStart := 0 },
  { event := event40809
    frameStart := 0 },
  { event := event40810
    frameStart := 0 },
  { event := event40811
    frameStart := 0 },
  { event := event40812
    frameStart := 0 },
  { event := event40813
    frameStart := 0 },
  { event := event40814
    frameStart := 0 },
  { event := event40815
    frameStart := 0 }
]

def eventLeaf2551 : Array AnnotatedEvent := #[
  { event := event40816
    frameStart := 0 },
  { event := event40817
    frameStart := 0 },
  { event := event40818
    frameStart := 0 },
  { event := event40819
    frameStart := 0 },
  { event := event40820
    frameStart := 0 },
  { event := event40821
    frameStart := 0 },
  { event := event40822
    frameStart := 0 },
  { event := event40823
    frameStart := 0 },
  { event := event40824
    frameStart := 0 },
  { event := event40825
    frameStart := 0 },
  { event := event40826
    frameStart := 0 },
  { event := event40827
    frameStart := 0 },
  { event := event40828
    frameStart := 0 },
  { event := event40829
    frameStart := 0 },
  { event := event40830
    frameStart := 0 },
  { event := event40831
    frameStart := 0 }
]

def eventLeaf2552 : Array AnnotatedEvent := #[
  { event := event40832
    frameStart := 0 },
  { event := event40833
    frameStart := 0 },
  { event := event40834
    frameStart := 0 },
  { event := event40835
    frameStart := 0 },
  { event := event40836
    frameStart := 0 },
  { event := event40837
    frameStart := 0 },
  { event := event40838
    frameStart := 0 },
  { event := event40839
    frameStart := 0 },
  { event := event40840
    frameStart := 0 },
  { event := event40841
    frameStart := 0 },
  { event := event40842
    frameStart := 0 },
  { event := event40843
    frameStart := 0 },
  { event := event40844
    frameStart := 0 },
  { event := event40845
    frameStart := 0 },
  { event := event40846
    frameStart := 0 },
  { event := event40847
    frameStart := 0 }
]

def eventLeaf2553 : Array AnnotatedEvent := #[
  { event := event40848
    frameStart := 0 },
  { event := event40849
    frameStart := 0 },
  { event := event40850
    frameStart := 0 },
  { event := event40851
    frameStart := 0 },
  { event := event40852
    frameStart := 0 },
  { event := event40853
    frameStart := 0 },
  { event := event40854
    frameStart := 0 },
  { event := event40855
    frameStart := 0 },
  { event := event40856
    frameStart := 0 },
  { event := event40857
    frameStart := 0 },
  { event := event40858
    frameStart := 0 },
  { event := event40859
    frameStart := 0 },
  { event := event40860
    frameStart := 0 },
  { event := event40861
    frameStart := 0 },
  { event := event40862
    frameStart := 0 },
  { event := event40863
    frameStart := 0 }
]

def eventLeaf2554 : Array AnnotatedEvent := #[
  { event := event40864
    frameStart := 0 },
  { event := event40865
    frameStart := 0 },
  { event := event40866
    frameStart := 0 },
  { event := event40867
    frameStart := 0 },
  { event := event40868
    frameStart := 0 },
  { event := event40869
    frameStart := 0 },
  { event := event40870
    frameStart := 0 },
  { event := event40871
    frameStart := 0 },
  { event := event40872
    frameStart := 40872 },
  { event := event40873
    frameStart := 40872 },
  { event := event40874
    frameStart := 40872 },
  { event := event40875
    frameStart := 40872 },
  { event := event40876
    frameStart := 40872 },
  { event := event40877
    frameStart := 40872 },
  { event := event40878
    frameStart := 40872 },
  { event := event40879
    frameStart := 40872 }
]

def eventLeaf2555 : Array AnnotatedEvent := #[
  { event := event40880
    frameStart := 40872 },
  { event := event40881
    frameStart := 40872 },
  { event := event40882
    frameStart := 40872 },
  { event := event40883
    frameStart := 40872 },
  { event := event40884
    frameStart := 40872 },
  { event := event40885
    frameStart := 40872 },
  { event := event40886
    frameStart := 40872 },
  { event := event40887
    frameStart := 40872 },
  { event := event40888
    frameStart := 40872 },
  { event := event40889
    frameStart := 40872 },
  { event := event40890
    frameStart := 40872 },
  { event := event40891
    frameStart := 40872 },
  { event := event40892
    frameStart := 40872 },
  { event := event40893
    frameStart := 40872 },
  { event := event40894
    frameStart := 40872 },
  { event := event40895
    frameStart := 40872 }
]

def eventLeaf2556 : Array AnnotatedEvent := #[
  { event := event40896
    frameStart := 40872 },
  { event := event40897
    frameStart := 40872 },
  { event := event40898
    frameStart := 40872 },
  { event := event40899
    frameStart := 40872 },
  { event := event40900
    frameStart := 40872 },
  { event := event40901
    frameStart := 40872 },
  { event := event40902
    frameStart := 40872 },
  { event := event40903
    frameStart := 40872 },
  { event := event40904
    frameStart := 40872 },
  { event := event40905
    frameStart := 40872 },
  { event := event40906
    frameStart := 40872 },
  { event := event40907
    frameStart := 40872 },
  { event := event40908
    frameStart := 40872 },
  { event := event40909
    frameStart := 40872 },
  { event := event40910
    frameStart := 40872 },
  { event := event40911
    frameStart := 40872 }
]

def eventLeaf2557 : Array AnnotatedEvent := #[
  { event := event40912
    frameStart := 40872 },
  { event := event40913
    frameStart := 40872 },
  { event := event40914
    frameStart := 40872 },
  { event := event40915
    frameStart := 40872 },
  { event := event40916
    frameStart := 40872 },
  { event := event40917
    frameStart := 40872 },
  { event := event40918
    frameStart := 40872 },
  { event := event40919
    frameStart := 40872 },
  { event := event40920
    frameStart := 40872 },
  { event := event40921
    frameStart := 40872 },
  { event := event40922
    frameStart := 40872 },
  { event := event40923
    frameStart := 40872 },
  { event := event40924
    frameStart := 40872 },
  { event := event40925
    frameStart := 40872 },
  { event := event40926
    frameStart := 40872 },
  { event := event40927
    frameStart := 40872 }
]

def eventLeaf2558 : Array AnnotatedEvent := #[
  { event := event40928
    frameStart := 40872 },
  { event := event40929
    frameStart := 40872 },
  { event := event40930
    frameStart := 40872 },
  { event := event40931
    frameStart := 40872 },
  { event := event40932
    frameStart := 40872 },
  { event := event40933
    frameStart := 40872 },
  { event := event40934
    frameStart := 40872 },
  { event := event40935
    frameStart := 40872 },
  { event := event40936
    frameStart := 40872 },
  { event := event40937
    frameStart := 40872 },
  { event := event40938
    frameStart := 40872 },
  { event := event40939
    frameStart := 40872 },
  { event := event40940
    frameStart := 40872 },
  { event := event40941
    frameStart := 40872 },
  { event := event40942
    frameStart := 40872 },
  { event := event40943
    frameStart := 40872 }
]

def eventLeaf2559 : Array AnnotatedEvent := #[
  { event := event40944
    frameStart := 40872 },
  { event := event40945
    frameStart := 40872 },
  { event := event40946
    frameStart := 40872 },
  { event := event40947
    frameStart := 40872 },
  { event := event40948
    frameStart := 40872 },
  { event := event40949
    frameStart := 40872 },
  { event := event40950
    frameStart := 40872 },
  { event := event40951
    frameStart := 40872 },
  { event := event40952
    frameStart := 40872 },
  { event := event40953
    frameStart := 40872 },
  { event := event40954
    frameStart := 40872 },
  { event := event40955
    frameStart := 40872 },
  { event := event40956
    frameStart := 40872 },
  { event := event40957
    frameStart := 40872 },
  { event := event40958
    frameStart := 40872 },
  { event := event40959
    frameStart := 40872 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events159
