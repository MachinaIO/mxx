import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events559

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event143104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61679⟩⟩) (.sum [.result 143100 .summary, .result 139696 .summary])

def exact143105RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143105RawTermsValid :
    exact143105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143105 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61679⟩⟩) exact143105RawTerms .large 143103 (.finite 257515860087126057990209472036864) (some (143104))

def event143106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64659⟩⟩) 0 ⟨61679⟩ 143105

def event143107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64659⟩⟩) 1 ⟨64658⟩ 139214

def event143108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64659⟩⟩) (.sum [.predecessor 0 143106 .coefficient, .predecessor 1 143107 .coefficient])

def event143109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64659⟩⟩) (.sum [.result 143105 .summary, .result 139214 .summary])

def exact143110RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143110RawTermsValid :
    exact143110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143110 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64659⟩⟩) exact143110RawTerms .large 143108 (.finite 289706631804066638652128995049472) (some (143109))

def event143111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69628⟩⟩) 0 ⟨64659⟩ 143110

def event143112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69628⟩⟩) 1 ⟨69627⟩ 138732

def event143113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69628⟩⟩) (.sum [.predecessor 0 143111 .coefficient, .predecessor 1 143112 .coefficient])

def event143114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69628⟩⟩) (.sum [.result 143110 .summary, .result 138732 .summary])

def exact143115RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143115RawTermsValid :
    exact143115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69628⟩⟩) exact143115RawTerms .large 143113 (.finite 321897992872344281445771187322880) (some (143114))

def event143116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69629⟩⟩) 0 ⟨69628⟩ 143115

def event143117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69629⟩⟩) 1 ⟨28117⟩ 138250

def event143118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69629⟩⟩) (.sum [.predecessor 0 143116 .coefficient, .predecessor 1 143117 .coefficient])

def event143119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69629⟩⟩) (.sum [.result 143115 .summary, .result 138250 .summary])

def exact143120RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143120RawTermsValid :
    exact143120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143120 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69629⟩⟩) exact143120RawTerms .large 143118 (.finite 354089550391067611616654269349888) (some (143119))

def event143121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69630⟩⟩) 0 ⟨69629⟩ 143120

def event143122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69630⟩⟩) 1 ⟨30797⟩ 137768

def event143123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69630⟩⟩) (.sum [.predecessor 0 143121 .coefficient, .predecessor 1 143122 .coefficient])

def event143124 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69630⟩⟩) (.sum [.result 143120 .summary, .result 137768 .summary])

def exact143125RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143125RawTermsValid :
    exact143125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143125 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69630⟩⟩) exact143125RawTerms .large 143123 (.finite 386281697261128003919260020637696) (some (143124))

def event143126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69631⟩⟩) 0 ⟨69630⟩ 143125

def event143127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69631⟩⟩) 1 ⟨36457⟩ 137286

def event143128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69631⟩⟩) (.sum [.predecessor 0 143126 .coefficient, .predecessor 1 143127 .coefficient])

def event143129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69631⟩⟩) (.sum [.result 143125 .summary, .result 137286 .summary])

def exact143130RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143130RawTermsValid :
    exact143130RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143130 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69631⟩⟩) exact143130RawTerms .large 143128 (.finite 418474237032079770976347551432704) (some (143129))

def event143131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69632⟩⟩) 0 ⟨69631⟩ 143130

def event143132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69632⟩⟩) 1 ⟨39137⟩ 136804

def event143133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69632⟩⟩) (.sum [.predecessor 0 143131 .coefficient, .predecessor 1 143132 .coefficient])

def event143134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69632⟩⟩) (.sum [.result 143130 .summary, .result 136804 .summary])

def exact143135RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143135RawTermsValid :
    exact143135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143135 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69632⟩⟩) exact143135RawTerms .large 143133 (.finite 450666973253477225410675971981312) (some (143134))

def event143136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69633⟩⟩) 0 ⟨69632⟩ 143135

def event143137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69633⟩⟩) 1 ⟨41817⟩ 136322

def event143138 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69633⟩⟩) (.sum [.predecessor 0 143136 .coefficient, .predecessor 1 143137 .coefficient])

def event143139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69633⟩⟩) (.sum [.result 143135 .summary, .result 136322 .summary])

def exact143140RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143140RawTermsValid :
    exact143140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69633⟩⟩) exact143140RawTerms .large 143138 (.finite 482860102375766054599486172037120) (some (143139))

def event143141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69634⟩⟩) 0 ⟨69633⟩ 143140

def event143142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69634⟩⟩) 1 ⟨44497⟩ 135840

def event143143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69634⟩⟩) (.sum [.predecessor 0 143141 .coefficient, .predecessor 1 143142 .coefficient])

def event143144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69634⟩⟩) (.sum [.result 143140 .summary, .result 135840 .summary])

def exact143145RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143145RawTermsValid :
    exact143145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69634⟩⟩) exact143145RawTerms .large 143143 (.finite 515053820849391945920019041353728) (some (143144))

def event143146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69635⟩⟩) 0 ⟨69634⟩ 143145

def event143147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69635⟩⟩) 1 ⟨47177⟩ 135358

def event143148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69635⟩⟩) (.sum [.predecessor 0 143146 .coefficient, .predecessor 1 143147 .coefficient])

def event143149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69635⟩⟩) (.sum [.result 143145 .summary, .result 135358 .summary])

def exact143150RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143150RawTermsValid :
    exact143150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143150 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69635⟩⟩) exact143150RawTerms .large 143148 (.finite 547248128674354899372274579931136) (some (143149))

def event143151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69636⟩⟩) 0 ⟨69635⟩ 143150

def event143152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69636⟩⟩) 1 ⟨49857⟩ 134876

def event143153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69636⟩⟩) (.sum [.predecessor 0 143151 .coefficient, .predecessor 1 143152 .coefficient])

def event143154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69636⟩⟩) (.sum [.result 143150 .summary, .result 134876 .summary])

def exact143155RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143155RawTermsValid :
    exact143155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143155 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69636⟩⟩) exact143155RawTerms .large 143153 (.finite 579442632949763540201771008262144) (some (143154))

def event143156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71019⟩⟩) 0 ⟨69636⟩ 143155

def event143157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71019⟩⟩) 1 ⟨71017⟩ 134378

def event143158 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71019⟩⟩) (.product (.predecessor 0 143156 .coefficient) (.predecessor 1 143157 .coefficient) (⟨false, false, none, none, none⟩))

def event143159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71019⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) [⟨.result 134378 .coefficient, false, none⟩])

def event143160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71019⟩⟩) (.product (.result 143155 .summary) (.transfer 143159) (⟨false, false, none, none, none⟩))

def event143161 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 17⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143162 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 29⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143163 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143164 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143163 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143165 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 16⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143166 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 28⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143167 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143168 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143167 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143169 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 15⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143170 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 27⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143171 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143172 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143171 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143173 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 14⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143174 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 26⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143175 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143176 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143175 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143177 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 13⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143178 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 25⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143179 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143180 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143179 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143181 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 12⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143182 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 24⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143183 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143184 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143183 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143185 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 11⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143186 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 22⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143187 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143188 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143187 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143189 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 10⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143190 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 21⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143191 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143192 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143191 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143193 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 9⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143194 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 35⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143195 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143196 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143195 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143197 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 8⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143198 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 34⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143199 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143200 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143199 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143201 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 7⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143202 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 33⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143203 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143204 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143203 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143205 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 6⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143206 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 32⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143207 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143208 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143207 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143209 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 5⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143210 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 31⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143211 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143212 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143211 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143213 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 4⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143214 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 30⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143215 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143216 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143215 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143217 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 3⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143218 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 23⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143219 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143220 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143219 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143221 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 2⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143222 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 20⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143223 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143224 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143223 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143225 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 1⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143226 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 19⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143227 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143228 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143227 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event143229 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 0⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event143230 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .operator (⟨143155, 18⟩, ⟨134378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event143231 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 134375)

def event143232 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71019⟩⟩, .relation 143231 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def exact143233RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩]

theorem exact143233RawTermsValid :
    exact143233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143233 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71019⟩⟩) exact143233RawTerms .large 143158 (.finite 6221717896068416040249469304417135687106560) (some (143160))

def event143234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68300⟩⟩) 0 ⟨66121⟩ 6560

def event143235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68300⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact143236RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩, (1)⟩]

theorem exact143236RawTermsValid :
    exact143236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68300⟩⟩) exact143236RawTerms (.finite 5647228698) 143235 .exactZero (none)

def event143237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68302⟩⟩) 0 ⟨68300⟩ 143236

def event143238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68302⟩⟩) 1 ⟨2370⟩ 4

def event143239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68302⟩⟩) (.scale (.predecessor 0 143237 .coefficient) (.value (.predecessor 1 143238 .coefficient)))

def exact143240RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩, (1)⟩]

theorem exact143240RawTermsValid :
    exact143240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143240 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68302⟩⟩) exact143240RawTerms (.finite 5647228698) 143239 .exactZero (none)

def event143241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68303⟩⟩) 0 ⟨5473⟩ 134495

def event143242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68303⟩⟩) 1 ⟨68302⟩ 143240

def event143243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68303⟩⟩) (.product (.predecessor 0 143241 .coefficient) (.predecessor 1 143242 .coefficient) (⟨false, false, none, none, none⟩))

def event143244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68303⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩) [⟨.result 143236 .coefficient, false, none⟩])

def event143245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68303⟩⟩) (.product (.result 134495 .summary) (.transfer 143244) (⟨false, false, none, none, none⟩))

def event143246 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .operator (⟨134495, 0⟩, ⟨143240, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩, (1)⟩)

def event143247 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68301⟩⟩)

def event143248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event143249 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event143250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event143251 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event143252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event143253 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event143254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event143255 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event143256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 143255

def event143257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 143253

def event143258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 143256 .coefficient) (.value (.predecessor 1 143257 .coefficient)))

def event143259 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event143260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 143259

def event143261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 143251

def event143262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 143260 .coefficient, .predecessor 1 143261 .coefficient])

def event143263 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event143264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 143263

def event143265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 143249

def event143266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 143265 .coefficient))

def event143267 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event143268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47666⟩⟩) 0 ⟨5469⟩ 143267

def event143269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47666⟩⟩) (.authority (.programFamilyFact))

def exact143270RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩]

theorem exact143270RawTermsValid :
    exact143270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47666⟩⟩) exact143270RawTerms (.finite 60) 143269 .exactZero (none)

def event143271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14976⟩⟩) 0 ⟨5469⟩ 143267

def event143272 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14976⟩⟩) (.authority (.programFamilyFact))

def exact143273RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩], []⟩, (1)⟩]

theorem exact143273RawTermsValid :
    exact143273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143273 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14976⟩⟩) exact143273RawTerms (.finite 60) 143272 .exactZero (none)

def event143274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 0 ⟨14976⟩ 143273

def event143275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 1 ⟨47666⟩ 143270

def event143276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47667⟩⟩) (.product (.predecessor 0 143274 .coefficient) (.predecessor 1 143275 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event143277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47667⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩) [⟨.result 143273 .coefficient, true, some 1⟩, ⟨.result 143270 .coefficient, true, some 1⟩])

def event143278 : Event := .survivorFold (1) 143277

def exact143279RawTerms : List Term := []

theorem exact143279RawTermsValid :
    exact143279RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143279 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47667⟩⟩) exact143279RawTerms (.finite 3600) 143276 (.finite 3600) (some (143277))

def event143280 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47668⟩⟩) 0 ⟨47667⟩ 143279

def event143281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.identity (.predecessor 0 143280 .coefficient))

def event143282 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.finite 3600)

def event143283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48092⟩⟩) 0 ⟨47668⟩ 143282

def event143284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48092⟩⟩) (.authority (.programFamilyFact))

def exact143285RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], []⟩, (1)⟩]

theorem exact143285RawTermsValid :
    exact143285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48092⟩⟩) exact143285RawTerms (.finite 60) 143284 .exactZero (none)

def event143286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48093⟩⟩) 0 ⟨48092⟩ 143285

def event143287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48093⟩⟩) (.identity (.predecessor 0 143286 .coefficient))

def event143288 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48093⟩⟩) (.finite 60)

def event143289 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48272⟩⟩) 0 ⟨48093⟩ 143288

def event143290 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48272⟩⟩) (.authority (.programFamilyFact))

def exact143291RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], []⟩, (1)⟩]

theorem exact143291RawTermsValid :
    exact143291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143291 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48272⟩⟩) exact143291RawTerms (.finite 63) 143290 .exactZero (none)

def event143292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44986⟩⟩) 0 ⟨5469⟩ 143267

def event143293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44986⟩⟩) (.authority (.programFamilyFact))

def exact143294RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44986⟩⟩], []⟩, (1)⟩]

theorem exact143294RawTermsValid :
    exact143294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143294 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44986⟩⟩) exact143294RawTerms (.finite 58) 143293 .exactZero (none)

def event143295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14676⟩⟩) 0 ⟨5469⟩ 143267

def event143296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14676⟩⟩) (.authority (.programFamilyFact))

def exact143297RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14676⟩⟩], []⟩, (1)⟩]

theorem exact143297RawTermsValid :
    exact143297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14676⟩⟩) exact143297RawTerms (.finite 58) 143296 .exactZero (none)

def event143298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44987⟩⟩) 0 ⟨14676⟩ 143297

def event143299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44987⟩⟩) 1 ⟨44986⟩ 143294

def event143300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44987⟩⟩) (.product (.predecessor 0 143298 .coefficient) (.predecessor 1 143299 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event143301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44987⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], []⟩) [⟨.result 143297 .coefficient, true, some 1⟩, ⟨.result 143294 .coefficient, true, some 1⟩])

def event143302 : Event := .survivorFold (1) 143301

def exact143303RawTerms : List Term := []

theorem exact143303RawTermsValid :
    exact143303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143303 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44987⟩⟩) exact143303RawTerms (.finite 3364) 143300 (.finite 3364) (some (143301))

def event143304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44988⟩⟩) 0 ⟨44987⟩ 143303

def event143305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44988⟩⟩) (.identity (.predecessor 0 143304 .coefficient))

def event143306 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44988⟩⟩) (.finite 3364)

def event143307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45412⟩⟩) 0 ⟨44988⟩ 143306

def event143308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45412⟩⟩) (.authority (.programFamilyFact))

def exact143309RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45412⟩⟩], []⟩, (1)⟩]

theorem exact143309RawTermsValid :
    exact143309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45412⟩⟩) exact143309RawTerms (.finite 58) 143308 .exactZero (none)

def event143310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45413⟩⟩) 0 ⟨45412⟩ 143309

def event143311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45413⟩⟩) (.identity (.predecessor 0 143310 .coefficient))

def event143312 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45413⟩⟩) (.finite 58)

def event143313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45592⟩⟩) 0 ⟨45413⟩ 143312

def event143314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45592⟩⟩) (.authority (.programFamilyFact))

def exact143315RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], []⟩, (1)⟩]

theorem exact143315RawTermsValid :
    exact143315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143315 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45592⟩⟩) exact143315RawTerms (.finite 63) 143314 .exactZero (none)

def event143316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42306⟩⟩) 0 ⟨5469⟩ 143267

def event143317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42306⟩⟩) (.authority (.programFamilyFact))

def exact143318RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact143318RawTermsValid :
    exact143318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143318 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42306⟩⟩) exact143318RawTerms (.finite 52) 143317 .exactZero (none)

def event143319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14376⟩⟩) 0 ⟨5469⟩ 143267

def event143320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14376⟩⟩) (.authority (.programFamilyFact))

def exact143321RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩], []⟩, (1)⟩]

theorem exact143321RawTermsValid :
    exact143321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14376⟩⟩) exact143321RawTerms (.finite 52) 143320 .exactZero (none)

def event143322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 0 ⟨14376⟩ 143321

def event143323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 1 ⟨42306⟩ 143318

def event143324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.product (.predecessor 0 143322 .coefficient) (.predecessor 1 143323 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event143325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩) [⟨.result 143321 .coefficient, true, some 1⟩, ⟨.result 143318 .coefficient, true, some 1⟩])

def event143326 : Event := .survivorFold (1) 143325

def exact143327RawTerms : List Term := []

theorem exact143327RawTermsValid :
    exact143327RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143327 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42307⟩⟩) exact143327RawTerms (.finite 2704) 143324 (.finite 2704) (some (143325))

def event143328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42308⟩⟩) 0 ⟨42307⟩ 143327

def event143329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.identity (.predecessor 0 143328 .coefficient))

def event143330 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.finite 2704)

def event143331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42732⟩⟩) 0 ⟨42308⟩ 143330

def event143332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42732⟩⟩) (.authority (.programFamilyFact))

def exact143333RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], []⟩, (1)⟩]

theorem exact143333RawTermsValid :
    exact143333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42732⟩⟩) exact143333RawTerms (.finite 52) 143332 .exactZero (none)

def event143334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42733⟩⟩) 0 ⟨42732⟩ 143333

def event143335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.identity (.predecessor 0 143334 .coefficient))

def event143336 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.finite 52)

def event143337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42908⟩⟩) 0 ⟨42733⟩ 143336

def event143338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42908⟩⟩) (.authority (.programFamilyFact))

def exact143339RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], []⟩, (1)⟩]

theorem exact143339RawTermsValid :
    exact143339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143339 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42908⟩⟩) exact143339RawTerms (.finite 63) 143338 .exactZero (none)

def event143340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39626⟩⟩) 0 ⟨5469⟩ 143267

def event143341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39626⟩⟩) (.authority (.programFamilyFact))

def exact143342RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39626⟩⟩], []⟩, (1)⟩]

theorem exact143342RawTermsValid :
    exact143342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39626⟩⟩) exact143342RawTerms (.finite 46) 143341 .exactZero (none)

def event143343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14076⟩⟩) 0 ⟨5469⟩ 143267

def event143344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14076⟩⟩) (.authority (.programFamilyFact))

def exact143345RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩], []⟩, (1)⟩]

theorem exact143345RawTermsValid :
    exact143345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143345 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14076⟩⟩) exact143345RawTerms (.finite 46) 143344 .exactZero (none)

def event143346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39627⟩⟩) 0 ⟨14076⟩ 143345

def event143347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39627⟩⟩) 1 ⟨39626⟩ 143342

def event143348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39627⟩⟩) (.product (.predecessor 0 143346 .coefficient) (.predecessor 1 143347 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event143349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39627⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], []⟩) [⟨.result 143345 .coefficient, true, some 1⟩, ⟨.result 143342 .coefficient, true, some 1⟩])

def event143350 : Event := .survivorFold (1) 143349

def exact143351RawTerms : List Term := []

theorem exact143351RawTermsValid :
    exact143351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39627⟩⟩) exact143351RawTerms (.finite 2116) 143348 (.finite 2116) (some (143349))

def event143352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39628⟩⟩) 0 ⟨39627⟩ 143351

def event143353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39628⟩⟩) (.identity (.predecessor 0 143352 .coefficient))

def event143354 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39628⟩⟩) (.finite 2116)

def event143355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40052⟩⟩) 0 ⟨39628⟩ 143354

def event143356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40052⟩⟩) (.authority (.programFamilyFact))

def exact143357RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40052⟩⟩], []⟩, (1)⟩]

theorem exact143357RawTermsValid :
    exact143357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143357 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40052⟩⟩) exact143357RawTerms (.finite 46) 143356 .exactZero (none)

def event143358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40053⟩⟩) 0 ⟨40052⟩ 143357

def event143359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40053⟩⟩) (.identity (.predecessor 0 143358 .coefficient))

def eventLeaf8944 : Array AnnotatedEvent := #[
  { event := event143104
    frameStart := 0 },
  { event := event143105
    frameStart := 0 },
  { event := event143106
    frameStart := 0 },
  { event := event143107
    frameStart := 0 },
  { event := event143108
    frameStart := 0 },
  { event := event143109
    frameStart := 0 },
  { event := event143110
    frameStart := 0 },
  { event := event143111
    frameStart := 0 },
  { event := event143112
    frameStart := 0 },
  { event := event143113
    frameStart := 0 },
  { event := event143114
    frameStart := 0 },
  { event := event143115
    frameStart := 0 },
  { event := event143116
    frameStart := 0 },
  { event := event143117
    frameStart := 0 },
  { event := event143118
    frameStart := 0 },
  { event := event143119
    frameStart := 0 }
]

def eventLeaf8945 : Array AnnotatedEvent := #[
  { event := event143120
    frameStart := 0 },
  { event := event143121
    frameStart := 0 },
  { event := event143122
    frameStart := 0 },
  { event := event143123
    frameStart := 0 },
  { event := event143124
    frameStart := 0 },
  { event := event143125
    frameStart := 0 },
  { event := event143126
    frameStart := 0 },
  { event := event143127
    frameStart := 0 },
  { event := event143128
    frameStart := 0 },
  { event := event143129
    frameStart := 0 },
  { event := event143130
    frameStart := 0 },
  { event := event143131
    frameStart := 0 },
  { event := event143132
    frameStart := 0 },
  { event := event143133
    frameStart := 0 },
  { event := event143134
    frameStart := 0 },
  { event := event143135
    frameStart := 0 }
]

def eventLeaf8946 : Array AnnotatedEvent := #[
  { event := event143136
    frameStart := 0 },
  { event := event143137
    frameStart := 0 },
  { event := event143138
    frameStart := 0 },
  { event := event143139
    frameStart := 0 },
  { event := event143140
    frameStart := 0 },
  { event := event143141
    frameStart := 0 },
  { event := event143142
    frameStart := 0 },
  { event := event143143
    frameStart := 0 },
  { event := event143144
    frameStart := 0 },
  { event := event143145
    frameStart := 0 },
  { event := event143146
    frameStart := 0 },
  { event := event143147
    frameStart := 0 },
  { event := event143148
    frameStart := 0 },
  { event := event143149
    frameStart := 0 },
  { event := event143150
    frameStart := 0 },
  { event := event143151
    frameStart := 0 }
]

def eventLeaf8947 : Array AnnotatedEvent := #[
  { event := event143152
    frameStart := 0 },
  { event := event143153
    frameStart := 0 },
  { event := event143154
    frameStart := 0 },
  { event := event143155
    frameStart := 0 },
  { event := event143156
    frameStart := 0 },
  { event := event143157
    frameStart := 0 },
  { event := event143158
    frameStart := 0 },
  { event := event143159
    frameStart := 0 },
  { event := event143160
    frameStart := 0 },
  { event := event143161
    frameStart := 0 },
  { event := event143162
    frameStart := 0 },
  { event := event143163
    frameStart := 0 },
  { event := event143164
    frameStart := 0 },
  { event := event143165
    frameStart := 0 },
  { event := event143166
    frameStart := 0 },
  { event := event143167
    frameStart := 0 }
]

def eventLeaf8948 : Array AnnotatedEvent := #[
  { event := event143168
    frameStart := 0 },
  { event := event143169
    frameStart := 0 },
  { event := event143170
    frameStart := 0 },
  { event := event143171
    frameStart := 0 },
  { event := event143172
    frameStart := 0 },
  { event := event143173
    frameStart := 0 },
  { event := event143174
    frameStart := 0 },
  { event := event143175
    frameStart := 0 },
  { event := event143176
    frameStart := 0 },
  { event := event143177
    frameStart := 0 },
  { event := event143178
    frameStart := 0 },
  { event := event143179
    frameStart := 0 },
  { event := event143180
    frameStart := 0 },
  { event := event143181
    frameStart := 0 },
  { event := event143182
    frameStart := 0 },
  { event := event143183
    frameStart := 0 }
]

def eventLeaf8949 : Array AnnotatedEvent := #[
  { event := event143184
    frameStart := 0 },
  { event := event143185
    frameStart := 0 },
  { event := event143186
    frameStart := 0 },
  { event := event143187
    frameStart := 0 },
  { event := event143188
    frameStart := 0 },
  { event := event143189
    frameStart := 0 },
  { event := event143190
    frameStart := 0 },
  { event := event143191
    frameStart := 0 },
  { event := event143192
    frameStart := 0 },
  { event := event143193
    frameStart := 0 },
  { event := event143194
    frameStart := 0 },
  { event := event143195
    frameStart := 0 },
  { event := event143196
    frameStart := 0 },
  { event := event143197
    frameStart := 0 },
  { event := event143198
    frameStart := 0 },
  { event := event143199
    frameStart := 0 }
]

def eventLeaf8950 : Array AnnotatedEvent := #[
  { event := event143200
    frameStart := 0 },
  { event := event143201
    frameStart := 0 },
  { event := event143202
    frameStart := 0 },
  { event := event143203
    frameStart := 0 },
  { event := event143204
    frameStart := 0 },
  { event := event143205
    frameStart := 0 },
  { event := event143206
    frameStart := 0 },
  { event := event143207
    frameStart := 0 },
  { event := event143208
    frameStart := 0 },
  { event := event143209
    frameStart := 0 },
  { event := event143210
    frameStart := 0 },
  { event := event143211
    frameStart := 0 },
  { event := event143212
    frameStart := 0 },
  { event := event143213
    frameStart := 0 },
  { event := event143214
    frameStart := 0 },
  { event := event143215
    frameStart := 0 }
]

def eventLeaf8951 : Array AnnotatedEvent := #[
  { event := event143216
    frameStart := 0 },
  { event := event143217
    frameStart := 0 },
  { event := event143218
    frameStart := 0 },
  { event := event143219
    frameStart := 0 },
  { event := event143220
    frameStart := 0 },
  { event := event143221
    frameStart := 0 },
  { event := event143222
    frameStart := 0 },
  { event := event143223
    frameStart := 0 },
  { event := event143224
    frameStart := 0 },
  { event := event143225
    frameStart := 0 },
  { event := event143226
    frameStart := 0 },
  { event := event143227
    frameStart := 0 },
  { event := event143228
    frameStart := 0 },
  { event := event143229
    frameStart := 0 },
  { event := event143230
    frameStart := 0 },
  { event := event143231
    frameStart := 0 }
]

def eventLeaf8952 : Array AnnotatedEvent := #[
  { event := event143232
    frameStart := 0 },
  { event := event143233
    frameStart := 0 },
  { event := event143234
    frameStart := 0 },
  { event := event143235
    frameStart := 0 },
  { event := event143236
    frameStart := 0 },
  { event := event143237
    frameStart := 0 },
  { event := event143238
    frameStart := 0 },
  { event := event143239
    frameStart := 0 },
  { event := event143240
    frameStart := 0 },
  { event := event143241
    frameStart := 0 },
  { event := event143242
    frameStart := 0 },
  { event := event143243
    frameStart := 0 },
  { event := event143244
    frameStart := 0 },
  { event := event143245
    frameStart := 0 },
  { event := event143246
    frameStart := 0 },
  { event := event143247
    frameStart := 143247 }
]

def eventLeaf8953 : Array AnnotatedEvent := #[
  { event := event143248
    frameStart := 143247 },
  { event := event143249
    frameStart := 143247 },
  { event := event143250
    frameStart := 143247 },
  { event := event143251
    frameStart := 143247 },
  { event := event143252
    frameStart := 143247 },
  { event := event143253
    frameStart := 143247 },
  { event := event143254
    frameStart := 143247 },
  { event := event143255
    frameStart := 143247 },
  { event := event143256
    frameStart := 143247 },
  { event := event143257
    frameStart := 143247 },
  { event := event143258
    frameStart := 143247 },
  { event := event143259
    frameStart := 143247 },
  { event := event143260
    frameStart := 143247 },
  { event := event143261
    frameStart := 143247 },
  { event := event143262
    frameStart := 143247 },
  { event := event143263
    frameStart := 143247 }
]

def eventLeaf8954 : Array AnnotatedEvent := #[
  { event := event143264
    frameStart := 143247 },
  { event := event143265
    frameStart := 143247 },
  { event := event143266
    frameStart := 143247 },
  { event := event143267
    frameStart := 143247 },
  { event := event143268
    frameStart := 143247 },
  { event := event143269
    frameStart := 143247 },
  { event := event143270
    frameStart := 143247 },
  { event := event143271
    frameStart := 143247 },
  { event := event143272
    frameStart := 143247 },
  { event := event143273
    frameStart := 143247 },
  { event := event143274
    frameStart := 143247 },
  { event := event143275
    frameStart := 143247 },
  { event := event143276
    frameStart := 143247 },
  { event := event143277
    frameStart := 143247 },
  { event := event143278
    frameStart := 143247 },
  { event := event143279
    frameStart := 143247 }
]

def eventLeaf8955 : Array AnnotatedEvent := #[
  { event := event143280
    frameStart := 143247 },
  { event := event143281
    frameStart := 143247 },
  { event := event143282
    frameStart := 143247 },
  { event := event143283
    frameStart := 143247 },
  { event := event143284
    frameStart := 143247 },
  { event := event143285
    frameStart := 143247 },
  { event := event143286
    frameStart := 143247 },
  { event := event143287
    frameStart := 143247 },
  { event := event143288
    frameStart := 143247 },
  { event := event143289
    frameStart := 143247 },
  { event := event143290
    frameStart := 143247 },
  { event := event143291
    frameStart := 143247 },
  { event := event143292
    frameStart := 143247 },
  { event := event143293
    frameStart := 143247 },
  { event := event143294
    frameStart := 143247 },
  { event := event143295
    frameStart := 143247 }
]

def eventLeaf8956 : Array AnnotatedEvent := #[
  { event := event143296
    frameStart := 143247 },
  { event := event143297
    frameStart := 143247 },
  { event := event143298
    frameStart := 143247 },
  { event := event143299
    frameStart := 143247 },
  { event := event143300
    frameStart := 143247 },
  { event := event143301
    frameStart := 143247 },
  { event := event143302
    frameStart := 143247 },
  { event := event143303
    frameStart := 143247 },
  { event := event143304
    frameStart := 143247 },
  { event := event143305
    frameStart := 143247 },
  { event := event143306
    frameStart := 143247 },
  { event := event143307
    frameStart := 143247 },
  { event := event143308
    frameStart := 143247 },
  { event := event143309
    frameStart := 143247 },
  { event := event143310
    frameStart := 143247 },
  { event := event143311
    frameStart := 143247 }
]

def eventLeaf8957 : Array AnnotatedEvent := #[
  { event := event143312
    frameStart := 143247 },
  { event := event143313
    frameStart := 143247 },
  { event := event143314
    frameStart := 143247 },
  { event := event143315
    frameStart := 143247 },
  { event := event143316
    frameStart := 143247 },
  { event := event143317
    frameStart := 143247 },
  { event := event143318
    frameStart := 143247 },
  { event := event143319
    frameStart := 143247 },
  { event := event143320
    frameStart := 143247 },
  { event := event143321
    frameStart := 143247 },
  { event := event143322
    frameStart := 143247 },
  { event := event143323
    frameStart := 143247 },
  { event := event143324
    frameStart := 143247 },
  { event := event143325
    frameStart := 143247 },
  { event := event143326
    frameStart := 143247 },
  { event := event143327
    frameStart := 143247 }
]

def eventLeaf8958 : Array AnnotatedEvent := #[
  { event := event143328
    frameStart := 143247 },
  { event := event143329
    frameStart := 143247 },
  { event := event143330
    frameStart := 143247 },
  { event := event143331
    frameStart := 143247 },
  { event := event143332
    frameStart := 143247 },
  { event := event143333
    frameStart := 143247 },
  { event := event143334
    frameStart := 143247 },
  { event := event143335
    frameStart := 143247 },
  { event := event143336
    frameStart := 143247 },
  { event := event143337
    frameStart := 143247 },
  { event := event143338
    frameStart := 143247 },
  { event := event143339
    frameStart := 143247 },
  { event := event143340
    frameStart := 143247 },
  { event := event143341
    frameStart := 143247 },
  { event := event143342
    frameStart := 143247 },
  { event := event143343
    frameStart := 143247 }
]

def eventLeaf8959 : Array AnnotatedEvent := #[
  { event := event143344
    frameStart := 143247 },
  { event := event143345
    frameStart := 143247 },
  { event := event143346
    frameStart := 143247 },
  { event := event143347
    frameStart := 143247 },
  { event := event143348
    frameStart := 143247 },
  { event := event143349
    frameStart := 143247 },
  { event := event143350
    frameStart := 143247 },
  { event := event143351
    frameStart := 143247 },
  { event := event143352
    frameStart := 143247 },
  { event := event143353
    frameStart := 143247 },
  { event := event143354
    frameStart := 143247 },
  { event := event143355
    frameStart := 143247 },
  { event := event143356
    frameStart := 143247 },
  { event := event143357
    frameStart := 143247 },
  { event := event143358
    frameStart := 143247 },
  { event := event143359
    frameStart := 143247 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events559
