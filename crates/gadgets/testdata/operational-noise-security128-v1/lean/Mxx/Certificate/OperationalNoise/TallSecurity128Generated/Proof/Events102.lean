import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events102

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact26112RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26112RawTermsValid :
    exact26112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26112 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69495⟩⟩) exact26112RawTerms .large 26110 (.finite 321897992872344281445771187322880) (some (26111))

def event26113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69496⟩⟩) 0 ⟨69495⟩ 26112

def event26114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69496⟩⟩) 1 ⟨28074⟩ 21057

def event26115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69496⟩⟩) (.sum [.predecessor 0 26113 .coefficient, .predecessor 1 26114 .coefficient])

def event26116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69496⟩⟩) (.sum [.result 26112 .summary, .result 21057 .summary])

def exact26117RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26117RawTermsValid :
    exact26117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26117 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69496⟩⟩) exact26117RawTerms .large 26115 (.finite 354089550391067611616654269349888) (some (26116))

def event26118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69497⟩⟩) 0 ⟨69496⟩ 26117

def event26119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69497⟩⟩) 1 ⟨30754⟩ 20556

def event26120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69497⟩⟩) (.sum [.predecessor 0 26118 .coefficient, .predecessor 1 26119 .coefficient])

def event26121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69497⟩⟩) (.sum [.result 26117 .summary, .result 20556 .summary])

def exact26122RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26122RawTermsValid :
    exact26122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69497⟩⟩) exact26122RawTerms .large 26120 (.finite 386281697261128003919260020637696) (some (26121))

def event26123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69498⟩⟩) 0 ⟨69497⟩ 26122

def event26124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69498⟩⟩) 1 ⟨36414⟩ 20055

def event26125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69498⟩⟩) (.sum [.predecessor 0 26123 .coefficient, .predecessor 1 26124 .coefficient])

def event26126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69498⟩⟩) (.sum [.result 26122 .summary, .result 20055 .summary])

def exact26127RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26127RawTermsValid :
    exact26127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26127 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69498⟩⟩) exact26127RawTerms .large 26125 (.finite 418474237032079770976347551432704) (some (26126))

def event26128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69499⟩⟩) 0 ⟨69498⟩ 26127

def event26129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69499⟩⟩) 1 ⟨39094⟩ 19554

def event26130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69499⟩⟩) (.sum [.predecessor 0 26128 .coefficient, .predecessor 1 26129 .coefficient])

def event26131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69499⟩⟩) (.sum [.result 26127 .summary, .result 19554 .summary])

def exact26132RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26132RawTermsValid :
    exact26132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26132 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69499⟩⟩) exact26132RawTerms .large 26130 (.finite 450666973253477225410675971981312) (some (26131))

def event26133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69500⟩⟩) 0 ⟨69499⟩ 26132

def event26134 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69500⟩⟩) 1 ⟨41774⟩ 19053

def event26135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69500⟩⟩) (.sum [.predecessor 0 26133 .coefficient, .predecessor 1 26134 .coefficient])

def event26136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69500⟩⟩) (.sum [.result 26132 .summary, .result 19053 .summary])

def exact26137RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26137RawTermsValid :
    exact26137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26137 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69500⟩⟩) exact26137RawTerms .large 26135 (.finite 482860102375766054599486172037120) (some (26136))

def event26138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69501⟩⟩) 0 ⟨69500⟩ 26137

def event26139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69501⟩⟩) 1 ⟨44454⟩ 18552

def event26140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69501⟩⟩) (.sum [.predecessor 0 26138 .coefficient, .predecessor 1 26139 .coefficient])

def event26141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69501⟩⟩) (.sum [.result 26137 .summary, .result 18552 .summary])

def exact26142RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26142RawTermsValid :
    exact26142RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26142 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69501⟩⟩) exact26142RawTerms .large 26140 (.finite 515053820849391945920019041353728) (some (26141))

def event26143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69502⟩⟩) 0 ⟨69501⟩ 26142

def event26144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69502⟩⟩) 1 ⟨47134⟩ 18051

def event26145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69502⟩⟩) (.sum [.predecessor 0 26143 .coefficient, .predecessor 1 26144 .coefficient])

def event26146 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69502⟩⟩) (.sum [.result 26142 .summary, .result 18051 .summary])

def exact26147RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26147RawTermsValid :
    exact26147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26147 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69502⟩⟩) exact26147RawTerms .large 26145 (.finite 547248128674354899372274579931136) (some (26146))

def event26148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69503⟩⟩) 0 ⟨69502⟩ 26147

def event26149 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69503⟩⟩) 1 ⟨49814⟩ 17550

def event26150 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69503⟩⟩) (.sum [.predecessor 0 26148 .coefficient, .predecessor 1 26149 .coefficient])

def event26151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69503⟩⟩) (.sum [.result 26147 .summary, .result 17550 .summary])

def exact26152RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26152RawTermsValid :
    exact26152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26152 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69503⟩⟩) exact26152RawTerms .large 26150 (.finite 579442632949763540201771008262144) (some (26151))

def event26153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70970⟩⟩) 0 ⟨69503⟩ 26152

def event26154 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70970⟩⟩) 1 ⟨70968⟩ 17027

def event26155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70970⟩⟩) (.product (.predecessor 0 26153 .coefficient) (.predecessor 1 26154 .coefficient) (⟨false, false, none, none, none⟩))

def event26156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70970⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) [⟨.result 17027 .coefficient, false, none⟩])

def event26157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70970⟩⟩) (.product (.result 26152 .summary) (.transfer 26156) (⟨false, false, none, none, none⟩))

def event26158 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 29⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26159 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26160 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26159 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26161 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 17⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26162 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 28⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26163 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26164 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26163 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26165 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 16⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26166 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 27⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26167 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26168 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26167 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26169 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 15⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26170 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 26⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26171 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26172 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26171 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26173 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 14⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26174 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 25⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26175 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26176 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26175 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26177 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 13⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26178 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 24⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26179 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26180 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26179 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26181 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 12⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26182 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 22⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26183 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26184 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26183 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26185 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 11⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26186 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 21⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26187 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26188 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26187 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26189 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 10⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26190 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 35⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26191 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26192 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26191 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26193 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 9⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26194 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 34⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26195 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26196 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26195 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26197 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 8⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26198 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 33⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26199 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26200 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26199 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26201 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 7⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26202 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 32⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26203 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26204 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26203 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26205 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 6⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26206 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 31⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26207 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26208 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26207 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26209 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 5⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26210 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 30⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26211 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26212 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26211 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26213 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 4⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26214 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 23⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26215 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26216 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26215 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26217 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 3⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26218 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 20⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26219 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26220 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26219 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26221 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 2⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26222 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 19⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26223 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26224 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26223 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26225 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 1⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def event26226 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 18⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (-1)⟩)

def event26227 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70970⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 17024)

def event26228 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .relation 26227 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩)

def event26229 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70970⟩⟩, .operator (⟨26152, 0⟩, ⟨17027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩)

def exact26230RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65993⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (-1)⟩]

theorem exact26230RawTermsValid :
    exact26230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26230 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70970⟩⟩) exact26230RawTerms .large 26155 (.finite 6221717896068416040249469304417135687106560) (some (26157))

def event26231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68283⟩⟩) 0 ⟨66003⟩ 533

def event26232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68283⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact26233RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩, (1)⟩]

theorem exact26233RawTermsValid :
    exact26233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26233 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68283⟩⟩) exact26233RawTerms (.finite 5647228698) 26232 .exactZero (none)

def event26234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68285⟩⟩) 0 ⟨68283⟩ 26233

def event26235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68285⟩⟩) 1 ⟨2370⟩ 4

def event26236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68285⟩⟩) (.scale (.predecessor 0 26234 .coefficient) (.value (.predecessor 1 26235 .coefficient)))

def exact26237RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩, (1)⟩]

theorem exact26237RawTermsValid :
    exact26237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26237 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68285⟩⟩) exact26237RawTerms (.finite 5647228698) 26236 .exactZero (none)

def event26238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68286⟩⟩) 0 ⟨5443⟩ 17169

def event26239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68286⟩⟩) 1 ⟨68285⟩ 26237

def event26240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68286⟩⟩) (.product (.predecessor 0 26238 .coefficient) (.predecessor 1 26239 .coefficient) (⟨false, false, none, none, none⟩))

def event26241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68286⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩) [⟨.result 26233 .coefficient, false, none⟩])

def event26242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68286⟩⟩) (.product (.result 17169 .summary) (.transfer 26241) (⟨false, false, none, none, none⟩))

def event26243 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68286⟩⟩, .operator (⟨17169, 0⟩, ⟨26237, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩, (1)⟩)

def event26244 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68284⟩⟩)

def event26245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event26246 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event26247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event26248 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event26249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event26250 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event26251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event26252 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event26253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 26252

def event26254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 26250

def event26255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 26253 .coefficient) (.value (.predecessor 1 26254 .coefficient)))

def event26256 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event26257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 26256

def event26258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 26248

def event26259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 26257 .coefficient, .predecessor 1 26258 .coefficient])

def event26260 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event26261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 26260

def event26262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 26246

def event26263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 26262 .coefficient))

def event26264 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event26265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47626⟩⟩) 0 ⟨5439⟩ 26264

def event26266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47626⟩⟩) (.authority (.programFamilyFact))

def exact26267RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩]

theorem exact26267RawTermsValid :
    exact26267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26267 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47626⟩⟩) exact26267RawTerms (.finite 60) 26266 .exactZero (none)

def event26268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14951⟩⟩) 0 ⟨5439⟩ 26264

def event26269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14951⟩⟩) (.authority (.programFamilyFact))

def exact26270RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩], []⟩, (1)⟩]

theorem exact26270RawTermsValid :
    exact26270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14951⟩⟩) exact26270RawTerms (.finite 60) 26269 .exactZero (none)

def event26271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 0 ⟨14951⟩ 26270

def event26272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 1 ⟨47626⟩ 26267

def event26273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47627⟩⟩) (.product (.predecessor 0 26271 .coefficient) (.predecessor 1 26272 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47627⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩) [⟨.result 26270 .coefficient, true, some 1⟩, ⟨.result 26267 .coefficient, true, some 1⟩])

def event26275 : Event := .survivorFold (1) 26274

def exact26276RawTerms : List Term := []

theorem exact26276RawTermsValid :
    exact26276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26276 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47627⟩⟩) exact26276RawTerms (.finite 3600) 26273 (.finite 3600) (some (26274))

def event26277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47628⟩⟩) 0 ⟨47627⟩ 26276

def event26278 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.identity (.predecessor 0 26277 .coefficient))

def event26279 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.finite 3600)

def event26280 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48078⟩⟩) 0 ⟨47628⟩ 26279

def event26281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48078⟩⟩) (.authority (.programFamilyFact))

def exact26282RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], []⟩, (1)⟩]

theorem exact26282RawTermsValid :
    exact26282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26282 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48078⟩⟩) exact26282RawTerms (.finite 60) 26281 .exactZero (none)

def event26283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48079⟩⟩) 0 ⟨48078⟩ 26282

def event26284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48079⟩⟩) (.identity (.predecessor 0 26283 .coefficient))

def event26285 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48079⟩⟩) (.finite 60)

def event26286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48249⟩⟩) 0 ⟨48079⟩ 26285

def event26287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48249⟩⟩) (.authority (.programFamilyFact))

def exact26288RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], []⟩, (1)⟩]

theorem exact26288RawTermsValid :
    exact26288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26288 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48249⟩⟩) exact26288RawTerms (.finite 63) 26287 .exactZero (none)

def event26289 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44946⟩⟩) 0 ⟨5439⟩ 26264

def event26290 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44946⟩⟩) (.authority (.programFamilyFact))

def exact26291RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44946⟩⟩], []⟩, (1)⟩]

theorem exact26291RawTermsValid :
    exact26291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26291 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44946⟩⟩) exact26291RawTerms (.finite 58) 26290 .exactZero (none)

def event26292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14651⟩⟩) 0 ⟨5439⟩ 26264

def event26293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14651⟩⟩) (.authority (.programFamilyFact))

def exact26294RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14651⟩⟩], []⟩, (1)⟩]

theorem exact26294RawTermsValid :
    exact26294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26294 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14651⟩⟩) exact26294RawTerms (.finite 58) 26293 .exactZero (none)

def event26295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44947⟩⟩) 0 ⟨14651⟩ 26294

def event26296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44947⟩⟩) 1 ⟨44946⟩ 26291

def event26297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44947⟩⟩) (.product (.predecessor 0 26295 .coefficient) (.predecessor 1 26296 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44947⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], []⟩) [⟨.result 26294 .coefficient, true, some 1⟩, ⟨.result 26291 .coefficient, true, some 1⟩])

def event26299 : Event := .survivorFold (1) 26298

def exact26300RawTerms : List Term := []

theorem exact26300RawTermsValid :
    exact26300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44947⟩⟩) exact26300RawTerms (.finite 3364) 26297 (.finite 3364) (some (26298))

def event26301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44948⟩⟩) 0 ⟨44947⟩ 26300

def event26302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44948⟩⟩) (.identity (.predecessor 0 26301 .coefficient))

def event26303 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44948⟩⟩) (.finite 3364)

def event26304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45398⟩⟩) 0 ⟨44948⟩ 26303

def event26305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45398⟩⟩) (.authority (.programFamilyFact))

def exact26306RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], []⟩, (1)⟩]

theorem exact26306RawTermsValid :
    exact26306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26306 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45398⟩⟩) exact26306RawTerms (.finite 58) 26305 .exactZero (none)

def event26307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45399⟩⟩) 0 ⟨45398⟩ 26306

def event26308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45399⟩⟩) (.identity (.predecessor 0 26307 .coefficient))

def event26309 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45399⟩⟩) (.finite 58)

def event26310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45569⟩⟩) 0 ⟨45399⟩ 26309

def event26311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45569⟩⟩) (.authority (.programFamilyFact))

def exact26312RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45569⟩⟩], []⟩, (1)⟩]

theorem exact26312RawTermsValid :
    exact26312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26312 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45569⟩⟩) exact26312RawTerms (.finite 63) 26311 .exactZero (none)

def event26313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42266⟩⟩) 0 ⟨5439⟩ 26264

def event26314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42266⟩⟩) (.authority (.programFamilyFact))

def exact26315RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42266⟩⟩], []⟩, (1)⟩]

theorem exact26315RawTermsValid :
    exact26315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26315 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42266⟩⟩) exact26315RawTerms (.finite 52) 26314 .exactZero (none)

def event26316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14351⟩⟩) 0 ⟨5439⟩ 26264

def event26317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14351⟩⟩) (.authority (.programFamilyFact))

def exact26318RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14351⟩⟩], []⟩, (1)⟩]

theorem exact26318RawTermsValid :
    exact26318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26318 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14351⟩⟩) exact26318RawTerms (.finite 52) 26317 .exactZero (none)

def event26319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42267⟩⟩) 0 ⟨14351⟩ 26318

def event26320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42267⟩⟩) 1 ⟨42266⟩ 26315

def event26321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42267⟩⟩) (.product (.predecessor 0 26319 .coefficient) (.predecessor 1 26320 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42267⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14351⟩⟩, ⟨.program ⟨257⟩, ⟨42266⟩⟩], []⟩) [⟨.result 26318 .coefficient, true, some 1⟩, ⟨.result 26315 .coefficient, true, some 1⟩])

def event26323 : Event := .survivorFold (1) 26322

def exact26324RawTerms : List Term := []

theorem exact26324RawTermsValid :
    exact26324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26324 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42267⟩⟩) exact26324RawTerms (.finite 2704) 26321 (.finite 2704) (some (26322))

def event26325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42268⟩⟩) 0 ⟨42267⟩ 26324

def event26326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42268⟩⟩) (.identity (.predecessor 0 26325 .coefficient))

def event26327 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42268⟩⟩) (.finite 2704)

def event26328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42718⟩⟩) 0 ⟨42268⟩ 26327

def event26329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42718⟩⟩) (.authority (.programFamilyFact))

def exact26330RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], []⟩, (1)⟩]

theorem exact26330RawTermsValid :
    exact26330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42718⟩⟩) exact26330RawTerms (.finite 52) 26329 .exactZero (none)

def event26331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42719⟩⟩) 0 ⟨42718⟩ 26330

def event26332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42719⟩⟩) (.identity (.predecessor 0 26331 .coefficient))

def event26333 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42719⟩⟩) (.finite 52)

def event26334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42885⟩⟩) 0 ⟨42719⟩ 26333

def event26335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42885⟩⟩) (.authority (.programFamilyFact))

def exact26336RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42885⟩⟩], []⟩, (1)⟩]

theorem exact26336RawTermsValid :
    exact26336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42885⟩⟩) exact26336RawTerms (.finite 63) 26335 .exactZero (none)

def event26337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39586⟩⟩) 0 ⟨5439⟩ 26264

def event26338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39586⟩⟩) (.authority (.programFamilyFact))

def exact26339RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39586⟩⟩], []⟩, (1)⟩]

theorem exact26339RawTermsValid :
    exact26339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26339 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39586⟩⟩) exact26339RawTerms (.finite 46) 26338 .exactZero (none)

def event26340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14051⟩⟩) 0 ⟨5439⟩ 26264

def event26341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14051⟩⟩) (.authority (.programFamilyFact))

def exact26342RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩], []⟩, (1)⟩]

theorem exact26342RawTermsValid :
    exact26342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14051⟩⟩) exact26342RawTerms (.finite 46) 26341 .exactZero (none)

def event26343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39587⟩⟩) 0 ⟨14051⟩ 26342

def event26344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39587⟩⟩) 1 ⟨39586⟩ 26339

def event26345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39587⟩⟩) (.product (.predecessor 0 26343 .coefficient) (.predecessor 1 26344 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26346 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39587⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], []⟩) [⟨.result 26342 .coefficient, true, some 1⟩, ⟨.result 26339 .coefficient, true, some 1⟩])

def event26347 : Event := .survivorFold (1) 26346

def exact26348RawTerms : List Term := []

theorem exact26348RawTermsValid :
    exact26348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26348 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39587⟩⟩) exact26348RawTerms (.finite 2116) 26345 (.finite 2116) (some (26346))

def event26349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39588⟩⟩) 0 ⟨39587⟩ 26348

def event26350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39588⟩⟩) (.identity (.predecessor 0 26349 .coefficient))

def event26351 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39588⟩⟩) (.finite 2116)

def event26352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40038⟩⟩) 0 ⟨39588⟩ 26351

def event26353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40038⟩⟩) (.authority (.programFamilyFact))

def exact26354RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], []⟩, (1)⟩]

theorem exact26354RawTermsValid :
    exact26354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40038⟩⟩) exact26354RawTerms (.finite 46) 26353 .exactZero (none)

def event26355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40039⟩⟩) 0 ⟨40038⟩ 26354

def event26356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40039⟩⟩) (.identity (.predecessor 0 26355 .coefficient))

def event26357 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40039⟩⟩) (.finite 46)

def event26358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40205⟩⟩) 0 ⟨40039⟩ 26357

def event26359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40205⟩⟩) (.authority (.programFamilyFact))

def exact26360RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40205⟩⟩], []⟩, (1)⟩]

theorem exact26360RawTermsValid :
    exact26360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40205⟩⟩) exact26360RawTerms (.finite 63) 26359 .exactZero (none)

def event26361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36906⟩⟩) 0 ⟨5439⟩ 26264

def event26362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36906⟩⟩) (.authority (.programFamilyFact))

def exact26363RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨36906⟩⟩], []⟩, (1)⟩]

theorem exact26363RawTermsValid :
    exact26363RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26363 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36906⟩⟩) exact26363RawTerms (.finite 42) 26362 .exactZero (none)

def event26364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13751⟩⟩) 0 ⟨5439⟩ 26264

def event26365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13751⟩⟩) (.authority (.programFamilyFact))

def exact26366RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩], []⟩, (1)⟩]

theorem exact26366RawTermsValid :
    exact26366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13751⟩⟩) exact26366RawTerms (.finite 42) 26365 .exactZero (none)

def event26367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36907⟩⟩) 0 ⟨13751⟩ 26366

def eventLeaf1632 : Array AnnotatedEvent := #[
  { event := event26112
    frameStart := 0 },
  { event := event26113
    frameStart := 0 },
  { event := event26114
    frameStart := 0 },
  { event := event26115
    frameStart := 0 },
  { event := event26116
    frameStart := 0 },
  { event := event26117
    frameStart := 0 },
  { event := event26118
    frameStart := 0 },
  { event := event26119
    frameStart := 0 },
  { event := event26120
    frameStart := 0 },
  { event := event26121
    frameStart := 0 },
  { event := event26122
    frameStart := 0 },
  { event := event26123
    frameStart := 0 },
  { event := event26124
    frameStart := 0 },
  { event := event26125
    frameStart := 0 },
  { event := event26126
    frameStart := 0 },
  { event := event26127
    frameStart := 0 }
]

def eventLeaf1633 : Array AnnotatedEvent := #[
  { event := event26128
    frameStart := 0 },
  { event := event26129
    frameStart := 0 },
  { event := event26130
    frameStart := 0 },
  { event := event26131
    frameStart := 0 },
  { event := event26132
    frameStart := 0 },
  { event := event26133
    frameStart := 0 },
  { event := event26134
    frameStart := 0 },
  { event := event26135
    frameStart := 0 },
  { event := event26136
    frameStart := 0 },
  { event := event26137
    frameStart := 0 },
  { event := event26138
    frameStart := 0 },
  { event := event26139
    frameStart := 0 },
  { event := event26140
    frameStart := 0 },
  { event := event26141
    frameStart := 0 },
  { event := event26142
    frameStart := 0 },
  { event := event26143
    frameStart := 0 }
]

def eventLeaf1634 : Array AnnotatedEvent := #[
  { event := event26144
    frameStart := 0 },
  { event := event26145
    frameStart := 0 },
  { event := event26146
    frameStart := 0 },
  { event := event26147
    frameStart := 0 },
  { event := event26148
    frameStart := 0 },
  { event := event26149
    frameStart := 0 },
  { event := event26150
    frameStart := 0 },
  { event := event26151
    frameStart := 0 },
  { event := event26152
    frameStart := 0 },
  { event := event26153
    frameStart := 0 },
  { event := event26154
    frameStart := 0 },
  { event := event26155
    frameStart := 0 },
  { event := event26156
    frameStart := 0 },
  { event := event26157
    frameStart := 0 },
  { event := event26158
    frameStart := 0 },
  { event := event26159
    frameStart := 0 }
]

def eventLeaf1635 : Array AnnotatedEvent := #[
  { event := event26160
    frameStart := 0 },
  { event := event26161
    frameStart := 0 },
  { event := event26162
    frameStart := 0 },
  { event := event26163
    frameStart := 0 },
  { event := event26164
    frameStart := 0 },
  { event := event26165
    frameStart := 0 },
  { event := event26166
    frameStart := 0 },
  { event := event26167
    frameStart := 0 },
  { event := event26168
    frameStart := 0 },
  { event := event26169
    frameStart := 0 },
  { event := event26170
    frameStart := 0 },
  { event := event26171
    frameStart := 0 },
  { event := event26172
    frameStart := 0 },
  { event := event26173
    frameStart := 0 },
  { event := event26174
    frameStart := 0 },
  { event := event26175
    frameStart := 0 }
]

def eventLeaf1636 : Array AnnotatedEvent := #[
  { event := event26176
    frameStart := 0 },
  { event := event26177
    frameStart := 0 },
  { event := event26178
    frameStart := 0 },
  { event := event26179
    frameStart := 0 },
  { event := event26180
    frameStart := 0 },
  { event := event26181
    frameStart := 0 },
  { event := event26182
    frameStart := 0 },
  { event := event26183
    frameStart := 0 },
  { event := event26184
    frameStart := 0 },
  { event := event26185
    frameStart := 0 },
  { event := event26186
    frameStart := 0 },
  { event := event26187
    frameStart := 0 },
  { event := event26188
    frameStart := 0 },
  { event := event26189
    frameStart := 0 },
  { event := event26190
    frameStart := 0 },
  { event := event26191
    frameStart := 0 }
]

def eventLeaf1637 : Array AnnotatedEvent := #[
  { event := event26192
    frameStart := 0 },
  { event := event26193
    frameStart := 0 },
  { event := event26194
    frameStart := 0 },
  { event := event26195
    frameStart := 0 },
  { event := event26196
    frameStart := 0 },
  { event := event26197
    frameStart := 0 },
  { event := event26198
    frameStart := 0 },
  { event := event26199
    frameStart := 0 },
  { event := event26200
    frameStart := 0 },
  { event := event26201
    frameStart := 0 },
  { event := event26202
    frameStart := 0 },
  { event := event26203
    frameStart := 0 },
  { event := event26204
    frameStart := 0 },
  { event := event26205
    frameStart := 0 },
  { event := event26206
    frameStart := 0 },
  { event := event26207
    frameStart := 0 }
]

def eventLeaf1638 : Array AnnotatedEvent := #[
  { event := event26208
    frameStart := 0 },
  { event := event26209
    frameStart := 0 },
  { event := event26210
    frameStart := 0 },
  { event := event26211
    frameStart := 0 },
  { event := event26212
    frameStart := 0 },
  { event := event26213
    frameStart := 0 },
  { event := event26214
    frameStart := 0 },
  { event := event26215
    frameStart := 0 },
  { event := event26216
    frameStart := 0 },
  { event := event26217
    frameStart := 0 },
  { event := event26218
    frameStart := 0 },
  { event := event26219
    frameStart := 0 },
  { event := event26220
    frameStart := 0 },
  { event := event26221
    frameStart := 0 },
  { event := event26222
    frameStart := 0 },
  { event := event26223
    frameStart := 0 }
]

def eventLeaf1639 : Array AnnotatedEvent := #[
  { event := event26224
    frameStart := 0 },
  { event := event26225
    frameStart := 0 },
  { event := event26226
    frameStart := 0 },
  { event := event26227
    frameStart := 0 },
  { event := event26228
    frameStart := 0 },
  { event := event26229
    frameStart := 0 },
  { event := event26230
    frameStart := 0 },
  { event := event26231
    frameStart := 0 },
  { event := event26232
    frameStart := 0 },
  { event := event26233
    frameStart := 0 },
  { event := event26234
    frameStart := 0 },
  { event := event26235
    frameStart := 0 },
  { event := event26236
    frameStart := 0 },
  { event := event26237
    frameStart := 0 },
  { event := event26238
    frameStart := 0 },
  { event := event26239
    frameStart := 0 }
]

def eventLeaf1640 : Array AnnotatedEvent := #[
  { event := event26240
    frameStart := 0 },
  { event := event26241
    frameStart := 0 },
  { event := event26242
    frameStart := 0 },
  { event := event26243
    frameStart := 0 },
  { event := event26244
    frameStart := 26244 },
  { event := event26245
    frameStart := 26244 },
  { event := event26246
    frameStart := 26244 },
  { event := event26247
    frameStart := 26244 },
  { event := event26248
    frameStart := 26244 },
  { event := event26249
    frameStart := 26244 },
  { event := event26250
    frameStart := 26244 },
  { event := event26251
    frameStart := 26244 },
  { event := event26252
    frameStart := 26244 },
  { event := event26253
    frameStart := 26244 },
  { event := event26254
    frameStart := 26244 },
  { event := event26255
    frameStart := 26244 }
]

def eventLeaf1641 : Array AnnotatedEvent := #[
  { event := event26256
    frameStart := 26244 },
  { event := event26257
    frameStart := 26244 },
  { event := event26258
    frameStart := 26244 },
  { event := event26259
    frameStart := 26244 },
  { event := event26260
    frameStart := 26244 },
  { event := event26261
    frameStart := 26244 },
  { event := event26262
    frameStart := 26244 },
  { event := event26263
    frameStart := 26244 },
  { event := event26264
    frameStart := 26244 },
  { event := event26265
    frameStart := 26244 },
  { event := event26266
    frameStart := 26244 },
  { event := event26267
    frameStart := 26244 },
  { event := event26268
    frameStart := 26244 },
  { event := event26269
    frameStart := 26244 },
  { event := event26270
    frameStart := 26244 },
  { event := event26271
    frameStart := 26244 }
]

def eventLeaf1642 : Array AnnotatedEvent := #[
  { event := event26272
    frameStart := 26244 },
  { event := event26273
    frameStart := 26244 },
  { event := event26274
    frameStart := 26244 },
  { event := event26275
    frameStart := 26244 },
  { event := event26276
    frameStart := 26244 },
  { event := event26277
    frameStart := 26244 },
  { event := event26278
    frameStart := 26244 },
  { event := event26279
    frameStart := 26244 },
  { event := event26280
    frameStart := 26244 },
  { event := event26281
    frameStart := 26244 },
  { event := event26282
    frameStart := 26244 },
  { event := event26283
    frameStart := 26244 },
  { event := event26284
    frameStart := 26244 },
  { event := event26285
    frameStart := 26244 },
  { event := event26286
    frameStart := 26244 },
  { event := event26287
    frameStart := 26244 }
]

def eventLeaf1643 : Array AnnotatedEvent := #[
  { event := event26288
    frameStart := 26244 },
  { event := event26289
    frameStart := 26244 },
  { event := event26290
    frameStart := 26244 },
  { event := event26291
    frameStart := 26244 },
  { event := event26292
    frameStart := 26244 },
  { event := event26293
    frameStart := 26244 },
  { event := event26294
    frameStart := 26244 },
  { event := event26295
    frameStart := 26244 },
  { event := event26296
    frameStart := 26244 },
  { event := event26297
    frameStart := 26244 },
  { event := event26298
    frameStart := 26244 },
  { event := event26299
    frameStart := 26244 },
  { event := event26300
    frameStart := 26244 },
  { event := event26301
    frameStart := 26244 },
  { event := event26302
    frameStart := 26244 },
  { event := event26303
    frameStart := 26244 }
]

def eventLeaf1644 : Array AnnotatedEvent := #[
  { event := event26304
    frameStart := 26244 },
  { event := event26305
    frameStart := 26244 },
  { event := event26306
    frameStart := 26244 },
  { event := event26307
    frameStart := 26244 },
  { event := event26308
    frameStart := 26244 },
  { event := event26309
    frameStart := 26244 },
  { event := event26310
    frameStart := 26244 },
  { event := event26311
    frameStart := 26244 },
  { event := event26312
    frameStart := 26244 },
  { event := event26313
    frameStart := 26244 },
  { event := event26314
    frameStart := 26244 },
  { event := event26315
    frameStart := 26244 },
  { event := event26316
    frameStart := 26244 },
  { event := event26317
    frameStart := 26244 },
  { event := event26318
    frameStart := 26244 },
  { event := event26319
    frameStart := 26244 }
]

def eventLeaf1645 : Array AnnotatedEvent := #[
  { event := event26320
    frameStart := 26244 },
  { event := event26321
    frameStart := 26244 },
  { event := event26322
    frameStart := 26244 },
  { event := event26323
    frameStart := 26244 },
  { event := event26324
    frameStart := 26244 },
  { event := event26325
    frameStart := 26244 },
  { event := event26326
    frameStart := 26244 },
  { event := event26327
    frameStart := 26244 },
  { event := event26328
    frameStart := 26244 },
  { event := event26329
    frameStart := 26244 },
  { event := event26330
    frameStart := 26244 },
  { event := event26331
    frameStart := 26244 },
  { event := event26332
    frameStart := 26244 },
  { event := event26333
    frameStart := 26244 },
  { event := event26334
    frameStart := 26244 },
  { event := event26335
    frameStart := 26244 }
]

def eventLeaf1646 : Array AnnotatedEvent := #[
  { event := event26336
    frameStart := 26244 },
  { event := event26337
    frameStart := 26244 },
  { event := event26338
    frameStart := 26244 },
  { event := event26339
    frameStart := 26244 },
  { event := event26340
    frameStart := 26244 },
  { event := event26341
    frameStart := 26244 },
  { event := event26342
    frameStart := 26244 },
  { event := event26343
    frameStart := 26244 },
  { event := event26344
    frameStart := 26244 },
  { event := event26345
    frameStart := 26244 },
  { event := event26346
    frameStart := 26244 },
  { event := event26347
    frameStart := 26244 },
  { event := event26348
    frameStart := 26244 },
  { event := event26349
    frameStart := 26244 },
  { event := event26350
    frameStart := 26244 },
  { event := event26351
    frameStart := 26244 }
]

def eventLeaf1647 : Array AnnotatedEvent := #[
  { event := event26352
    frameStart := 26244 },
  { event := event26353
    frameStart := 26244 },
  { event := event26354
    frameStart := 26244 },
  { event := event26355
    frameStart := 26244 },
  { event := event26356
    frameStart := 26244 },
  { event := event26357
    frameStart := 26244 },
  { event := event26358
    frameStart := 26244 },
  { event := event26359
    frameStart := 26244 },
  { event := event26360
    frameStart := 26244 },
  { event := event26361
    frameStart := 26244 },
  { event := event26362
    frameStart := 26244 },
  { event := event26363
    frameStart := 26244 },
  { event := event26364
    frameStart := 26244 },
  { event := event26365
    frameStart := 26244 },
  { event := event26366
    frameStart := 26244 },
  { event := event26367
    frameStart := 26244 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events102
