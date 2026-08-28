import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events278

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event71168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57255⟩⟩) (.sum [.predecessor 0 71166 .coefficient, .predecessor 1 71167 .coefficient])

def exact71169RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩]

theorem exact71169RawTermsValid :
    exact71169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71169 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57255⟩⟩) exact71169RawTerms (.finite 374) 71168 .exactZero (none)

def event71170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60235⟩⟩) 0 ⟨57255⟩ 71169

def event71171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60235⟩⟩) 1 ⟨60234⟩ 70984

def event71172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60235⟩⟩) (.sum [.predecessor 0 71170 .coefficient, .predecessor 1 71171 .coefficient])

def exact71173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩]

theorem exact71173RawTermsValid :
    exact71173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60235⟩⟩) exact71173RawTerms (.finite 435) 71172 .exactZero (none)

def event71174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63215⟩⟩) 0 ⟨60235⟩ 71173

def event71175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63215⟩⟩) 1 ⟨63214⟩ 70961

def event71176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63215⟩⟩) (.sum [.predecessor 0 71174 .coefficient, .predecessor 1 71175 .coefficient])

def exact71177RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩]

theorem exact71177RawTermsValid :
    exact71177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71177 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63215⟩⟩) exact71177RawTerms (.finite 496) 71176 .exactZero (none)

def event71178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67092⟩⟩) 0 ⟨63215⟩ 71177

def event71179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67092⟩⟩) 1 ⟨67091⟩ 70938

def event71180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67092⟩⟩) (.sum [.predecessor 0 71178 .coefficient, .predecessor 1 71179 .coefficient])

def exact71181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71181RawTermsValid :
    exact71181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67092⟩⟩) exact71181RawTerms (.finite 558) 71180 .exactZero (none)

def event71182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67093⟩⟩) 0 ⟨67092⟩ 71181

def event71183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67093⟩⟩) 1 ⟨26710⟩ 70915

def event71184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67093⟩⟩) (.sum [.predecessor 0 71182 .coefficient, .predecessor 1 71183 .coefficient])

def exact71185RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71185RawTermsValid :
    exact71185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71185 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67093⟩⟩) exact71185RawTerms (.finite 620) 71184 .exactZero (none)

def event71186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67094⟩⟩) 0 ⟨67093⟩ 71185

def event71187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67094⟩⟩) 1 ⟨29390⟩ 70892

def event71188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67094⟩⟩) (.sum [.predecessor 0 71186 .coefficient, .predecessor 1 71187 .coefficient])

def exact71189RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71189RawTermsValid :
    exact71189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71189 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67094⟩⟩) exact71189RawTerms (.finite 682) 71188 .exactZero (none)

def event71190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67095⟩⟩) 0 ⟨67094⟩ 71189

def event71191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67095⟩⟩) 1 ⟨35054⟩ 70869

def event71192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67095⟩⟩) (.sum [.predecessor 0 71190 .coefficient, .predecessor 1 71191 .coefficient])

def exact71193RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71193RawTermsValid :
    exact71193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67095⟩⟩) exact71193RawTerms (.finite 744) 71192 .exactZero (none)

def event71194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67096⟩⟩) 0 ⟨67095⟩ 71193

def event71195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67096⟩⟩) 1 ⟨37734⟩ 70846

def event71196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67096⟩⟩) (.sum [.predecessor 0 71194 .coefficient, .predecessor 1 71195 .coefficient])

def exact71197RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71197RawTermsValid :
    exact71197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71197 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67096⟩⟩) exact71197RawTerms (.finite 807) 71196 .exactZero (none)

def event71198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67097⟩⟩) 0 ⟨67096⟩ 71197

def event71199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67097⟩⟩) 1 ⟨40410⟩ 70823

def event71200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67097⟩⟩) (.sum [.predecessor 0 71198 .coefficient, .predecessor 1 71199 .coefficient])

def exact71201RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71201RawTermsValid :
    exact71201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67097⟩⟩) exact71201RawTerms (.finite 870) 71200 .exactZero (none)

def event71202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67098⟩⟩) 0 ⟨67097⟩ 71201

def event71203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67098⟩⟩) 1 ⟨43090⟩ 70800

def event71204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67098⟩⟩) (.sum [.predecessor 0 71202 .coefficient, .predecessor 1 71203 .coefficient])

def exact71205RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71205RawTermsValid :
    exact71205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71205 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67098⟩⟩) exact71205RawTerms (.finite 933) 71204 .exactZero (none)

def event71206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67099⟩⟩) 0 ⟨67098⟩ 71205

def event71207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67099⟩⟩) 1 ⟨45774⟩ 70777

def event71208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67099⟩⟩) (.sum [.predecessor 0 71206 .coefficient, .predecessor 1 71207 .coefficient])

def exact71209RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71209RawTermsValid :
    exact71209RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71209 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67099⟩⟩) exact71209RawTerms (.finite 996) 71208 .exactZero (none)

def event71210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67100⟩⟩) 0 ⟨67099⟩ 71209

def event71211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67100⟩⟩) 1 ⟨48454⟩ 70754

def event71212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67100⟩⟩) (.sum [.predecessor 0 71210 .coefficient, .predecessor 1 71211 .coefficient])

def exact71213RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71213RawTermsValid :
    exact71213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71213 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67100⟩⟩) exact71213RawTerms (.finite 1059) 71212 .exactZero (none)

def event71214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67101⟩⟩) 0 ⟨67100⟩ 71213

def event71215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67101⟩⟩) (.identity (.predecessor 0 71214 .coefficient))

def event71216 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨67101⟩⟩) (.finite 1059)

def event71217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68871⟩⟩) 0 ⟨67101⟩ 71216

def event71218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68871⟩⟩) (.authority (.programFamilyFact))

def event71219 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68871⟩⟩) (.finite 1152)

def event71220 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event71221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68872⟩⟩) 0 ⟨7177⟩ 71220

def event71222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68872⟩⟩) 1 ⟨68871⟩ 71219

def event71223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68872⟩⟩) (.authority (.operator))

def exact71224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (1)⟩]

theorem exact71224RawTermsValid :
    exact71224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68872⟩⟩) exact71224RawTerms .large 71223 .exactZero (none)

def event71225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71469⟩⟩) 0 ⟨68872⟩ 71224

def event71226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71469⟩⟩) (.authority (.operator))

def exact71227RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩]

theorem exact71227RawTermsValid :
    exact71227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71469⟩⟩) exact71227RawTerms (.finite 8192) 71226 .exactZero (none)

def event71228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event71229 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event71230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69115⟩⟩) 0 ⟨67101⟩ 71216

def event71231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69115⟩⟩) 1 ⟨136⟩ 71229

def event71232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69115⟩⟩) (.sum [.predecessor 0 71230 .coefficient, .predecessor 1 71231 .coefficient])

def event71233 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69115⟩⟩) (.finite 1059)

def event71234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69116⟩⟩) 0 ⟨69115⟩ 71233

def event71235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69116⟩⟩) (.identity (.predecessor 0 71234 .coefficient))

def exact71236RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact71236RawTermsValid :
    exact71236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69116⟩⟩) exact71236RawTerms (.finite 1059) 71235 .exactZero (none)

def event71237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact71238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact71238RawTermsValid :
    exact71238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact71238RawTerms .large 71237 .exactZero (none)

def event71239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69117⟩⟩) 0 ⟨6908⟩ 71238

def event71240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69117⟩⟩) 1 ⟨69116⟩ 71236

def event71241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69117⟩⟩) (.product (.predecessor 0 71239 .coefficient) (.predecessor 1 71240 .coefficient) (⟨false, false, none, none, none⟩))

def event71242 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71243 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71244 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71245 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71246 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71247 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71248 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71249 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71250 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71251 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71252 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71253 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71254 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71255 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71257 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71258 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event71259 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69117⟩⟩, .operator (⟨71238, 0⟩, ⟨71236, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact71260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact71260RawTermsValid :
    exact71260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69117⟩⟩) exact71260RawTerms .large 71241 .exactZero (none)

def event71261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 71220

def event71262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact71263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact71263RawTermsValid :
    exact71263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact71263RawTerms .large 71262 .exactZero (none)

def event71264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 71220

def event71265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact71266RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact71266RawTermsValid :
    exact71266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact71266RawTerms .large 71265 .exactZero (none)

def event71267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 71220

def event71268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact71269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact71269RawTermsValid :
    exact71269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71269 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact71269RawTerms .large 71268 .exactZero (none)

def event71270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 71220

def event71271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact71272RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact71272RawTermsValid :
    exact71272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact71272RawTerms .large 71271 .exactZero (none)

def event71273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 71220

def event71274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact71275RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact71275RawTermsValid :
    exact71275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71275 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact71275RawTerms .large 71274 .exactZero (none)

def event71276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7222⟩⟩) 0 ⟨7177⟩ 71220

def event71277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7222⟩⟩) (.authority (.operator))

def exact71278RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact71278RawTermsValid :
    exact71278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71278 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7222⟩⟩) exact71278RawTerms .large 71277 .exactZero (none)

def event71279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 71220

def event71280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact71281RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact71281RawTermsValid :
    exact71281RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71281 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact71281RawTerms .large 71280 .exactZero (none)

def event71282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 71220

def event71283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact71284RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact71284RawTermsValid :
    exact71284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact71284RawTerms .large 71283 .exactZero (none)

def event71285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 71220

def event71286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact71287RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact71287RawTermsValid :
    exact71287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71287 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact71287RawTerms .large 71286 .exactZero (none)

def event71288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7214⟩⟩) 0 ⟨7177⟩ 71220

def event71289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7214⟩⟩) (.authority (.operator))

def exact71290RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact71290RawTermsValid :
    exact71290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7214⟩⟩) exact71290RawTerms .large 71289 .exactZero (none)

def event71291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 71220

def event71292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact71293RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact71293RawTermsValid :
    exact71293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact71293RawTerms .large 71292 .exactZero (none)

def event71294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 71220

def event71295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact71296RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact71296RawTermsValid :
    exact71296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact71296RawTerms .large 71295 .exactZero (none)

def event71297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 71220

def event71298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact71299RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact71299RawTermsValid :
    exact71299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact71299RawTerms .large 71298 .exactZero (none)

def event71300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 71220

def event71301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact71302RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact71302RawTermsValid :
    exact71302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact71302RawTerms .large 71301 .exactZero (none)

def event71303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 71220

def event71304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact71305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact71305RawTermsValid :
    exact71305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact71305RawTerms .large 71304 .exactZero (none)

def event71306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7202⟩⟩) 0 ⟨7177⟩ 71220

def event71307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7202⟩⟩) (.authority (.operator))

def exact71308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact71308RawTermsValid :
    exact71308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7202⟩⟩) exact71308RawTerms .large 71307 .exactZero (none)

def event71309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 71220

def event71310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact71311RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact71311RawTermsValid :
    exact71311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71311 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact71311RawTerms .large 71310 .exactZero (none)

def event71312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 71220

def event71313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact71314RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact71314RawTermsValid :
    exact71314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact71314RawTerms .large 71313 .exactZero (none)

def event71315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 0 ⟨7198⟩ 71314

def event71316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 1 ⟨7200⟩ 71311

def event71317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7309⟩⟩) (.sum [.predecessor 0 71315 .coefficient, .predecessor 1 71316 .coefficient])

def exact71318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact71318RawTermsValid :
    exact71318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71318 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7309⟩⟩) exact71318RawTerms .large 71317 .exactZero (none)

def event71319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 0 ⟨7309⟩ 71318

def event71320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 1 ⟨7202⟩ 71308

def event71321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7310⟩⟩) (.sum [.predecessor 0 71319 .coefficient, .predecessor 1 71320 .coefficient])

def exact71322RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact71322RawTermsValid :
    exact71322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7310⟩⟩) exact71322RawTerms .large 71321 .exactZero (none)

def event71323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 0 ⟨7310⟩ 71322

def event71324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 1 ⟨7204⟩ 71305

def event71325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7311⟩⟩) (.sum [.predecessor 0 71323 .coefficient, .predecessor 1 71324 .coefficient])

def exact71326RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact71326RawTermsValid :
    exact71326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7311⟩⟩) exact71326RawTerms .large 71325 .exactZero (none)

def event71327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 0 ⟨7311⟩ 71326

def event71328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 1 ⟨7206⟩ 71302

def event71329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7312⟩⟩) (.sum [.predecessor 0 71327 .coefficient, .predecessor 1 71328 .coefficient])

def exact71330RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact71330RawTermsValid :
    exact71330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7312⟩⟩) exact71330RawTerms .large 71329 .exactZero (none)

def event71331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 0 ⟨7312⟩ 71330

def event71332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 1 ⟨7208⟩ 71299

def event71333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7313⟩⟩) (.sum [.predecessor 0 71331 .coefficient, .predecessor 1 71332 .coefficient])

def exact71334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact71334RawTermsValid :
    exact71334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7313⟩⟩) exact71334RawTerms .large 71333 .exactZero (none)

def event71335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 0 ⟨7313⟩ 71334

def event71336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 1 ⟨7210⟩ 71296

def event71337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7314⟩⟩) (.sum [.predecessor 0 71335 .coefficient, .predecessor 1 71336 .coefficient])

def exact71338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact71338RawTermsValid :
    exact71338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7314⟩⟩) exact71338RawTerms .large 71337 .exactZero (none)

def event71339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 0 ⟨7314⟩ 71338

def event71340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 1 ⟨7212⟩ 71293

def event71341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7315⟩⟩) (.sum [.predecessor 0 71339 .coefficient, .predecessor 1 71340 .coefficient])

def exact71342RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact71342RawTermsValid :
    exact71342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7315⟩⟩) exact71342RawTerms .large 71341 .exactZero (none)

def event71343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 0 ⟨7315⟩ 71342

def event71344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 1 ⟨7214⟩ 71290

def event71345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7316⟩⟩) (.sum [.predecessor 0 71343 .coefficient, .predecessor 1 71344 .coefficient])

def exact71346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact71346RawTermsValid :
    exact71346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7316⟩⟩) exact71346RawTerms .large 71345 .exactZero (none)

def event71347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 0 ⟨7316⟩ 71346

def event71348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 1 ⟨7216⟩ 71287

def event71349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7317⟩⟩) (.sum [.predecessor 0 71347 .coefficient, .predecessor 1 71348 .coefficient])

def exact71350RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact71350RawTermsValid :
    exact71350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71350 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7317⟩⟩) exact71350RawTerms .large 71349 .exactZero (none)

def event71351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 0 ⟨7317⟩ 71350

def event71352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 1 ⟨7218⟩ 71284

def event71353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7318⟩⟩) (.sum [.predecessor 0 71351 .coefficient, .predecessor 1 71352 .coefficient])

def exact71354RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact71354RawTermsValid :
    exact71354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7318⟩⟩) exact71354RawTerms .large 71353 .exactZero (none)

def event71355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 0 ⟨7318⟩ 71354

def event71356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 1 ⟨7220⟩ 71281

def event71357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7319⟩⟩) (.sum [.predecessor 0 71355 .coefficient, .predecessor 1 71356 .coefficient])

def exact71358RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact71358RawTermsValid :
    exact71358RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71358 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7319⟩⟩) exact71358RawTerms .large 71357 .exactZero (none)

def event71359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 0 ⟨7319⟩ 71358

def event71360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 1 ⟨7222⟩ 71278

def event71361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7320⟩⟩) (.sum [.predecessor 0 71359 .coefficient, .predecessor 1 71360 .coefficient])

def exact71362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact71362RawTermsValid :
    exact71362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7320⟩⟩) exact71362RawTerms .large 71361 .exactZero (none)

def event71363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 0 ⟨7320⟩ 71362

def event71364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 1 ⟨7224⟩ 71275

def event71365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7321⟩⟩) (.sum [.predecessor 0 71363 .coefficient, .predecessor 1 71364 .coefficient])

def exact71366RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact71366RawTermsValid :
    exact71366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7321⟩⟩) exact71366RawTerms .large 71365 .exactZero (none)

def event71367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 0 ⟨7321⟩ 71366

def event71368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 1 ⟨7226⟩ 71272

def event71369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7322⟩⟩) (.sum [.predecessor 0 71367 .coefficient, .predecessor 1 71368 .coefficient])

def exact71370RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact71370RawTermsValid :
    exact71370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71370 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7322⟩⟩) exact71370RawTerms .large 71369 .exactZero (none)

def event71371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 0 ⟨7322⟩ 71370

def event71372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 1 ⟨7228⟩ 71269

def event71373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7323⟩⟩) (.sum [.predecessor 0 71371 .coefficient, .predecessor 1 71372 .coefficient])

def exact71374RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact71374RawTermsValid :
    exact71374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71374 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7323⟩⟩) exact71374RawTerms .large 71373 .exactZero (none)

def event71375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 0 ⟨7323⟩ 71374

def event71376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 1 ⟨7230⟩ 71266

def event71377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7324⟩⟩) (.sum [.predecessor 0 71375 .coefficient, .predecessor 1 71376 .coefficient])

def exact71378RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact71378RawTermsValid :
    exact71378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7324⟩⟩) exact71378RawTerms .large 71377 .exactZero (none)

def event71379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 0 ⟨7324⟩ 71378

def event71380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 1 ⟨7232⟩ 71263

def event71381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7325⟩⟩) (.sum [.predecessor 0 71379 .coefficient, .predecessor 1 71380 .coefficient])

def exact71382RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact71382RawTermsValid :
    exact71382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7325⟩⟩) exact71382RawTerms .large 71381 .exactZero (none)

def event71383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69118⟩⟩) 0 ⟨7325⟩ 71382

def event71384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69118⟩⟩) 1 ⟨69117⟩ 71260

def event71385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69118⟩⟩) (.sum [.predecessor 0 71383 .coefficient, .predecessor 1 71384 .coefficient])

def exact71386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact71386RawTermsValid :
    exact71386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71386 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69118⟩⟩) exact71386RawTerms .large 71385 .exactZero (none)

def event71387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71470⟩⟩) 0 ⟨69118⟩ 71386

def event71388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71470⟩⟩) 1 ⟨71469⟩ 71227

def event71389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71470⟩⟩) (.product (.predecessor 0 71387 .coefficient) (.predecessor 1 71388 .coefficient) (⟨false, false, none, none, none⟩))

def event71390 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 17⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71391 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 16⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71392 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 15⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71393 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 14⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71394 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 13⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71395 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 12⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71396 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 11⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71397 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 10⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71398 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 9⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71399 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 8⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 7⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71401 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 6⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 5⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71403 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 4⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71404 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 3⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71405 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 2⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71406 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 1⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71407 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 0⟩, ⟨71227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event71408 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 29⟩, ⟨71227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event71409 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71470⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 71224)

def event71410 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .relation 71409 0, ⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event71411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 28⟩, ⟨71227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event71412 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71470⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 71224)

def event71413 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .relation 71412 0, ⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event71414 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 27⟩, ⟨71227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event71415 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71470⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 71224)

def event71416 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .relation 71415 0, ⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event71417 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 26⟩, ⟨71227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event71418 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71470⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 71224)

def event71419 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .relation 71418 0, ⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event71420 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 25⟩, ⟨71227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event71421 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71470⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 71224)

def event71422 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .relation 71421 0, ⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event71423 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71470⟩⟩, .operator (⟨71386, 24⟩, ⟨71227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def eventLeaf4448 : Array AnnotatedEvent := #[
  { event := event71168
    frameStart := 70711 },
  { event := event71169
    frameStart := 70711 },
  { event := event71170
    frameStart := 70711 },
  { event := event71171
    frameStart := 70711 },
  { event := event71172
    frameStart := 70711 },
  { event := event71173
    frameStart := 70711 },
  { event := event71174
    frameStart := 70711 },
  { event := event71175
    frameStart := 70711 },
  { event := event71176
    frameStart := 70711 },
  { event := event71177
    frameStart := 70711 },
  { event := event71178
    frameStart := 70711 },
  { event := event71179
    frameStart := 70711 },
  { event := event71180
    frameStart := 70711 },
  { event := event71181
    frameStart := 70711 },
  { event := event71182
    frameStart := 70711 },
  { event := event71183
    frameStart := 70711 }
]

def eventLeaf4449 : Array AnnotatedEvent := #[
  { event := event71184
    frameStart := 70711 },
  { event := event71185
    frameStart := 70711 },
  { event := event71186
    frameStart := 70711 },
  { event := event71187
    frameStart := 70711 },
  { event := event71188
    frameStart := 70711 },
  { event := event71189
    frameStart := 70711 },
  { event := event71190
    frameStart := 70711 },
  { event := event71191
    frameStart := 70711 },
  { event := event71192
    frameStart := 70711 },
  { event := event71193
    frameStart := 70711 },
  { event := event71194
    frameStart := 70711 },
  { event := event71195
    frameStart := 70711 },
  { event := event71196
    frameStart := 70711 },
  { event := event71197
    frameStart := 70711 },
  { event := event71198
    frameStart := 70711 },
  { event := event71199
    frameStart := 70711 }
]

def eventLeaf4450 : Array AnnotatedEvent := #[
  { event := event71200
    frameStart := 70711 },
  { event := event71201
    frameStart := 70711 },
  { event := event71202
    frameStart := 70711 },
  { event := event71203
    frameStart := 70711 },
  { event := event71204
    frameStart := 70711 },
  { event := event71205
    frameStart := 70711 },
  { event := event71206
    frameStart := 70711 },
  { event := event71207
    frameStart := 70711 },
  { event := event71208
    frameStart := 70711 },
  { event := event71209
    frameStart := 70711 },
  { event := event71210
    frameStart := 70711 },
  { event := event71211
    frameStart := 70711 },
  { event := event71212
    frameStart := 70711 },
  { event := event71213
    frameStart := 70711 },
  { event := event71214
    frameStart := 70711 },
  { event := event71215
    frameStart := 70711 }
]

def eventLeaf4451 : Array AnnotatedEvent := #[
  { event := event71216
    frameStart := 70711 },
  { event := event71217
    frameStart := 70711 },
  { event := event71218
    frameStart := 70711 },
  { event := event71219
    frameStart := 70711 },
  { event := event71220
    frameStart := 70711 },
  { event := event71221
    frameStart := 70711 },
  { event := event71222
    frameStart := 70711 },
  { event := event71223
    frameStart := 70711 },
  { event := event71224
    frameStart := 70711 },
  { event := event71225
    frameStart := 70711 },
  { event := event71226
    frameStart := 70711 },
  { event := event71227
    frameStart := 70711 },
  { event := event71228
    frameStart := 70711 },
  { event := event71229
    frameStart := 70711 },
  { event := event71230
    frameStart := 70711 },
  { event := event71231
    frameStart := 70711 }
]

def eventLeaf4452 : Array AnnotatedEvent := #[
  { event := event71232
    frameStart := 70711 },
  { event := event71233
    frameStart := 70711 },
  { event := event71234
    frameStart := 70711 },
  { event := event71235
    frameStart := 70711 },
  { event := event71236
    frameStart := 70711 },
  { event := event71237
    frameStart := 70711 },
  { event := event71238
    frameStart := 70711 },
  { event := event71239
    frameStart := 70711 },
  { event := event71240
    frameStart := 70711 },
  { event := event71241
    frameStart := 70711 },
  { event := event71242
    frameStart := 70711 },
  { event := event71243
    frameStart := 70711 },
  { event := event71244
    frameStart := 70711 },
  { event := event71245
    frameStart := 70711 },
  { event := event71246
    frameStart := 70711 },
  { event := event71247
    frameStart := 70711 }
]

def eventLeaf4453 : Array AnnotatedEvent := #[
  { event := event71248
    frameStart := 70711 },
  { event := event71249
    frameStart := 70711 },
  { event := event71250
    frameStart := 70711 },
  { event := event71251
    frameStart := 70711 },
  { event := event71252
    frameStart := 70711 },
  { event := event71253
    frameStart := 70711 },
  { event := event71254
    frameStart := 70711 },
  { event := event71255
    frameStart := 70711 },
  { event := event71256
    frameStart := 70711 },
  { event := event71257
    frameStart := 70711 },
  { event := event71258
    frameStart := 70711 },
  { event := event71259
    frameStart := 70711 },
  { event := event71260
    frameStart := 70711 },
  { event := event71261
    frameStart := 70711 },
  { event := event71262
    frameStart := 70711 },
  { event := event71263
    frameStart := 70711 }
]

def eventLeaf4454 : Array AnnotatedEvent := #[
  { event := event71264
    frameStart := 70711 },
  { event := event71265
    frameStart := 70711 },
  { event := event71266
    frameStart := 70711 },
  { event := event71267
    frameStart := 70711 },
  { event := event71268
    frameStart := 70711 },
  { event := event71269
    frameStart := 70711 },
  { event := event71270
    frameStart := 70711 },
  { event := event71271
    frameStart := 70711 },
  { event := event71272
    frameStart := 70711 },
  { event := event71273
    frameStart := 70711 },
  { event := event71274
    frameStart := 70711 },
  { event := event71275
    frameStart := 70711 },
  { event := event71276
    frameStart := 70711 },
  { event := event71277
    frameStart := 70711 },
  { event := event71278
    frameStart := 70711 },
  { event := event71279
    frameStart := 70711 }
]

def eventLeaf4455 : Array AnnotatedEvent := #[
  { event := event71280
    frameStart := 70711 },
  { event := event71281
    frameStart := 70711 },
  { event := event71282
    frameStart := 70711 },
  { event := event71283
    frameStart := 70711 },
  { event := event71284
    frameStart := 70711 },
  { event := event71285
    frameStart := 70711 },
  { event := event71286
    frameStart := 70711 },
  { event := event71287
    frameStart := 70711 },
  { event := event71288
    frameStart := 70711 },
  { event := event71289
    frameStart := 70711 },
  { event := event71290
    frameStart := 70711 },
  { event := event71291
    frameStart := 70711 },
  { event := event71292
    frameStart := 70711 },
  { event := event71293
    frameStart := 70711 },
  { event := event71294
    frameStart := 70711 },
  { event := event71295
    frameStart := 70711 }
]

def eventLeaf4456 : Array AnnotatedEvent := #[
  { event := event71296
    frameStart := 70711 },
  { event := event71297
    frameStart := 70711 },
  { event := event71298
    frameStart := 70711 },
  { event := event71299
    frameStart := 70711 },
  { event := event71300
    frameStart := 70711 },
  { event := event71301
    frameStart := 70711 },
  { event := event71302
    frameStart := 70711 },
  { event := event71303
    frameStart := 70711 },
  { event := event71304
    frameStart := 70711 },
  { event := event71305
    frameStart := 70711 },
  { event := event71306
    frameStart := 70711 },
  { event := event71307
    frameStart := 70711 },
  { event := event71308
    frameStart := 70711 },
  { event := event71309
    frameStart := 70711 },
  { event := event71310
    frameStart := 70711 },
  { event := event71311
    frameStart := 70711 }
]

def eventLeaf4457 : Array AnnotatedEvent := #[
  { event := event71312
    frameStart := 70711 },
  { event := event71313
    frameStart := 70711 },
  { event := event71314
    frameStart := 70711 },
  { event := event71315
    frameStart := 70711 },
  { event := event71316
    frameStart := 70711 },
  { event := event71317
    frameStart := 70711 },
  { event := event71318
    frameStart := 70711 },
  { event := event71319
    frameStart := 70711 },
  { event := event71320
    frameStart := 70711 },
  { event := event71321
    frameStart := 70711 },
  { event := event71322
    frameStart := 70711 },
  { event := event71323
    frameStart := 70711 },
  { event := event71324
    frameStart := 70711 },
  { event := event71325
    frameStart := 70711 },
  { event := event71326
    frameStart := 70711 },
  { event := event71327
    frameStart := 70711 }
]

def eventLeaf4458 : Array AnnotatedEvent := #[
  { event := event71328
    frameStart := 70711 },
  { event := event71329
    frameStart := 70711 },
  { event := event71330
    frameStart := 70711 },
  { event := event71331
    frameStart := 70711 },
  { event := event71332
    frameStart := 70711 },
  { event := event71333
    frameStart := 70711 },
  { event := event71334
    frameStart := 70711 },
  { event := event71335
    frameStart := 70711 },
  { event := event71336
    frameStart := 70711 },
  { event := event71337
    frameStart := 70711 },
  { event := event71338
    frameStart := 70711 },
  { event := event71339
    frameStart := 70711 },
  { event := event71340
    frameStart := 70711 },
  { event := event71341
    frameStart := 70711 },
  { event := event71342
    frameStart := 70711 },
  { event := event71343
    frameStart := 70711 }
]

def eventLeaf4459 : Array AnnotatedEvent := #[
  { event := event71344
    frameStart := 70711 },
  { event := event71345
    frameStart := 70711 },
  { event := event71346
    frameStart := 70711 },
  { event := event71347
    frameStart := 70711 },
  { event := event71348
    frameStart := 70711 },
  { event := event71349
    frameStart := 70711 },
  { event := event71350
    frameStart := 70711 },
  { event := event71351
    frameStart := 70711 },
  { event := event71352
    frameStart := 70711 },
  { event := event71353
    frameStart := 70711 },
  { event := event71354
    frameStart := 70711 },
  { event := event71355
    frameStart := 70711 },
  { event := event71356
    frameStart := 70711 },
  { event := event71357
    frameStart := 70711 },
  { event := event71358
    frameStart := 70711 },
  { event := event71359
    frameStart := 70711 }
]

def eventLeaf4460 : Array AnnotatedEvent := #[
  { event := event71360
    frameStart := 70711 },
  { event := event71361
    frameStart := 70711 },
  { event := event71362
    frameStart := 70711 },
  { event := event71363
    frameStart := 70711 },
  { event := event71364
    frameStart := 70711 },
  { event := event71365
    frameStart := 70711 },
  { event := event71366
    frameStart := 70711 },
  { event := event71367
    frameStart := 70711 },
  { event := event71368
    frameStart := 70711 },
  { event := event71369
    frameStart := 70711 },
  { event := event71370
    frameStart := 70711 },
  { event := event71371
    frameStart := 70711 },
  { event := event71372
    frameStart := 70711 },
  { event := event71373
    frameStart := 70711 },
  { event := event71374
    frameStart := 70711 },
  { event := event71375
    frameStart := 70711 }
]

def eventLeaf4461 : Array AnnotatedEvent := #[
  { event := event71376
    frameStart := 70711 },
  { event := event71377
    frameStart := 70711 },
  { event := event71378
    frameStart := 70711 },
  { event := event71379
    frameStart := 70711 },
  { event := event71380
    frameStart := 70711 },
  { event := event71381
    frameStart := 70711 },
  { event := event71382
    frameStart := 70711 },
  { event := event71383
    frameStart := 70711 },
  { event := event71384
    frameStart := 70711 },
  { event := event71385
    frameStart := 70711 },
  { event := event71386
    frameStart := 70711 },
  { event := event71387
    frameStart := 70711 },
  { event := event71388
    frameStart := 70711 },
  { event := event71389
    frameStart := 70711 },
  { event := event71390
    frameStart := 70711 },
  { event := event71391
    frameStart := 70711 }
]

def eventLeaf4462 : Array AnnotatedEvent := #[
  { event := event71392
    frameStart := 70711 },
  { event := event71393
    frameStart := 70711 },
  { event := event71394
    frameStart := 70711 },
  { event := event71395
    frameStart := 70711 },
  { event := event71396
    frameStart := 70711 },
  { event := event71397
    frameStart := 70711 },
  { event := event71398
    frameStart := 70711 },
  { event := event71399
    frameStart := 70711 },
  { event := event71400
    frameStart := 70711 },
  { event := event71401
    frameStart := 70711 },
  { event := event71402
    frameStart := 70711 },
  { event := event71403
    frameStart := 70711 },
  { event := event71404
    frameStart := 70711 },
  { event := event71405
    frameStart := 70711 },
  { event := event71406
    frameStart := 70711 },
  { event := event71407
    frameStart := 70711 }
]

def eventLeaf4463 : Array AnnotatedEvent := #[
  { event := event71408
    frameStart := 70711 },
  { event := event71409
    frameStart := 70711 },
  { event := event71410
    frameStart := 70711 },
  { event := event71411
    frameStart := 70711 },
  { event := event71412
    frameStart := 70711 },
  { event := event71413
    frameStart := 70711 },
  { event := event71414
    frameStart := 70711 },
  { event := event71415
    frameStart := 70711 },
  { event := event71416
    frameStart := 70711 },
  { event := event71417
    frameStart := 70711 },
  { event := event71418
    frameStart := 70711 },
  { event := event71419
    frameStart := 70711 },
  { event := event71420
    frameStart := 70711 },
  { event := event71421
    frameStart := 70711 },
  { event := event71422
    frameStart := 70711 },
  { event := event71423
    frameStart := 70711 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events278
