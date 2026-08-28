import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events735

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event188160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51219⟩⟩) (.sum [.predecessor 0 188158 .coefficient, .predecessor 1 188159 .coefficient])

def exact188161RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩]

theorem exact188161RawTermsValid :
    exact188161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188161 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51219⟩⟩) exact188161RawTerms (.finite 255) 188160 .exactZero (none)

def event188162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54199⟩⟩) 0 ⟨51219⟩ 188161

def event188163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54199⟩⟩) 1 ⟨54198⟩ 188030

def event188164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54199⟩⟩) (.sum [.predecessor 0 188162 .coefficient, .predecessor 1 188163 .coefficient])

def exact188165RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩]

theorem exact188165RawTermsValid :
    exact188165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188165 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54199⟩⟩) exact188165RawTerms (.finite 314) 188164 .exactZero (none)

def event188166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57179⟩⟩) 0 ⟨54199⟩ 188165

def event188167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57179⟩⟩) 1 ⟨57178⟩ 188007

def event188168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57179⟩⟩) (.sum [.predecessor 0 188166 .coefficient, .predecessor 1 188167 .coefficient])

def exact188169RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩]

theorem exact188169RawTermsValid :
    exact188169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188169 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57179⟩⟩) exact188169RawTerms (.finite 374) 188168 .exactZero (none)

def event188170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60159⟩⟩) 0 ⟨57179⟩ 188169

def event188171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60159⟩⟩) 1 ⟨60158⟩ 187984

def event188172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60159⟩⟩) (.sum [.predecessor 0 188170 .coefficient, .predecessor 1 188171 .coefficient])

def exact188173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩]

theorem exact188173RawTermsValid :
    exact188173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60159⟩⟩) exact188173RawTerms (.finite 435) 188172 .exactZero (none)

def event188174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63139⟩⟩) 0 ⟨60159⟩ 188173

def event188175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63139⟩⟩) 1 ⟨63138⟩ 187961

def event188176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63139⟩⟩) (.sum [.predecessor 0 188174 .coefficient, .predecessor 1 188175 .coefficient])

def exact188177RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩]

theorem exact188177RawTermsValid :
    exact188177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188177 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63139⟩⟩) exact188177RawTerms (.finite 496) 188176 .exactZero (none)

def event188178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66812⟩⟩) 0 ⟨63139⟩ 188177

def event188179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66812⟩⟩) 1 ⟨66811⟩ 187938

def event188180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66812⟩⟩) (.sum [.predecessor 0 188178 .coefficient, .predecessor 1 188179 .coefficient])

def exact188181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188181RawTermsValid :
    exact188181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66812⟩⟩) exact188181RawTerms (.finite 558) 188180 .exactZero (none)

def event188182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66813⟩⟩) 0 ⟨66812⟩ 188181

def event188183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66813⟩⟩) 1 ⟨26658⟩ 187915

def event188184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66813⟩⟩) (.sum [.predecessor 0 188182 .coefficient, .predecessor 1 188183 .coefficient])

def exact188185RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188185RawTermsValid :
    exact188185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188185 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66813⟩⟩) exact188185RawTerms (.finite 620) 188184 .exactZero (none)

def event188186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66814⟩⟩) 0 ⟨66813⟩ 188185

def event188187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66814⟩⟩) 1 ⟨29338⟩ 187892

def event188188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66814⟩⟩) (.sum [.predecessor 0 188186 .coefficient, .predecessor 1 188187 .coefficient])

def exact188189RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188189RawTermsValid :
    exact188189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188189 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66814⟩⟩) exact188189RawTerms (.finite 682) 188188 .exactZero (none)

def event188190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66815⟩⟩) 0 ⟨66814⟩ 188189

def event188191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66815⟩⟩) 1 ⟨35002⟩ 187869

def event188192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66815⟩⟩) (.sum [.predecessor 0 188190 .coefficient, .predecessor 1 188191 .coefficient])

def exact188193RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188193RawTermsValid :
    exact188193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66815⟩⟩) exact188193RawTerms (.finite 744) 188192 .exactZero (none)

def event188194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66816⟩⟩) 0 ⟨66815⟩ 188193

def event188195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66816⟩⟩) 1 ⟨37682⟩ 187846

def event188196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66816⟩⟩) (.sum [.predecessor 0 188194 .coefficient, .predecessor 1 188195 .coefficient])

def exact188197RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188197RawTermsValid :
    exact188197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188197 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66816⟩⟩) exact188197RawTerms (.finite 807) 188196 .exactZero (none)

def event188198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66817⟩⟩) 0 ⟨66816⟩ 188197

def event188199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66817⟩⟩) 1 ⟨40358⟩ 187823

def event188200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66817⟩⟩) (.sum [.predecessor 0 188198 .coefficient, .predecessor 1 188199 .coefficient])

def exact188201RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188201RawTermsValid :
    exact188201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66817⟩⟩) exact188201RawTerms (.finite 870) 188200 .exactZero (none)

def event188202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66818⟩⟩) 0 ⟨66817⟩ 188201

def event188203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66818⟩⟩) 1 ⟨43038⟩ 187800

def event188204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66818⟩⟩) (.sum [.predecessor 0 188202 .coefficient, .predecessor 1 188203 .coefficient])

def exact188205RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188205RawTermsValid :
    exact188205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188205 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66818⟩⟩) exact188205RawTerms (.finite 933) 188204 .exactZero (none)

def event188206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66819⟩⟩) 0 ⟨66818⟩ 188205

def event188207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66819⟩⟩) 1 ⟨45722⟩ 187777

def event188208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66819⟩⟩) (.sum [.predecessor 0 188206 .coefficient, .predecessor 1 188207 .coefficient])

def exact188209RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188209RawTermsValid :
    exact188209RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188209 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66819⟩⟩) exact188209RawTerms (.finite 996) 188208 .exactZero (none)

def event188210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66820⟩⟩) 0 ⟨66819⟩ 188209

def event188211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66820⟩⟩) 1 ⟨48402⟩ 187754

def event188212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66820⟩⟩) (.sum [.predecessor 0 188210 .coefficient, .predecessor 1 188211 .coefficient])

def exact188213RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188213RawTermsValid :
    exact188213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188213 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66820⟩⟩) exact188213RawTerms (.finite 1059) 188212 .exactZero (none)

def event188214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66821⟩⟩) 0 ⟨66820⟩ 188213

def event188215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66821⟩⟩) (.identity (.predecessor 0 188214 .coefficient))

def event188216 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨66821⟩⟩) (.finite 1059)

def event188217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68847⟩⟩) 0 ⟨66821⟩ 188216

def event188218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68847⟩⟩) (.authority (.programFamilyFact))

def event188219 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68847⟩⟩) (.finite 1152)

def event188220 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event188221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68848⟩⟩) 0 ⟨7177⟩ 188220

def event188222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68848⟩⟩) 1 ⟨68847⟩ 188219

def event188223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68848⟩⟩) (.authority (.operator))

def exact188224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (1)⟩]

theorem exact188224RawTermsValid :
    exact188224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68848⟩⟩) exact188224RawTerms .large 188223 .exactZero (none)

def event188225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71329⟩⟩) 0 ⟨68848⟩ 188224

def event188226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71329⟩⟩) (.authority (.operator))

def exact188227RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩]

theorem exact188227RawTermsValid :
    exact188227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71329⟩⟩) exact188227RawTerms (.finite 8192) 188226 .exactZero (none)

def event188228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event188229 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event188230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69099⟩⟩) 0 ⟨66821⟩ 188216

def event188231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69099⟩⟩) 1 ⟨136⟩ 188229

def event188232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69099⟩⟩) (.sum [.predecessor 0 188230 .coefficient, .predecessor 1 188231 .coefficient])

def event188233 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69099⟩⟩) (.finite 1059)

def event188234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69100⟩⟩) 0 ⟨69099⟩ 188233

def event188235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69100⟩⟩) (.identity (.predecessor 0 188234 .coefficient))

def exact188236RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], []⟩, (1)⟩]

theorem exact188236RawTermsValid :
    exact188236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69100⟩⟩) exact188236RawTerms (.finite 1059) 188235 .exactZero (none)

def event188237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact188238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact188238RawTermsValid :
    exact188238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact188238RawTerms .large 188237 .exactZero (none)

def event188239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69101⟩⟩) 0 ⟨6908⟩ 188238

def event188240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69101⟩⟩) 1 ⟨69100⟩ 188236

def event188241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69101⟩⟩) (.product (.predecessor 0 188239 .coefficient) (.predecessor 1 188240 .coefficient) (⟨false, false, none, none, none⟩))

def event188242 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188243 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188244 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188245 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188246 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188247 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188248 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188249 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188250 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188251 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188252 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188253 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188254 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188255 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188257 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188258 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event188259 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69101⟩⟩, .operator (⟨188238, 0⟩, ⟨188236, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact188260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact188260RawTermsValid :
    exact188260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69101⟩⟩) exact188260RawTerms .large 188241 .exactZero (none)

def event188261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 188220

def event188262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact188263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact188263RawTermsValid :
    exact188263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact188263RawTerms .large 188262 .exactZero (none)

def event188264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 188220

def event188265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact188266RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact188266RawTermsValid :
    exact188266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact188266RawTerms .large 188265 .exactZero (none)

def event188267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 188220

def event188268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact188269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact188269RawTermsValid :
    exact188269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188269 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact188269RawTerms .large 188268 .exactZero (none)

def event188270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 188220

def event188271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact188272RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact188272RawTermsValid :
    exact188272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact188272RawTerms .large 188271 .exactZero (none)

def event188273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 188220

def event188274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact188275RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact188275RawTermsValid :
    exact188275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188275 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact188275RawTerms .large 188274 .exactZero (none)

def event188276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7222⟩⟩) 0 ⟨7177⟩ 188220

def event188277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7222⟩⟩) (.authority (.operator))

def exact188278RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact188278RawTermsValid :
    exact188278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188278 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7222⟩⟩) exact188278RawTerms .large 188277 .exactZero (none)

def event188279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 188220

def event188280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact188281RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact188281RawTermsValid :
    exact188281RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188281 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact188281RawTerms .large 188280 .exactZero (none)

def event188282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 188220

def event188283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact188284RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact188284RawTermsValid :
    exact188284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact188284RawTerms .large 188283 .exactZero (none)

def event188285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 188220

def event188286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact188287RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact188287RawTermsValid :
    exact188287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188287 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact188287RawTerms .large 188286 .exactZero (none)

def event188288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7214⟩⟩) 0 ⟨7177⟩ 188220

def event188289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7214⟩⟩) (.authority (.operator))

def exact188290RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact188290RawTermsValid :
    exact188290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7214⟩⟩) exact188290RawTerms .large 188289 .exactZero (none)

def event188291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 188220

def event188292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact188293RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact188293RawTermsValid :
    exact188293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact188293RawTerms .large 188292 .exactZero (none)

def event188294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 188220

def event188295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact188296RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact188296RawTermsValid :
    exact188296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact188296RawTerms .large 188295 .exactZero (none)

def event188297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 188220

def event188298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact188299RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact188299RawTermsValid :
    exact188299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact188299RawTerms .large 188298 .exactZero (none)

def event188300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 188220

def event188301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact188302RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact188302RawTermsValid :
    exact188302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact188302RawTerms .large 188301 .exactZero (none)

def event188303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 188220

def event188304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact188305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact188305RawTermsValid :
    exact188305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact188305RawTerms .large 188304 .exactZero (none)

def event188306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7202⟩⟩) 0 ⟨7177⟩ 188220

def event188307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7202⟩⟩) (.authority (.operator))

def exact188308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact188308RawTermsValid :
    exact188308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7202⟩⟩) exact188308RawTerms .large 188307 .exactZero (none)

def event188309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 188220

def event188310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact188311RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact188311RawTermsValid :
    exact188311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188311 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact188311RawTerms .large 188310 .exactZero (none)

def event188312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 188220

def event188313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact188314RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact188314RawTermsValid :
    exact188314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact188314RawTerms .large 188313 .exactZero (none)

def event188315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 0 ⟨7198⟩ 188314

def event188316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 1 ⟨7200⟩ 188311

def event188317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7309⟩⟩) (.sum [.predecessor 0 188315 .coefficient, .predecessor 1 188316 .coefficient])

def exact188318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact188318RawTermsValid :
    exact188318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188318 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7309⟩⟩) exact188318RawTerms .large 188317 .exactZero (none)

def event188319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 0 ⟨7309⟩ 188318

def event188320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 1 ⟨7202⟩ 188308

def event188321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7310⟩⟩) (.sum [.predecessor 0 188319 .coefficient, .predecessor 1 188320 .coefficient])

def exact188322RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact188322RawTermsValid :
    exact188322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7310⟩⟩) exact188322RawTerms .large 188321 .exactZero (none)

def event188323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 0 ⟨7310⟩ 188322

def event188324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 1 ⟨7204⟩ 188305

def event188325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7311⟩⟩) (.sum [.predecessor 0 188323 .coefficient, .predecessor 1 188324 .coefficient])

def exact188326RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact188326RawTermsValid :
    exact188326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7311⟩⟩) exact188326RawTerms .large 188325 .exactZero (none)

def event188327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 0 ⟨7311⟩ 188326

def event188328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 1 ⟨7206⟩ 188302

def event188329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7312⟩⟩) (.sum [.predecessor 0 188327 .coefficient, .predecessor 1 188328 .coefficient])

def exact188330RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact188330RawTermsValid :
    exact188330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7312⟩⟩) exact188330RawTerms .large 188329 .exactZero (none)

def event188331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 0 ⟨7312⟩ 188330

def event188332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 1 ⟨7208⟩ 188299

def event188333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7313⟩⟩) (.sum [.predecessor 0 188331 .coefficient, .predecessor 1 188332 .coefficient])

def exact188334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact188334RawTermsValid :
    exact188334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7313⟩⟩) exact188334RawTerms .large 188333 .exactZero (none)

def event188335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 0 ⟨7313⟩ 188334

def event188336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 1 ⟨7210⟩ 188296

def event188337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7314⟩⟩) (.sum [.predecessor 0 188335 .coefficient, .predecessor 1 188336 .coefficient])

def exact188338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact188338RawTermsValid :
    exact188338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7314⟩⟩) exact188338RawTerms .large 188337 .exactZero (none)

def event188339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 0 ⟨7314⟩ 188338

def event188340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 1 ⟨7212⟩ 188293

def event188341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7315⟩⟩) (.sum [.predecessor 0 188339 .coefficient, .predecessor 1 188340 .coefficient])

def exact188342RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact188342RawTermsValid :
    exact188342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7315⟩⟩) exact188342RawTerms .large 188341 .exactZero (none)

def event188343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 0 ⟨7315⟩ 188342

def event188344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 1 ⟨7214⟩ 188290

def event188345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7316⟩⟩) (.sum [.predecessor 0 188343 .coefficient, .predecessor 1 188344 .coefficient])

def exact188346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact188346RawTermsValid :
    exact188346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7316⟩⟩) exact188346RawTerms .large 188345 .exactZero (none)

def event188347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 0 ⟨7316⟩ 188346

def event188348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 1 ⟨7216⟩ 188287

def event188349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7317⟩⟩) (.sum [.predecessor 0 188347 .coefficient, .predecessor 1 188348 .coefficient])

def exact188350RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact188350RawTermsValid :
    exact188350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188350 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7317⟩⟩) exact188350RawTerms .large 188349 .exactZero (none)

def event188351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 0 ⟨7317⟩ 188350

def event188352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 1 ⟨7218⟩ 188284

def event188353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7318⟩⟩) (.sum [.predecessor 0 188351 .coefficient, .predecessor 1 188352 .coefficient])

def exact188354RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact188354RawTermsValid :
    exact188354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7318⟩⟩) exact188354RawTerms .large 188353 .exactZero (none)

def event188355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 0 ⟨7318⟩ 188354

def event188356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 1 ⟨7220⟩ 188281

def event188357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7319⟩⟩) (.sum [.predecessor 0 188355 .coefficient, .predecessor 1 188356 .coefficient])

def exact188358RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact188358RawTermsValid :
    exact188358RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188358 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7319⟩⟩) exact188358RawTerms .large 188357 .exactZero (none)

def event188359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 0 ⟨7319⟩ 188358

def event188360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 1 ⟨7222⟩ 188278

def event188361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7320⟩⟩) (.sum [.predecessor 0 188359 .coefficient, .predecessor 1 188360 .coefficient])

def exact188362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact188362RawTermsValid :
    exact188362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7320⟩⟩) exact188362RawTerms .large 188361 .exactZero (none)

def event188363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 0 ⟨7320⟩ 188362

def event188364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 1 ⟨7224⟩ 188275

def event188365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7321⟩⟩) (.sum [.predecessor 0 188363 .coefficient, .predecessor 1 188364 .coefficient])

def exact188366RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact188366RawTermsValid :
    exact188366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7321⟩⟩) exact188366RawTerms .large 188365 .exactZero (none)

def event188367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 0 ⟨7321⟩ 188366

def event188368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 1 ⟨7226⟩ 188272

def event188369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7322⟩⟩) (.sum [.predecessor 0 188367 .coefficient, .predecessor 1 188368 .coefficient])

def exact188370RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact188370RawTermsValid :
    exact188370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188370 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7322⟩⟩) exact188370RawTerms .large 188369 .exactZero (none)

def event188371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 0 ⟨7322⟩ 188370

def event188372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 1 ⟨7228⟩ 188269

def event188373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7323⟩⟩) (.sum [.predecessor 0 188371 .coefficient, .predecessor 1 188372 .coefficient])

def exact188374RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact188374RawTermsValid :
    exact188374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188374 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7323⟩⟩) exact188374RawTerms .large 188373 .exactZero (none)

def event188375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 0 ⟨7323⟩ 188374

def event188376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 1 ⟨7230⟩ 188266

def event188377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7324⟩⟩) (.sum [.predecessor 0 188375 .coefficient, .predecessor 1 188376 .coefficient])

def exact188378RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact188378RawTermsValid :
    exact188378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7324⟩⟩) exact188378RawTerms .large 188377 .exactZero (none)

def event188379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 0 ⟨7324⟩ 188378

def event188380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 1 ⟨7232⟩ 188263

def event188381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7325⟩⟩) (.sum [.predecessor 0 188379 .coefficient, .predecessor 1 188380 .coefficient])

def exact188382RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact188382RawTermsValid :
    exact188382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7325⟩⟩) exact188382RawTerms .large 188381 .exactZero (none)

def event188383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69102⟩⟩) 0 ⟨7325⟩ 188382

def event188384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69102⟩⟩) 1 ⟨69101⟩ 188260

def event188385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69102⟩⟩) (.sum [.predecessor 0 188383 .coefficient, .predecessor 1 188384 .coefficient])

def exact188386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact188386RawTermsValid :
    exact188386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event188386 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69102⟩⟩) exact188386RawTerms .large 188385 .exactZero (none)

def event188387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71330⟩⟩) 0 ⟨69102⟩ 188386

def event188388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71330⟩⟩) 1 ⟨71329⟩ 188227

def event188389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71330⟩⟩) (.product (.predecessor 0 188387 .coefficient) (.predecessor 1 188388 .coefficient) (⟨false, false, none, none, none⟩))

def event188390 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 17⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188391 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 16⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188392 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 15⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188393 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 14⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188394 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 13⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188395 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 12⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188396 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 11⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188397 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 10⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188398 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 9⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188399 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 8⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 7⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188401 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 6⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 5⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188403 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 4⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188404 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 3⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188405 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 2⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188406 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 1⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188407 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 0⟩, ⟨188227, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event188408 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 29⟩, ⟨188227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event188409 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71330⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224)

def event188410 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .relation 188409 0, ⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event188411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 28⟩, ⟨188227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event188412 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71330⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224)

def event188413 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .relation 188412 0, ⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event188414 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71330⟩⟩, .operator (⟨188386, 27⟩, ⟨188227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event188415 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71330⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224)

def eventLeaf11760 : Array AnnotatedEvent := #[
  { event := event188160
    frameStart := 187711 },
  { event := event188161
    frameStart := 187711 },
  { event := event188162
    frameStart := 187711 },
  { event := event188163
    frameStart := 187711 },
  { event := event188164
    frameStart := 187711 },
  { event := event188165
    frameStart := 187711 },
  { event := event188166
    frameStart := 187711 },
  { event := event188167
    frameStart := 187711 },
  { event := event188168
    frameStart := 187711 },
  { event := event188169
    frameStart := 187711 },
  { event := event188170
    frameStart := 187711 },
  { event := event188171
    frameStart := 187711 },
  { event := event188172
    frameStart := 187711 },
  { event := event188173
    frameStart := 187711 },
  { event := event188174
    frameStart := 187711 },
  { event := event188175
    frameStart := 187711 }
]

def eventLeaf11761 : Array AnnotatedEvent := #[
  { event := event188176
    frameStart := 187711 },
  { event := event188177
    frameStart := 187711 },
  { event := event188178
    frameStart := 187711 },
  { event := event188179
    frameStart := 187711 },
  { event := event188180
    frameStart := 187711 },
  { event := event188181
    frameStart := 187711 },
  { event := event188182
    frameStart := 187711 },
  { event := event188183
    frameStart := 187711 },
  { event := event188184
    frameStart := 187711 },
  { event := event188185
    frameStart := 187711 },
  { event := event188186
    frameStart := 187711 },
  { event := event188187
    frameStart := 187711 },
  { event := event188188
    frameStart := 187711 },
  { event := event188189
    frameStart := 187711 },
  { event := event188190
    frameStart := 187711 },
  { event := event188191
    frameStart := 187711 }
]

def eventLeaf11762 : Array AnnotatedEvent := #[
  { event := event188192
    frameStart := 187711 },
  { event := event188193
    frameStart := 187711 },
  { event := event188194
    frameStart := 187711 },
  { event := event188195
    frameStart := 187711 },
  { event := event188196
    frameStart := 187711 },
  { event := event188197
    frameStart := 187711 },
  { event := event188198
    frameStart := 187711 },
  { event := event188199
    frameStart := 187711 },
  { event := event188200
    frameStart := 187711 },
  { event := event188201
    frameStart := 187711 },
  { event := event188202
    frameStart := 187711 },
  { event := event188203
    frameStart := 187711 },
  { event := event188204
    frameStart := 187711 },
  { event := event188205
    frameStart := 187711 },
  { event := event188206
    frameStart := 187711 },
  { event := event188207
    frameStart := 187711 }
]

def eventLeaf11763 : Array AnnotatedEvent := #[
  { event := event188208
    frameStart := 187711 },
  { event := event188209
    frameStart := 187711 },
  { event := event188210
    frameStart := 187711 },
  { event := event188211
    frameStart := 187711 },
  { event := event188212
    frameStart := 187711 },
  { event := event188213
    frameStart := 187711 },
  { event := event188214
    frameStart := 187711 },
  { event := event188215
    frameStart := 187711 },
  { event := event188216
    frameStart := 187711 },
  { event := event188217
    frameStart := 187711 },
  { event := event188218
    frameStart := 187711 },
  { event := event188219
    frameStart := 187711 },
  { event := event188220
    frameStart := 187711 },
  { event := event188221
    frameStart := 187711 },
  { event := event188222
    frameStart := 187711 },
  { event := event188223
    frameStart := 187711 }
]

def eventLeaf11764 : Array AnnotatedEvent := #[
  { event := event188224
    frameStart := 187711 },
  { event := event188225
    frameStart := 187711 },
  { event := event188226
    frameStart := 187711 },
  { event := event188227
    frameStart := 187711 },
  { event := event188228
    frameStart := 187711 },
  { event := event188229
    frameStart := 187711 },
  { event := event188230
    frameStart := 187711 },
  { event := event188231
    frameStart := 187711 },
  { event := event188232
    frameStart := 187711 },
  { event := event188233
    frameStart := 187711 },
  { event := event188234
    frameStart := 187711 },
  { event := event188235
    frameStart := 187711 },
  { event := event188236
    frameStart := 187711 },
  { event := event188237
    frameStart := 187711 },
  { event := event188238
    frameStart := 187711 },
  { event := event188239
    frameStart := 187711 }
]

def eventLeaf11765 : Array AnnotatedEvent := #[
  { event := event188240
    frameStart := 187711 },
  { event := event188241
    frameStart := 187711 },
  { event := event188242
    frameStart := 187711 },
  { event := event188243
    frameStart := 187711 },
  { event := event188244
    frameStart := 187711 },
  { event := event188245
    frameStart := 187711 },
  { event := event188246
    frameStart := 187711 },
  { event := event188247
    frameStart := 187711 },
  { event := event188248
    frameStart := 187711 },
  { event := event188249
    frameStart := 187711 },
  { event := event188250
    frameStart := 187711 },
  { event := event188251
    frameStart := 187711 },
  { event := event188252
    frameStart := 187711 },
  { event := event188253
    frameStart := 187711 },
  { event := event188254
    frameStart := 187711 },
  { event := event188255
    frameStart := 187711 }
]

def eventLeaf11766 : Array AnnotatedEvent := #[
  { event := event188256
    frameStart := 187711 },
  { event := event188257
    frameStart := 187711 },
  { event := event188258
    frameStart := 187711 },
  { event := event188259
    frameStart := 187711 },
  { event := event188260
    frameStart := 187711 },
  { event := event188261
    frameStart := 187711 },
  { event := event188262
    frameStart := 187711 },
  { event := event188263
    frameStart := 187711 },
  { event := event188264
    frameStart := 187711 },
  { event := event188265
    frameStart := 187711 },
  { event := event188266
    frameStart := 187711 },
  { event := event188267
    frameStart := 187711 },
  { event := event188268
    frameStart := 187711 },
  { event := event188269
    frameStart := 187711 },
  { event := event188270
    frameStart := 187711 },
  { event := event188271
    frameStart := 187711 }
]

def eventLeaf11767 : Array AnnotatedEvent := #[
  { event := event188272
    frameStart := 187711 },
  { event := event188273
    frameStart := 187711 },
  { event := event188274
    frameStart := 187711 },
  { event := event188275
    frameStart := 187711 },
  { event := event188276
    frameStart := 187711 },
  { event := event188277
    frameStart := 187711 },
  { event := event188278
    frameStart := 187711 },
  { event := event188279
    frameStart := 187711 },
  { event := event188280
    frameStart := 187711 },
  { event := event188281
    frameStart := 187711 },
  { event := event188282
    frameStart := 187711 },
  { event := event188283
    frameStart := 187711 },
  { event := event188284
    frameStart := 187711 },
  { event := event188285
    frameStart := 187711 },
  { event := event188286
    frameStart := 187711 },
  { event := event188287
    frameStart := 187711 }
]

def eventLeaf11768 : Array AnnotatedEvent := #[
  { event := event188288
    frameStart := 187711 },
  { event := event188289
    frameStart := 187711 },
  { event := event188290
    frameStart := 187711 },
  { event := event188291
    frameStart := 187711 },
  { event := event188292
    frameStart := 187711 },
  { event := event188293
    frameStart := 187711 },
  { event := event188294
    frameStart := 187711 },
  { event := event188295
    frameStart := 187711 },
  { event := event188296
    frameStart := 187711 },
  { event := event188297
    frameStart := 187711 },
  { event := event188298
    frameStart := 187711 },
  { event := event188299
    frameStart := 187711 },
  { event := event188300
    frameStart := 187711 },
  { event := event188301
    frameStart := 187711 },
  { event := event188302
    frameStart := 187711 },
  { event := event188303
    frameStart := 187711 }
]

def eventLeaf11769 : Array AnnotatedEvent := #[
  { event := event188304
    frameStart := 187711 },
  { event := event188305
    frameStart := 187711 },
  { event := event188306
    frameStart := 187711 },
  { event := event188307
    frameStart := 187711 },
  { event := event188308
    frameStart := 187711 },
  { event := event188309
    frameStart := 187711 },
  { event := event188310
    frameStart := 187711 },
  { event := event188311
    frameStart := 187711 },
  { event := event188312
    frameStart := 187711 },
  { event := event188313
    frameStart := 187711 },
  { event := event188314
    frameStart := 187711 },
  { event := event188315
    frameStart := 187711 },
  { event := event188316
    frameStart := 187711 },
  { event := event188317
    frameStart := 187711 },
  { event := event188318
    frameStart := 187711 },
  { event := event188319
    frameStart := 187711 }
]

def eventLeaf11770 : Array AnnotatedEvent := #[
  { event := event188320
    frameStart := 187711 },
  { event := event188321
    frameStart := 187711 },
  { event := event188322
    frameStart := 187711 },
  { event := event188323
    frameStart := 187711 },
  { event := event188324
    frameStart := 187711 },
  { event := event188325
    frameStart := 187711 },
  { event := event188326
    frameStart := 187711 },
  { event := event188327
    frameStart := 187711 },
  { event := event188328
    frameStart := 187711 },
  { event := event188329
    frameStart := 187711 },
  { event := event188330
    frameStart := 187711 },
  { event := event188331
    frameStart := 187711 },
  { event := event188332
    frameStart := 187711 },
  { event := event188333
    frameStart := 187711 },
  { event := event188334
    frameStart := 187711 },
  { event := event188335
    frameStart := 187711 }
]

def eventLeaf11771 : Array AnnotatedEvent := #[
  { event := event188336
    frameStart := 187711 },
  { event := event188337
    frameStart := 187711 },
  { event := event188338
    frameStart := 187711 },
  { event := event188339
    frameStart := 187711 },
  { event := event188340
    frameStart := 187711 },
  { event := event188341
    frameStart := 187711 },
  { event := event188342
    frameStart := 187711 },
  { event := event188343
    frameStart := 187711 },
  { event := event188344
    frameStart := 187711 },
  { event := event188345
    frameStart := 187711 },
  { event := event188346
    frameStart := 187711 },
  { event := event188347
    frameStart := 187711 },
  { event := event188348
    frameStart := 187711 },
  { event := event188349
    frameStart := 187711 },
  { event := event188350
    frameStart := 187711 },
  { event := event188351
    frameStart := 187711 }
]

def eventLeaf11772 : Array AnnotatedEvent := #[
  { event := event188352
    frameStart := 187711 },
  { event := event188353
    frameStart := 187711 },
  { event := event188354
    frameStart := 187711 },
  { event := event188355
    frameStart := 187711 },
  { event := event188356
    frameStart := 187711 },
  { event := event188357
    frameStart := 187711 },
  { event := event188358
    frameStart := 187711 },
  { event := event188359
    frameStart := 187711 },
  { event := event188360
    frameStart := 187711 },
  { event := event188361
    frameStart := 187711 },
  { event := event188362
    frameStart := 187711 },
  { event := event188363
    frameStart := 187711 },
  { event := event188364
    frameStart := 187711 },
  { event := event188365
    frameStart := 187711 },
  { event := event188366
    frameStart := 187711 },
  { event := event188367
    frameStart := 187711 }
]

def eventLeaf11773 : Array AnnotatedEvent := #[
  { event := event188368
    frameStart := 187711 },
  { event := event188369
    frameStart := 187711 },
  { event := event188370
    frameStart := 187711 },
  { event := event188371
    frameStart := 187711 },
  { event := event188372
    frameStart := 187711 },
  { event := event188373
    frameStart := 187711 },
  { event := event188374
    frameStart := 187711 },
  { event := event188375
    frameStart := 187711 },
  { event := event188376
    frameStart := 187711 },
  { event := event188377
    frameStart := 187711 },
  { event := event188378
    frameStart := 187711 },
  { event := event188379
    frameStart := 187711 },
  { event := event188380
    frameStart := 187711 },
  { event := event188381
    frameStart := 187711 },
  { event := event188382
    frameStart := 187711 },
  { event := event188383
    frameStart := 187711 }
]

def eventLeaf11774 : Array AnnotatedEvent := #[
  { event := event188384
    frameStart := 187711 },
  { event := event188385
    frameStart := 187711 },
  { event := event188386
    frameStart := 187711 },
  { event := event188387
    frameStart := 187711 },
  { event := event188388
    frameStart := 187711 },
  { event := event188389
    frameStart := 187711 },
  { event := event188390
    frameStart := 187711 },
  { event := event188391
    frameStart := 187711 },
  { event := event188392
    frameStart := 187711 },
  { event := event188393
    frameStart := 187711 },
  { event := event188394
    frameStart := 187711 },
  { event := event188395
    frameStart := 187711 },
  { event := event188396
    frameStart := 187711 },
  { event := event188397
    frameStart := 187711 },
  { event := event188398
    frameStart := 187711 },
  { event := event188399
    frameStart := 187711 }
]

def eventLeaf11775 : Array AnnotatedEvent := #[
  { event := event188400
    frameStart := 187711 },
  { event := event188401
    frameStart := 187711 },
  { event := event188402
    frameStart := 187711 },
  { event := event188403
    frameStart := 187711 },
  { event := event188404
    frameStart := 187711 },
  { event := event188405
    frameStart := 187711 },
  { event := event188406
    frameStart := 187711 },
  { event := event188407
    frameStart := 187711 },
  { event := event188408
    frameStart := 187711 },
  { event := event188409
    frameStart := 187711 },
  { event := event188410
    frameStart := 187711 },
  { event := event188411
    frameStart := 187711 },
  { event := event188412
    frameStart := 187711 },
  { event := event188413
    frameStart := 187711 },
  { event := event188414
    frameStart := 187711 },
  { event := event188415
    frameStart := 187711 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events735
