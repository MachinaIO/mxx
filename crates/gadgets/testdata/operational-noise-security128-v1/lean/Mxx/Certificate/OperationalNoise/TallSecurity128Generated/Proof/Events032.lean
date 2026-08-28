import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events032

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact8192RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩]

theorem exact8192RawTermsValid :
    exact8192RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8192 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22158⟩⟩) exact8192RawTerms (.finite 187661410175051153573232) 8190 .exactZero (none)

def event8193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18937⟩⟩) 0 ⟨18621⟩ 7959

def event8194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18937⟩⟩) (.authority (.programFamilyFact))

def exact8195RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩]

theorem exact8195RawTermsValid :
    exact8195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8195 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18937⟩⟩) exact8195RawTerms (.finite 3) 8194 .exactZero (none)

def event8196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18938⟩⟩) 0 ⟨18937⟩ 8195

def event8197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18938⟩⟩) 1 ⟨6846⟩ 703

def event8198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18938⟩⟩) (.product (.predecessor 0 8196 .coefficient) (.predecessor 1 8197 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event8199 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18938⟩⟩, .operator (⟨8195, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩)

def exact8200RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩]

theorem exact8200RawTermsValid :
    exact8200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8200 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18938⟩⟩) exact8200RawTerms (.finite 175932572039110456474905) 8198 .exactZero (none)

def event8201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16094⟩⟩) 0 ⟨15821⟩ 7982

def event8202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16094⟩⟩) (.authority (.programFamilyFact))

def exact8203RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8203RawTermsValid :
    exact8203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8203 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16094⟩⟩) exact8203RawTerms (.finite 2) 8202 .exactZero (none)

def event8204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16095⟩⟩) 0 ⟨16094⟩ 8203

def event8205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16095⟩⟩) 1 ⟨6863⟩ 713

def event8206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16095⟩⟩) (.product (.predecessor 0 8204 .coefficient) (.predecessor 1 8205 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event8207 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16095⟩⟩, .operator (⟨8203, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩)

def exact8208RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8208RawTermsValid :
    exact8208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16095⟩⟩) exact8208RawTerms (.finite 156384508479209294644360) 8206 .exactZero (none)

def event8209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16096⟩⟩) 0 ⟨6728⟩ 728

def event8210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16096⟩⟩) 1 ⟨16095⟩ 8208

def event8211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16096⟩⟩) (.sum [.predecessor 0 8209 .coefficient, .predecessor 1 8210 .coefficient])

def exact8212RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8212RawTermsValid :
    exact8212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16096⟩⟩) exact8212RawTerms (.finite 156384508479209294644360) 8211 .exactZero (none)

def event8213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18939⟩⟩) 0 ⟨16096⟩ 8212

def event8214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18939⟩⟩) 1 ⟨18938⟩ 8200

def event8215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18939⟩⟩) (.sum [.predecessor 0 8213 .coefficient, .predecessor 1 8214 .coefficient])

def exact8216RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8216RawTermsValid :
    exact8216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8216 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18939⟩⟩) exact8216RawTerms (.finite 332317080518319751119265) 8215 .exactZero (none)

def event8217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22159⟩⟩) 0 ⟨18939⟩ 8216

def event8218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22159⟩⟩) 1 ⟨22158⟩ 8192

def event8219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22159⟩⟩) (.sum [.predecessor 0 8217 .coefficient, .predecessor 1 8218 .coefficient])

def exact8220RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8220RawTermsValid :
    exact8220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22159⟩⟩) exact8220RawTerms (.finite 519978490693370904692497) 8219 .exactZero (none)

def event8221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32179⟩⟩) 0 ⟨22159⟩ 8220

def event8222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32179⟩⟩) 1 ⟨32178⟩ 8184

def event8223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32179⟩⟩) (.sum [.predecessor 0 8221 .coefficient, .predecessor 1 8222 .coefficient])

def exact8224RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8224RawTermsValid :
    exact8224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32179⟩⟩) exact8224RawTerms (.finite 721044287309497140663817) 8223 .exactZero (none)

def event8225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51243⟩⟩) 0 ⟨32179⟩ 8224

def event8226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51243⟩⟩) 1 ⟨51242⟩ 8176

def event8227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51243⟩⟩) (.sum [.predecessor 0 8225 .coefficient, .predecessor 1 8226 .coefficient])

def exact8228RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8228RawTermsValid :
    exact8228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8228 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51243⟩⟩) exact8228RawTerms (.finite 934295889781146178815217) 8227 .exactZero (none)

def event8229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54223⟩⟩) 0 ⟨51243⟩ 8228

def event8230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54223⟩⟩) 1 ⟨54222⟩ 8168

def event8231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54223⟩⟩) (.sum [.predecessor 0 8229 .coefficient, .predecessor 1 8230 .coefficient])

def exact8232RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8232RawTermsValid :
    exact8232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54223⟩⟩) exact8232RawTerms (.finite 1150828286136974432938177) 8231 .exactZero (none)

def event8233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57203⟩⟩) 0 ⟨54223⟩ 8232

def event8234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57203⟩⟩) 1 ⟨57202⟩ 8160

def event8235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57203⟩⟩) (.sum [.predecessor 0 8233 .coefficient, .predecessor 1 8234 .coefficient])

def exact8236RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8236RawTermsValid :
    exact8236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57203⟩⟩) exact8236RawTerms (.finite 1371606415754681672436097) 8235 .exactZero (none)

def event8237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60183⟩⟩) 0 ⟨57203⟩ 8236

def event8238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60183⟩⟩) 1 ⟨60182⟩ 8152

def event8239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60183⟩⟩) (.sum [.predecessor 0 8237 .coefficient, .predecessor 1 8238 .coefficient])

def exact8240RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8240RawTermsValid :
    exact8240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8240 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60183⟩⟩) exact8240RawTerms (.finite 1593837033067242249035977) 8239 .exactZero (none)

def event8241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63163⟩⟩) 0 ⟨60183⟩ 8240

def event8242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63163⟩⟩) 1 ⟨63162⟩ 8144

def event8243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63163⟩⟩) (.sum [.predecessor 0 8241 .coefficient, .predecessor 1 8242 .coefficient])

def exact8244RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact8244RawTermsValid :
    exact8244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8244 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63163⟩⟩) exact8244RawTerms (.finite 1818214806102629497873537) 8243 .exactZero (none)

def event8245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66870⟩⟩) 0 ⟨63163⟩ 8244

def event8246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66870⟩⟩) 1 ⟨66869⟩ 8136

def event8247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66870⟩⟩) (.sum [.predecessor 0 8245 .coefficient, .predecessor 1 8246 .coefficient])

def exact8248RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8248RawTermsValid :
    exact8248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66870⟩⟩) exact8248RawTerms (.finite 2044702714934587786668817) 8247 .exactZero (none)

def event8249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66871⟩⟩) 0 ⟨66870⟩ 8248

def event8250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66871⟩⟩) 1 ⟨26675⟩ 8128

def event8251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66871⟩⟩) (.sum [.predecessor 0 8249 .coefficient, .predecessor 1 8250 .coefficient])

def exact8252RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8252RawTermsValid :
    exact8252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8252 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66871⟩⟩) exact8252RawTerms (.finite 2271712485307633536959017) 8251 .exactZero (none)

def event8253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66872⟩⟩) 0 ⟨66871⟩ 8252

def event8254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66872⟩⟩) 1 ⟨29355⟩ 8120

def event8255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66872⟩⟩) (.sum [.predecessor 0 8253 .coefficient, .predecessor 1 8254 .coefficient])

def exact8256RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8256RawTermsValid :
    exact8256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8256 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66872⟩⟩) exact8256RawTerms (.finite 2499949335520533588602137) 8255 .exactZero (none)

def event8257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66873⟩⟩) 0 ⟨66872⟩ 8256

def event8258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66873⟩⟩) 1 ⟨35012⟩ 8112

def event8259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66873⟩⟩) (.sum [.predecessor 0 8257 .coefficient, .predecessor 1 8258 .coefficient])

def exact8260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8260RawTermsValid :
    exact8260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66873⟩⟩) exact8260RawTerms (.finite 2728804713782791092959737) 8259 .exactZero (none)

def event8261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66874⟩⟩) 0 ⟨66873⟩ 8260

def event8262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66874⟩⟩) 1 ⟨37692⟩ 8104

def event8263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66874⟩⟩) (.sum [.predecessor 0 8261 .coefficient, .predecessor 1 8262 .coefficient])

def exact8264RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8264RawTermsValid :
    exact8264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8264 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66874⟩⟩) exact8264RawTerms (.finite 2957926202950004710694497) 8263 .exactZero (none)

def event8265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66875⟩⟩) 0 ⟨66874⟩ 8264

def event8266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66875⟩⟩) 1 ⟨40375⟩ 8096

def event8267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66875⟩⟩) (.sum [.predecessor 0 8265 .coefficient, .predecessor 1 8266 .coefficient])

def exact8268RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8268RawTermsValid :
    exact8268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8268 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66875⟩⟩) exact8268RawTerms (.finite 3187511970717354526236217) 8267 .exactZero (none)

def event8269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66876⟩⟩) 0 ⟨66875⟩ 8268

def event8270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66876⟩⟩) 1 ⟨43055⟩ 8088

def event8271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66876⟩⟩) (.sum [.predecessor 0 8269 .coefficient, .predecessor 1 8270 .coefficient])

def exact8272RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8272RawTermsValid :
    exact8272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66876⟩⟩) exact8272RawTerms (.finite 3417662756781096507033577) 8271 .exactZero (none)

def event8273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66877⟩⟩) 0 ⟨66876⟩ 8272

def event8274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66877⟩⟩) 1 ⟨45732⟩ 8080

def event8275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66877⟩⟩) (.sum [.predecessor 0 8273 .coefficient, .predecessor 1 8274 .coefficient])

def exact8276RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8276RawTermsValid :
    exact8276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8276 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66877⟩⟩) exact8276RawTerms (.finite 3648263642165693263543057) 8275 .exactZero (none)

def event8277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66878⟩⟩) 0 ⟨66877⟩ 8276

def event8278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66878⟩⟩) 1 ⟨48412⟩ 8072

def event8279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66878⟩⟩) (.sum [.predecessor 0 8277 .coefficient, .predecessor 1 8278 .coefficient])

def exact8280RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8280RawTermsValid :
    exact8280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66878⟩⟩) exact8280RawTerms (.finite 3878994884184198780231457) 8279 .exactZero (none)

def event8281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67541⟩⟩) 0 ⟨66878⟩ 8280

def event8282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67541⟩⟩) 1 ⟨67539⟩ 8064

def event8283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67541⟩⟩) (.sum [.predecessor 0 8281 .coefficient, .predecessor 1 8282 .coefficient])

def exact8284RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8284RawTermsValid :
    exact8284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67541⟩⟩) exact8284RawTerms (.finite 8101376613122849735629177) 8283 .exactZero (none)

def event8285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67542⟩⟩) 0 ⟨67541⟩ 8284

def event8286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67542⟩⟩) 1 ⟨6765⟩ 7561

def event8287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67542⟩⟩) (.product (.predecessor 0 8285 .coefficient) (.predecessor 1 8286 .coefficient) (⟨false, true, none, none, some 1⟩))

def event8288 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 5⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], []⟩, (-1)⟩)

def event8289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 7⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], []⟩, (1)⟩)

def event8290 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 8⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], []⟩, (1)⟩)

def event8291 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 9⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], []⟩, (1)⟩)

def event8292 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 11⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], []⟩, (1)⟩)

def event8293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 12⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩, (1)⟩)

def event8294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 13⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩)

def event8295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 15⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩)

def event8296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 16⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩)

def event8297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 18⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩)

def event8298 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 0⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩)

def event8299 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 1⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩)

def event8300 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 2⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩)

def event8301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 3⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩)

def event8302 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 4⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩)

def event8303 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 6⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩)

def event8304 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 10⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩)

def event8305 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 14⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩)

def event8306 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67542⟩⟩, .operator (⟨8284, 17⟩, ⟨7561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩)

def exact8307RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], []⟩, (1)⟩]

theorem exact8307RawTermsValid :
    exact8307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8307 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67542⟩⟩) exact8307RawTerms (.finite 55627767500075853938083808449072567753429004894048303081797576416370224001882661191858962315950815784382110477312374410834170377657385335699443507046704046708946015012931458568260678579557123686400) 8287 .exactZero (none)

def event8308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6806⟩⟩) (.authority (.factStore))

def exact8309RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩], []⟩, (1)⟩]

theorem exact8309RawTermsValid :
    exact8309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6806⟩⟩) exact8309RawTerms (.finite 392208910876296843290869724658024391949918004018017135461780498791886113798803788492196058508406193818925718552453952606142266741361954240447112917026659566933549801769) 8308 .exactZero (none)

def event8310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6170⟩⟩) (.authority (.operator))

def event8311 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6170⟩⟩) (.finite 8)

def event8312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6172⟩⟩) 0 ⟨392⟩ 14

def event8313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6172⟩⟩) 1 ⟨6170⟩ 8311

def event8314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6172⟩⟩) (.sum [.predecessor 0 8312 .coefficient, .predecessor 1 8313 .coefficient])

def event8315 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6172⟩⟩) (.finite 655348)

def event8316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6182⟩⟩) 0 ⟨6172⟩ 8315

def event8317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6182⟩⟩) 1 ⟨5426⟩ 38

def event8318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6182⟩⟩) (.identity (.predecessor 1 8317 .coefficient))

def event8319 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6182⟩⟩) (.finite 655360)

def event8320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47906⟩⟩) 0 ⟨6182⟩ 8319

def event8321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47906⟩⟩) (.authority (.programFamilyFact))

def exact8322RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47906⟩⟩], []⟩, (1)⟩]

theorem exact8322RawTermsValid :
    exact8322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47906⟩⟩) exact8322RawTerms (.finite 60) 8321 .exactZero (none)

def event8323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15126⟩⟩) 0 ⟨6182⟩ 8319

def event8324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15126⟩⟩) (.authority (.programFamilyFact))

def exact8325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15126⟩⟩], []⟩, (1)⟩]

theorem exact8325RawTermsValid :
    exact8325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15126⟩⟩) exact8325RawTerms (.finite 60) 8324 .exactZero (none)

def event8326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47907⟩⟩) 0 ⟨15126⟩ 8325

def event8327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47907⟩⟩) 1 ⟨47906⟩ 8322

def event8328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47907⟩⟩) (.product (.predecessor 0 8326 .coefficient) (.predecessor 1 8327 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event8329 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47907⟩⟩, .operator (⟨8325, 0⟩, ⟨8322, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], []⟩, (1)⟩)

def exact8330RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], []⟩, (1)⟩]

theorem exact8330RawTermsValid :
    exact8330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47907⟩⟩) exact8330RawTerms (.finite 3600) 8328 .exactZero (none)

def event8331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47908⟩⟩) 0 ⟨47907⟩ 8330

def event8332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47908⟩⟩) (.identity (.predecessor 0 8331 .coefficient))

def event8333 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47908⟩⟩) (.finite 3600)

def event8334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48172⟩⟩) 0 ⟨47908⟩ 8333

def event8335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48172⟩⟩) (.authority (.programFamilyFact))

def exact8336RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48172⟩⟩], []⟩, (1)⟩]

theorem exact8336RawTermsValid :
    exact8336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48172⟩⟩) exact8336RawTerms (.finite 60) 8335 .exactZero (none)

def event8337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48173⟩⟩) 0 ⟨48172⟩ 8336

def event8338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48173⟩⟩) (.identity (.predecessor 0 8337 .coefficient))

def event8339 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48173⟩⟩) (.finite 60)

def event8340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48402⟩⟩) 0 ⟨48173⟩ 8339

def event8341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48402⟩⟩) (.authority (.programFamilyFact))

def exact8342RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], []⟩, (1)⟩]

theorem exact8342RawTermsValid :
    exact8342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48402⟩⟩) exact8342RawTerms (.finite 63) 8341 .exactZero (none)

def event8343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45226⟩⟩) 0 ⟨6182⟩ 8319

def event8344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45226⟩⟩) (.authority (.programFamilyFact))

def exact8345RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45226⟩⟩], []⟩, (1)⟩]

theorem exact8345RawTermsValid :
    exact8345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8345 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45226⟩⟩) exact8345RawTerms (.finite 58) 8344 .exactZero (none)

def event8346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14826⟩⟩) 0 ⟨6182⟩ 8319

def event8347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14826⟩⟩) (.authority (.programFamilyFact))

def exact8348RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14826⟩⟩], []⟩, (1)⟩]

theorem exact8348RawTermsValid :
    exact8348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8348 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14826⟩⟩) exact8348RawTerms (.finite 58) 8347 .exactZero (none)

def event8349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45227⟩⟩) 0 ⟨14826⟩ 8348

def event8350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45227⟩⟩) 1 ⟨45226⟩ 8345

def event8351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45227⟩⟩) (.product (.predecessor 0 8349 .coefficient) (.predecessor 1 8350 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event8352 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45227⟩⟩, .operator (⟨8348, 0⟩, ⟨8345, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], []⟩, (1)⟩)

def exact8353RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], []⟩, (1)⟩]

theorem exact8353RawTermsValid :
    exact8353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8353 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45227⟩⟩) exact8353RawTerms (.finite 3364) 8351 .exactZero (none)

def event8354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45228⟩⟩) 0 ⟨45227⟩ 8353

def event8355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45228⟩⟩) (.identity (.predecessor 0 8354 .coefficient))

def event8356 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45228⟩⟩) (.finite 3364)

def event8357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45492⟩⟩) 0 ⟨45228⟩ 8356

def event8358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45492⟩⟩) (.authority (.programFamilyFact))

def exact8359RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], []⟩, (1)⟩]

theorem exact8359RawTermsValid :
    exact8359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45492⟩⟩) exact8359RawTerms (.finite 58) 8358 .exactZero (none)

def event8360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45493⟩⟩) 0 ⟨45492⟩ 8359

def event8361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45493⟩⟩) (.identity (.predecessor 0 8360 .coefficient))

def event8362 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45493⟩⟩) (.finite 58)

def event8363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45722⟩⟩) 0 ⟨45493⟩ 8362

def event8364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45722⟩⟩) (.authority (.programFamilyFact))

def exact8365RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45722⟩⟩], []⟩, (1)⟩]

theorem exact8365RawTermsValid :
    exact8365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45722⟩⟩) exact8365RawTerms (.finite 63) 8364 .exactZero (none)

def event8366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42546⟩⟩) 0 ⟨6182⟩ 8319

def event8367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42546⟩⟩) (.authority (.programFamilyFact))

def exact8368RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42546⟩⟩], []⟩, (1)⟩]

theorem exact8368RawTermsValid :
    exact8368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42546⟩⟩) exact8368RawTerms (.finite 52) 8367 .exactZero (none)

def event8369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14526⟩⟩) 0 ⟨6182⟩ 8319

def event8370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14526⟩⟩) (.authority (.programFamilyFact))

def exact8371RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14526⟩⟩], []⟩, (1)⟩]

theorem exact8371RawTermsValid :
    exact8371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8371 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14526⟩⟩) exact8371RawTerms (.finite 52) 8370 .exactZero (none)

def event8372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42547⟩⟩) 0 ⟨14526⟩ 8371

def event8373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42547⟩⟩) 1 ⟨42546⟩ 8368

def event8374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42547⟩⟩) (.product (.predecessor 0 8372 .coefficient) (.predecessor 1 8373 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event8375 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42547⟩⟩, .operator (⟨8371, 0⟩, ⟨8368, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14526⟩⟩, ⟨.program ⟨257⟩, ⟨42546⟩⟩], []⟩, (1)⟩)

def exact8376RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14526⟩⟩, ⟨.program ⟨257⟩, ⟨42546⟩⟩], []⟩, (1)⟩]

theorem exact8376RawTermsValid :
    exact8376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8376 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42547⟩⟩) exact8376RawTerms (.finite 2704) 8374 .exactZero (none)

def event8377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42548⟩⟩) 0 ⟨42547⟩ 8376

def event8378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42548⟩⟩) (.identity (.predecessor 0 8377 .coefficient))

def event8379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42548⟩⟩) (.finite 2704)

def event8380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42812⟩⟩) 0 ⟨42548⟩ 8379

def event8381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42812⟩⟩) (.authority (.programFamilyFact))

def exact8382RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42812⟩⟩], []⟩, (1)⟩]

theorem exact8382RawTermsValid :
    exact8382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42812⟩⟩) exact8382RawTerms (.finite 52) 8381 .exactZero (none)

def event8383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42813⟩⟩) 0 ⟨42812⟩ 8382

def event8384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42813⟩⟩) (.identity (.predecessor 0 8383 .coefficient))

def event8385 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42813⟩⟩) (.finite 52)

def event8386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43038⟩⟩) 0 ⟨42813⟩ 8385

def event8387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43038⟩⟩) (.authority (.programFamilyFact))

def exact8388RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], []⟩, (1)⟩]

theorem exact8388RawTermsValid :
    exact8388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43038⟩⟩) exact8388RawTerms (.finite 63) 8387 .exactZero (none)

def event8389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39866⟩⟩) 0 ⟨6182⟩ 8319

def event8390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39866⟩⟩) (.authority (.programFamilyFact))

def exact8391RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39866⟩⟩], []⟩, (1)⟩]

theorem exact8391RawTermsValid :
    exact8391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8391 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39866⟩⟩) exact8391RawTerms (.finite 46) 8390 .exactZero (none)

def event8392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14226⟩⟩) 0 ⟨6182⟩ 8319

def event8393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14226⟩⟩) (.authority (.programFamilyFact))

def exact8394RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14226⟩⟩], []⟩, (1)⟩]

theorem exact8394RawTermsValid :
    exact8394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8394 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14226⟩⟩) exact8394RawTerms (.finite 46) 8393 .exactZero (none)

def event8395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39867⟩⟩) 0 ⟨14226⟩ 8394

def event8396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39867⟩⟩) 1 ⟨39866⟩ 8391

def event8397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39867⟩⟩) (.product (.predecessor 0 8395 .coefficient) (.predecessor 1 8396 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event8398 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39867⟩⟩, .operator (⟨8394, 0⟩, ⟨8391, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14226⟩⟩, ⟨.program ⟨257⟩, ⟨39866⟩⟩], []⟩, (1)⟩)

def exact8399RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14226⟩⟩, ⟨.program ⟨257⟩, ⟨39866⟩⟩], []⟩, (1)⟩]

theorem exact8399RawTermsValid :
    exact8399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8399 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39867⟩⟩) exact8399RawTerms (.finite 2116) 8397 .exactZero (none)

def event8400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39868⟩⟩) 0 ⟨39867⟩ 8399

def event8401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39868⟩⟩) (.identity (.predecessor 0 8400 .coefficient))

def event8402 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39868⟩⟩) (.finite 2116)

def event8403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40132⟩⟩) 0 ⟨39868⟩ 8402

def event8404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40132⟩⟩) (.authority (.programFamilyFact))

def exact8405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40132⟩⟩], []⟩, (1)⟩]

theorem exact8405RawTermsValid :
    exact8405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40132⟩⟩) exact8405RawTerms (.finite 46) 8404 .exactZero (none)

def event8406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40133⟩⟩) 0 ⟨40132⟩ 8405

def event8407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40133⟩⟩) (.identity (.predecessor 0 8406 .coefficient))

def event8408 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40133⟩⟩) (.finite 46)

def event8409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40358⟩⟩) 0 ⟨40133⟩ 8408

def event8410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40358⟩⟩) (.authority (.programFamilyFact))

def exact8411RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], []⟩, (1)⟩]

theorem exact8411RawTermsValid :
    exact8411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40358⟩⟩) exact8411RawTerms (.finite 63) 8410 .exactZero (none)

def event8412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37186⟩⟩) 0 ⟨6182⟩ 8319

def event8413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37186⟩⟩) (.authority (.programFamilyFact))

def exact8414RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37186⟩⟩], []⟩, (1)⟩]

theorem exact8414RawTermsValid :
    exact8414RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8414 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37186⟩⟩) exact8414RawTerms (.finite 42) 8413 .exactZero (none)

def event8415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13926⟩⟩) 0 ⟨6182⟩ 8319

def event8416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13926⟩⟩) (.authority (.programFamilyFact))

def exact8417RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13926⟩⟩], []⟩, (1)⟩]

theorem exact8417RawTermsValid :
    exact8417RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8417 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13926⟩⟩) exact8417RawTerms (.finite 42) 8416 .exactZero (none)

def event8418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37187⟩⟩) 0 ⟨13926⟩ 8417

def event8419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37187⟩⟩) 1 ⟨37186⟩ 8414

def event8420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37187⟩⟩) (.product (.predecessor 0 8418 .coefficient) (.predecessor 1 8419 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event8421 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37187⟩⟩, .operator (⟨8417, 0⟩, ⟨8414, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], []⟩, (1)⟩)

def exact8422RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13926⟩⟩, ⟨.program ⟨257⟩, ⟨37186⟩⟩], []⟩, (1)⟩]

theorem exact8422RawTermsValid :
    exact8422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8422 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37187⟩⟩) exact8422RawTerms (.finite 1764) 8420 .exactZero (none)

def event8423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37188⟩⟩) 0 ⟨37187⟩ 8422

def event8424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37188⟩⟩) (.identity (.predecessor 0 8423 .coefficient))

def event8425 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37188⟩⟩) (.finite 1764)

def event8426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37452⟩⟩) 0 ⟨37188⟩ 8425

def event8427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37452⟩⟩) (.authority (.programFamilyFact))

def exact8428RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37452⟩⟩], []⟩, (1)⟩]

theorem exact8428RawTermsValid :
    exact8428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8428 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37452⟩⟩) exact8428RawTerms (.finite 42) 8427 .exactZero (none)

def event8429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37453⟩⟩) 0 ⟨37452⟩ 8428

def event8430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37453⟩⟩) (.identity (.predecessor 0 8429 .coefficient))

def event8431 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37453⟩⟩) (.finite 42)

def event8432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37682⟩⟩) 0 ⟨37453⟩ 8431

def event8433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37682⟩⟩) (.authority (.programFamilyFact))

def exact8434RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], []⟩, (1)⟩]

theorem exact8434RawTermsValid :
    exact8434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8434 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37682⟩⟩) exact8434RawTerms (.finite 63) 8433 .exactZero (none)

def event8435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34506⟩⟩) 0 ⟨6182⟩ 8319

def event8436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34506⟩⟩) (.authority (.programFamilyFact))

def exact8437RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34506⟩⟩], []⟩, (1)⟩]

theorem exact8437RawTermsValid :
    exact8437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8437 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34506⟩⟩) exact8437RawTerms (.finite 40) 8436 .exactZero (none)

def event8438 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13626⟩⟩) 0 ⟨6182⟩ 8319

def event8439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13626⟩⟩) (.authority (.programFamilyFact))

def exact8440RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13626⟩⟩], []⟩, (1)⟩]

theorem exact8440RawTermsValid :
    exact8440RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8440 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13626⟩⟩) exact8440RawTerms (.finite 40) 8439 .exactZero (none)

def event8441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34507⟩⟩) 0 ⟨13626⟩ 8440

def event8442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34507⟩⟩) 1 ⟨34506⟩ 8437

def event8443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34507⟩⟩) (.product (.predecessor 0 8441 .coefficient) (.predecessor 1 8442 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event8444 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34507⟩⟩, .operator (⟨8440, 0⟩, ⟨8437, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], []⟩, (1)⟩)

def exact8445RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], []⟩, (1)⟩]

theorem exact8445RawTermsValid :
    exact8445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event8445 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34507⟩⟩) exact8445RawTerms (.finite 1600) 8443 .exactZero (none)

def event8446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34508⟩⟩) 0 ⟨34507⟩ 8445

def event8447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34508⟩⟩) (.identity (.predecessor 0 8446 .coefficient))

def eventLeaf512 : Array AnnotatedEvent := #[
  { event := event8192
    frameStart := 0 },
  { event := event8193
    frameStart := 0 },
  { event := event8194
    frameStart := 0 },
  { event := event8195
    frameStart := 0 },
  { event := event8196
    frameStart := 0 },
  { event := event8197
    frameStart := 0 },
  { event := event8198
    frameStart := 0 },
  { event := event8199
    frameStart := 0 },
  { event := event8200
    frameStart := 0 },
  { event := event8201
    frameStart := 0 },
  { event := event8202
    frameStart := 0 },
  { event := event8203
    frameStart := 0 },
  { event := event8204
    frameStart := 0 },
  { event := event8205
    frameStart := 0 },
  { event := event8206
    frameStart := 0 },
  { event := event8207
    frameStart := 0 }
]

def eventLeaf513 : Array AnnotatedEvent := #[
  { event := event8208
    frameStart := 0 },
  { event := event8209
    frameStart := 0 },
  { event := event8210
    frameStart := 0 },
  { event := event8211
    frameStart := 0 },
  { event := event8212
    frameStart := 0 },
  { event := event8213
    frameStart := 0 },
  { event := event8214
    frameStart := 0 },
  { event := event8215
    frameStart := 0 },
  { event := event8216
    frameStart := 0 },
  { event := event8217
    frameStart := 0 },
  { event := event8218
    frameStart := 0 },
  { event := event8219
    frameStart := 0 },
  { event := event8220
    frameStart := 0 },
  { event := event8221
    frameStart := 0 },
  { event := event8222
    frameStart := 0 },
  { event := event8223
    frameStart := 0 }
]

def eventLeaf514 : Array AnnotatedEvent := #[
  { event := event8224
    frameStart := 0 },
  { event := event8225
    frameStart := 0 },
  { event := event8226
    frameStart := 0 },
  { event := event8227
    frameStart := 0 },
  { event := event8228
    frameStart := 0 },
  { event := event8229
    frameStart := 0 },
  { event := event8230
    frameStart := 0 },
  { event := event8231
    frameStart := 0 },
  { event := event8232
    frameStart := 0 },
  { event := event8233
    frameStart := 0 },
  { event := event8234
    frameStart := 0 },
  { event := event8235
    frameStart := 0 },
  { event := event8236
    frameStart := 0 },
  { event := event8237
    frameStart := 0 },
  { event := event8238
    frameStart := 0 },
  { event := event8239
    frameStart := 0 }
]

def eventLeaf515 : Array AnnotatedEvent := #[
  { event := event8240
    frameStart := 0 },
  { event := event8241
    frameStart := 0 },
  { event := event8242
    frameStart := 0 },
  { event := event8243
    frameStart := 0 },
  { event := event8244
    frameStart := 0 },
  { event := event8245
    frameStart := 0 },
  { event := event8246
    frameStart := 0 },
  { event := event8247
    frameStart := 0 },
  { event := event8248
    frameStart := 0 },
  { event := event8249
    frameStart := 0 },
  { event := event8250
    frameStart := 0 },
  { event := event8251
    frameStart := 0 },
  { event := event8252
    frameStart := 0 },
  { event := event8253
    frameStart := 0 },
  { event := event8254
    frameStart := 0 },
  { event := event8255
    frameStart := 0 }
]

def eventLeaf516 : Array AnnotatedEvent := #[
  { event := event8256
    frameStart := 0 },
  { event := event8257
    frameStart := 0 },
  { event := event8258
    frameStart := 0 },
  { event := event8259
    frameStart := 0 },
  { event := event8260
    frameStart := 0 },
  { event := event8261
    frameStart := 0 },
  { event := event8262
    frameStart := 0 },
  { event := event8263
    frameStart := 0 },
  { event := event8264
    frameStart := 0 },
  { event := event8265
    frameStart := 0 },
  { event := event8266
    frameStart := 0 },
  { event := event8267
    frameStart := 0 },
  { event := event8268
    frameStart := 0 },
  { event := event8269
    frameStart := 0 },
  { event := event8270
    frameStart := 0 },
  { event := event8271
    frameStart := 0 }
]

def eventLeaf517 : Array AnnotatedEvent := #[
  { event := event8272
    frameStart := 0 },
  { event := event8273
    frameStart := 0 },
  { event := event8274
    frameStart := 0 },
  { event := event8275
    frameStart := 0 },
  { event := event8276
    frameStart := 0 },
  { event := event8277
    frameStart := 0 },
  { event := event8278
    frameStart := 0 },
  { event := event8279
    frameStart := 0 },
  { event := event8280
    frameStart := 0 },
  { event := event8281
    frameStart := 0 },
  { event := event8282
    frameStart := 0 },
  { event := event8283
    frameStart := 0 },
  { event := event8284
    frameStart := 0 },
  { event := event8285
    frameStart := 0 },
  { event := event8286
    frameStart := 0 },
  { event := event8287
    frameStart := 0 }
]

def eventLeaf518 : Array AnnotatedEvent := #[
  { event := event8288
    frameStart := 0 },
  { event := event8289
    frameStart := 0 },
  { event := event8290
    frameStart := 0 },
  { event := event8291
    frameStart := 0 },
  { event := event8292
    frameStart := 0 },
  { event := event8293
    frameStart := 0 },
  { event := event8294
    frameStart := 0 },
  { event := event8295
    frameStart := 0 },
  { event := event8296
    frameStart := 0 },
  { event := event8297
    frameStart := 0 },
  { event := event8298
    frameStart := 0 },
  { event := event8299
    frameStart := 0 },
  { event := event8300
    frameStart := 0 },
  { event := event8301
    frameStart := 0 },
  { event := event8302
    frameStart := 0 },
  { event := event8303
    frameStart := 0 }
]

def eventLeaf519 : Array AnnotatedEvent := #[
  { event := event8304
    frameStart := 0 },
  { event := event8305
    frameStart := 0 },
  { event := event8306
    frameStart := 0 },
  { event := event8307
    frameStart := 0 },
  { event := event8308
    frameStart := 0 },
  { event := event8309
    frameStart := 0 },
  { event := event8310
    frameStart := 0 },
  { event := event8311
    frameStart := 0 },
  { event := event8312
    frameStart := 0 },
  { event := event8313
    frameStart := 0 },
  { event := event8314
    frameStart := 0 },
  { event := event8315
    frameStart := 0 },
  { event := event8316
    frameStart := 0 },
  { event := event8317
    frameStart := 0 },
  { event := event8318
    frameStart := 0 },
  { event := event8319
    frameStart := 0 }
]

def eventLeaf520 : Array AnnotatedEvent := #[
  { event := event8320
    frameStart := 0 },
  { event := event8321
    frameStart := 0 },
  { event := event8322
    frameStart := 0 },
  { event := event8323
    frameStart := 0 },
  { event := event8324
    frameStart := 0 },
  { event := event8325
    frameStart := 0 },
  { event := event8326
    frameStart := 0 },
  { event := event8327
    frameStart := 0 },
  { event := event8328
    frameStart := 0 },
  { event := event8329
    frameStart := 0 },
  { event := event8330
    frameStart := 0 },
  { event := event8331
    frameStart := 0 },
  { event := event8332
    frameStart := 0 },
  { event := event8333
    frameStart := 0 },
  { event := event8334
    frameStart := 0 },
  { event := event8335
    frameStart := 0 }
]

def eventLeaf521 : Array AnnotatedEvent := #[
  { event := event8336
    frameStart := 0 },
  { event := event8337
    frameStart := 0 },
  { event := event8338
    frameStart := 0 },
  { event := event8339
    frameStart := 0 },
  { event := event8340
    frameStart := 0 },
  { event := event8341
    frameStart := 0 },
  { event := event8342
    frameStart := 0 },
  { event := event8343
    frameStart := 0 },
  { event := event8344
    frameStart := 0 },
  { event := event8345
    frameStart := 0 },
  { event := event8346
    frameStart := 0 },
  { event := event8347
    frameStart := 0 },
  { event := event8348
    frameStart := 0 },
  { event := event8349
    frameStart := 0 },
  { event := event8350
    frameStart := 0 },
  { event := event8351
    frameStart := 0 }
]

def eventLeaf522 : Array AnnotatedEvent := #[
  { event := event8352
    frameStart := 0 },
  { event := event8353
    frameStart := 0 },
  { event := event8354
    frameStart := 0 },
  { event := event8355
    frameStart := 0 },
  { event := event8356
    frameStart := 0 },
  { event := event8357
    frameStart := 0 },
  { event := event8358
    frameStart := 0 },
  { event := event8359
    frameStart := 0 },
  { event := event8360
    frameStart := 0 },
  { event := event8361
    frameStart := 0 },
  { event := event8362
    frameStart := 0 },
  { event := event8363
    frameStart := 0 },
  { event := event8364
    frameStart := 0 },
  { event := event8365
    frameStart := 0 },
  { event := event8366
    frameStart := 0 },
  { event := event8367
    frameStart := 0 }
]

def eventLeaf523 : Array AnnotatedEvent := #[
  { event := event8368
    frameStart := 0 },
  { event := event8369
    frameStart := 0 },
  { event := event8370
    frameStart := 0 },
  { event := event8371
    frameStart := 0 },
  { event := event8372
    frameStart := 0 },
  { event := event8373
    frameStart := 0 },
  { event := event8374
    frameStart := 0 },
  { event := event8375
    frameStart := 0 },
  { event := event8376
    frameStart := 0 },
  { event := event8377
    frameStart := 0 },
  { event := event8378
    frameStart := 0 },
  { event := event8379
    frameStart := 0 },
  { event := event8380
    frameStart := 0 },
  { event := event8381
    frameStart := 0 },
  { event := event8382
    frameStart := 0 },
  { event := event8383
    frameStart := 0 }
]

def eventLeaf524 : Array AnnotatedEvent := #[
  { event := event8384
    frameStart := 0 },
  { event := event8385
    frameStart := 0 },
  { event := event8386
    frameStart := 0 },
  { event := event8387
    frameStart := 0 },
  { event := event8388
    frameStart := 0 },
  { event := event8389
    frameStart := 0 },
  { event := event8390
    frameStart := 0 },
  { event := event8391
    frameStart := 0 },
  { event := event8392
    frameStart := 0 },
  { event := event8393
    frameStart := 0 },
  { event := event8394
    frameStart := 0 },
  { event := event8395
    frameStart := 0 },
  { event := event8396
    frameStart := 0 },
  { event := event8397
    frameStart := 0 },
  { event := event8398
    frameStart := 0 },
  { event := event8399
    frameStart := 0 }
]

def eventLeaf525 : Array AnnotatedEvent := #[
  { event := event8400
    frameStart := 0 },
  { event := event8401
    frameStart := 0 },
  { event := event8402
    frameStart := 0 },
  { event := event8403
    frameStart := 0 },
  { event := event8404
    frameStart := 0 },
  { event := event8405
    frameStart := 0 },
  { event := event8406
    frameStart := 0 },
  { event := event8407
    frameStart := 0 },
  { event := event8408
    frameStart := 0 },
  { event := event8409
    frameStart := 0 },
  { event := event8410
    frameStart := 0 },
  { event := event8411
    frameStart := 0 },
  { event := event8412
    frameStart := 0 },
  { event := event8413
    frameStart := 0 },
  { event := event8414
    frameStart := 0 },
  { event := event8415
    frameStart := 0 }
]

def eventLeaf526 : Array AnnotatedEvent := #[
  { event := event8416
    frameStart := 0 },
  { event := event8417
    frameStart := 0 },
  { event := event8418
    frameStart := 0 },
  { event := event8419
    frameStart := 0 },
  { event := event8420
    frameStart := 0 },
  { event := event8421
    frameStart := 0 },
  { event := event8422
    frameStart := 0 },
  { event := event8423
    frameStart := 0 },
  { event := event8424
    frameStart := 0 },
  { event := event8425
    frameStart := 0 },
  { event := event8426
    frameStart := 0 },
  { event := event8427
    frameStart := 0 },
  { event := event8428
    frameStart := 0 },
  { event := event8429
    frameStart := 0 },
  { event := event8430
    frameStart := 0 },
  { event := event8431
    frameStart := 0 }
]

def eventLeaf527 : Array AnnotatedEvent := #[
  { event := event8432
    frameStart := 0 },
  { event := event8433
    frameStart := 0 },
  { event := event8434
    frameStart := 0 },
  { event := event8435
    frameStart := 0 },
  { event := event8436
    frameStart := 0 },
  { event := event8437
    frameStart := 0 },
  { event := event8438
    frameStart := 0 },
  { event := event8439
    frameStart := 0 },
  { event := event8440
    frameStart := 0 },
  { event := event8441
    frameStart := 0 },
  { event := event8442
    frameStart := 0 },
  { event := event8443
    frameStart := 0 },
  { event := event8444
    frameStart := 0 },
  { event := event8445
    frameStart := 0 },
  { event := event8446
    frameStart := 0 },
  { event := event8447
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events032
