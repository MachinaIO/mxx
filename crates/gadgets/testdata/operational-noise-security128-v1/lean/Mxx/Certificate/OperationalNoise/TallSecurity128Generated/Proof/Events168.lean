import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events168

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event43008 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event43009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 43008

def event43010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 42994

def event43011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 43010 .coefficient))

def event43012 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event43013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40010⟩⟩) 0 ⟨11600⟩ 43012

def event43014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40010⟩⟩) (.authority (.programFamilyFact))

def exact43015RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40010⟩⟩], []⟩, (1)⟩]

theorem exact43015RawTermsValid :
    exact43015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40010⟩⟩) exact43015RawTerms (.finite 46) 43014 .exactZero (none)

def event43016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14316⟩⟩) 0 ⟨11600⟩ 43012

def event43017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14316⟩⟩) (.authority (.programFamilyFact))

def exact43018RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩], []⟩, (1)⟩]

theorem exact43018RawTermsValid :
    exact43018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43018 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14316⟩⟩) exact43018RawTerms (.finite 46) 43017 .exactZero (none)

def event43019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40011⟩⟩) 0 ⟨14316⟩ 43018

def event43020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40011⟩⟩) 1 ⟨40010⟩ 43015

def event43021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40011⟩⟩) (.product (.predecessor 0 43019 .coefficient) (.predecessor 1 43020 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event43022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40011⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], []⟩) [⟨.result 43018 .coefficient, true, some 1⟩, ⟨.result 43015 .coefficient, true, some 1⟩])

def event43023 : Event := .survivorFold (1) 43022

def exact43024RawTerms : List Term := []

theorem exact43024RawTermsValid :
    exact43024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40011⟩⟩) exact43024RawTerms (.finite 2116) 43021 (.finite 2116) (some (43022))

def event43025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40012⟩⟩) 0 ⟨40011⟩ 43024

def event43026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40012⟩⟩) (.identity (.predecessor 0 43025 .coefficient))

def event43027 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40012⟩⟩) (.finite 2116)

def event43028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40180⟩⟩) 0 ⟨40012⟩ 43027

def event43029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40180⟩⟩) (.authority (.programFamilyFact))

def exact43030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], []⟩, (1)⟩]

theorem exact43030RawTermsValid :
    exact43030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40180⟩⟩) exact43030RawTerms (.finite 46) 43029 .exactZero (none)

def event43031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40181⟩⟩) 0 ⟨40180⟩ 43030

def event43032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40181⟩⟩) (.identity (.predecessor 0 43031 .coefficient))

def event43033 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40181⟩⟩) (.finite 46)

def event43034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41032⟩⟩) 0 ⟨40181⟩ 43033

def event43035 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41032⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact43036RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩, (1)⟩]

theorem exact43036RawTermsValid :
    exact43036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43036 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41032⟩⟩) exact43036RawTerms (.finite 5647228698) 43035 .exactZero (none)

def event43037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact43038RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact43038RawTermsValid :
    exact43038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43038 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact43038RawTerms .large 43037 .exactZero (none)

def event43039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41033⟩⟩) 0 ⟨35⟩ 43038

def event43040 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41033⟩⟩) 1 ⟨41032⟩ 43036

def event43041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41033⟩⟩) (.product (.predecessor 0 43039 .coefficient) (.predecessor 1 43040 .coefficient) (⟨false, false, none, none, none⟩))

def event43042 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41033⟩⟩, .operator (⟨43038, 0⟩, ⟨43036, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩, (1)⟩)

def exact43043RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩, (1)⟩]

theorem exact43043RawTermsValid :
    exact43043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41033⟩⟩) exact43043RawTerms .large 43041 .exactZero (none)

def event43044 : Event := .preFoldPolynomial 43043 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩, (1)⟩] .exactZero none

def exact43045RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩, (1)⟩]

def event43045 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41033⟩⟩) 43044 exact43045RawTerms .large 43041 .exactZero (none)

def event43046 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨42213⟩⟩)

def event43047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event43048 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event43049 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event43050 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event43051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event43052 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event43053 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event43054 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event43055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 43054

def event43056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 43052

def event43057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 43055 .coefficient) (.value (.predecessor 1 43056 .coefficient)))

def event43058 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event43059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 43058

def event43060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 43050

def event43061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 43059 .coefficient, .predecessor 1 43060 .coefficient])

def event43062 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event43063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 43062

def event43064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 43048

def event43065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 43064 .coefficient))

def event43066 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event43067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40010⟩⟩) 0 ⟨11600⟩ 43066

def event43068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40010⟩⟩) (.authority (.programFamilyFact))

def exact43069RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40010⟩⟩], []⟩, (1)⟩]

theorem exact43069RawTermsValid :
    exact43069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43069 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40010⟩⟩) exact43069RawTerms (.finite 46) 43068 .exactZero (none)

def event43070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14316⟩⟩) 0 ⟨11600⟩ 43066

def event43071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14316⟩⟩) (.authority (.programFamilyFact))

def exact43072RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩], []⟩, (1)⟩]

theorem exact43072RawTermsValid :
    exact43072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43072 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14316⟩⟩) exact43072RawTerms (.finite 46) 43071 .exactZero (none)

def event43073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40011⟩⟩) 0 ⟨14316⟩ 43072

def event43074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40011⟩⟩) 1 ⟨40010⟩ 43069

def event43075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40011⟩⟩) (.product (.predecessor 0 43073 .coefficient) (.predecessor 1 43074 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event43076 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40011⟩⟩, .operator (⟨43072, 0⟩, ⟨43069, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], []⟩, (1)⟩)

def exact43077RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], []⟩, (1)⟩]

theorem exact43077RawTermsValid :
    exact43077RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43077 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40011⟩⟩) exact43077RawTerms (.finite 2116) 43075 .exactZero (none)

def event43078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40012⟩⟩) 0 ⟨40011⟩ 43077

def event43079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40012⟩⟩) (.identity (.predecessor 0 43078 .coefficient))

def event43080 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40012⟩⟩) (.finite 2116)

def event43081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40180⟩⟩) 0 ⟨40012⟩ 43080

def event43082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40180⟩⟩) (.authority (.programFamilyFact))

def exact43083RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], []⟩, (1)⟩]

theorem exact43083RawTermsValid :
    exact43083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40180⟩⟩) exact43083RawTerms (.finite 46) 43082 .exactZero (none)

def event43084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40181⟩⟩) 0 ⟨40180⟩ 43083

def event43085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40181⟩⟩) (.identity (.predecessor 0 43084 .coefficient))

def event43086 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40181⟩⟩) (.finite 46)

def event43087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41340⟩⟩) 0 ⟨40181⟩ 43086

def event43088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41340⟩⟩) (.authority (.programFamilyFact))

def event43089 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41340⟩⟩) (.finite 3720)

def event43090 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event43091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41341⟩⟩) 0 ⟨7177⟩ 43090

def event43092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41341⟩⟩) 1 ⟨41340⟩ 43089

def event43093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41341⟩⟩) (.authority (.operator))

def exact43094RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (1)⟩]

theorem exact43094RawTermsValid :
    exact43094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43094 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41341⟩⟩) exact43094RawTerms .large 43093 .exactZero (none)

def event43095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42208⟩⟩) 0 ⟨41341⟩ 43094

def event43096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42208⟩⟩) (.authority (.operator))

def exact43097RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (1)⟩]

theorem exact43097RawTermsValid :
    exact43097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42208⟩⟩) exact43097RawTerms (.finite 8192) 43096 .exactZero (none)

def event43098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event43099 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event43100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41502⟩⟩) 0 ⟨40181⟩ 43086

def event43101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41502⟩⟩) 1 ⟨136⟩ 43099

def event43102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41502⟩⟩) (.sum [.predecessor 0 43100 .coefficient, .predecessor 1 43101 .coefficient])

def event43103 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41502⟩⟩) (.finite 46)

def event43104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41503⟩⟩) 0 ⟨41502⟩ 43103

def event43105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41503⟩⟩) (.identity (.predecessor 0 43104 .coefficient))

def exact43106RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], []⟩, (1)⟩]

theorem exact43106RawTermsValid :
    exact43106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41503⟩⟩) exact43106RawTerms (.finite 46) 43105 .exactZero (none)

def event43107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact43108RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact43108RawTermsValid :
    exact43108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact43108RawTerms .large 43107 .exactZero (none)

def event43109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41504⟩⟩) 0 ⟨6908⟩ 43108

def event43110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41504⟩⟩) 1 ⟨41503⟩ 43106

def event43111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41504⟩⟩) (.product (.predecessor 0 43109 .coefficient) (.predecessor 1 43110 .coefficient) (⟨false, false, none, none, none⟩))

def event43112 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41504⟩⟩, .operator (⟨43108, 0⟩, ⟨43106, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact43113RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact43113RawTermsValid :
    exact43113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43113 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41504⟩⟩) exact43113RawTerms .large 43111 .exactZero (none)

def event43114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 43090

def event43115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact43116RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact43116RawTermsValid :
    exact43116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43116 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact43116RawTerms .large 43115 .exactZero (none)

def event43117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41505⟩⟩) 0 ⟨7193⟩ 43116

def event43118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41505⟩⟩) 1 ⟨41504⟩ 43113

def event43119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41505⟩⟩) (.sum [.predecessor 0 43117 .coefficient, .predecessor 1 43118 .coefficient])

def exact43120RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43120RawTermsValid :
    exact43120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43120 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41505⟩⟩) exact43120RawTerms .large 43119 .exactZero (none)

def event43121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42209⟩⟩) 0 ⟨41505⟩ 43120

def event43122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42209⟩⟩) 1 ⟨42208⟩ 43097

def event43123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42209⟩⟩) (.product (.predecessor 0 43121 .coefficient) (.predecessor 1 43122 .coefficient) (⟨false, false, none, none, none⟩))

def event43124 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42209⟩⟩, .operator (⟨43120, 0⟩, ⟨43097, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (1)⟩)

def event43125 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42209⟩⟩, .operator (⟨43120, 1⟩, ⟨43097, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (-1)⟩)

def event43126 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42209⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42208⟩⟩) ⟨41341⟩ 43094)

def event43127 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42209⟩⟩, .relation 43126 0, ⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (-1)⟩)

def exact43128RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (-1)⟩]

theorem exact43128RawTermsValid :
    exact43128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43128 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42209⟩⟩) exact43128RawTerms .large 43123 .exactZero (none)

def event43129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40439⟩⟩) 0 ⟨40181⟩ 43086

def event43130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40439⟩⟩) (.authority (.programFamilyFact))

def exact43131RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40439⟩⟩], []⟩, (1)⟩]

theorem exact43131RawTermsValid :
    exact43131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43131 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40439⟩⟩) exact43131RawTerms (.finite 46) 43130 .exactZero (none)

def event43132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40441⟩⟩) 0 ⟨6908⟩ 43108

def event43133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40441⟩⟩) 1 ⟨40439⟩ 43131

def event43134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40441⟩⟩) (.product (.predecessor 0 43132 .coefficient) (.predecessor 1 43133 .coefficient) (⟨false, true, none, none, some 1⟩))

def event43135 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40441⟩⟩, .operator (⟨43108, 0⟩, ⟨43131, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact43136RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact43136RawTermsValid :
    exact43136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43136 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40441⟩⟩) exact43136RawTerms .large 43134 .exactZero (none)

def event43137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7225⟩⟩) 0 ⟨7177⟩ 43090

def event43138 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7225⟩⟩) (.authority (.operator))

def exact43139RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩]

theorem exact43139RawTermsValid :
    exact43139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43139 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7225⟩⟩) exact43139RawTerms .large 43138 .exactZero (none)

def event43140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40442⟩⟩) 0 ⟨7225⟩ 43139

def event43141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40442⟩⟩) 1 ⟨40441⟩ 43136

def event43142 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40442⟩⟩) (.sum [.predecessor 0 43140 .coefficient, .predecessor 1 43141 .coefficient])

def exact43143RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43143RawTermsValid :
    exact43143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43143 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40442⟩⟩) exact43143RawTerms .large 43142 .exactZero (none)

def event43144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42213⟩⟩) 0 ⟨40442⟩ 43143

def event43145 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42213⟩⟩) 1 ⟨42209⟩ 43128

def event43146 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42213⟩⟩) (.sum [.predecessor 0 43144 .coefficient, .predecessor 1 43145 .coefficient])

def exact43147RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43147RawTermsValid :
    exact43147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43147 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42213⟩⟩) exact43147RawTerms .large 43146 .exactZero (none)

def event43148 : Event := .preFoldPolynomial 43147 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact43149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event43149 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨42213⟩⟩) 43148 exact43149RawTerms .large 43146 .exactZero (none)

def event43150 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨40181⟩⟩) ⟨⟨104⟩, ⟨86⟩, ⟨135⟩⟩ ⟨42992, 43150⟩

def event43151 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41035⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩) (1) 0 2 (.universal 43150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩) (none) 43149)

def event43152 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41035⟩⟩, .relation 43151 1, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩)

def event43153 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41035⟩⟩, .relation 43151 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (-1)⟩)

def event43154 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41035⟩⟩, .relation 43151 2, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (1)⟩)

def event43155 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41035⟩⟩, .relation 43151 3, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact43156RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43156RawTermsValid :
    exact43156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43156 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41035⟩⟩) exact43156RawTerms .large 42988 (.finite 202072841853861888) (some (42990))

def event43157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42211⟩⟩) 0 ⟨41035⟩ 43156

def event43158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42211⟩⟩) 1 ⟨42210⟩ 42978

def event43159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42211⟩⟩) (.sum [.predecessor 0 43157 .coefficient, .predecessor 1 43158 .coefficient])

def event43160 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42211⟩⟩, .operator (⟨43156, 0⟩, ⟨42978, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (1)⟩)

def event43161 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42211⟩⟩, .operator (⟨43156, 2⟩, ⟨42978, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (-1)⟩)

def event43162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42211⟩⟩) (.sum [.result 43156 .summary, .result 42978 .summary])

def exact43163RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43163RawTermsValid :
    exact43163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42211⟩⟩) exact43163RawTerms .large 43159 (.finite 32193129122288829188810200055808) (some (43162))

def event43164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42212⟩⟩) 0 ⟨42211⟩ 43163

def event43165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42212⟩⟩) 1 ⟨7160⟩ 15602

def event43166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42212⟩⟩) (.product (.predecessor 0 43164 .coefficient) (.predecessor 1 43165 .coefficient) (⟨false, false, none, none, none⟩))

def event43167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42212⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) [⟨.result 15598 .coefficient, false, none⟩])

def event43168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42212⟩⟩) (.product (.result 43163 .summary) (.transfer 43167) (⟨false, false, none, none, none⟩))

def event43169 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42212⟩⟩, .operator (⟨43163, 0⟩, ⟨15602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩)

def event43170 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42212⟩⟩, .operator (⟨43163, 1⟩, ⟨15602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event43171 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42212⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7159⟩⟩) ⟨7045⟩ 15595)

def event43172 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42212⟩⟩, .relation 43171 0, ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact43173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩]

theorem exact43173RawTermsValid :
    exact43173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42212⟩⟩) exact43173RawTerms .large 43166 (.finite 345671091840339265080175045977281837137920) (some (43168))

def event43174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38661⟩⟩) 0 ⟨7177⟩ 15500

def event43175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38661⟩⟩) 1 ⟨38660⟩ 33950

def event43176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38661⟩⟩) (.authority (.operator))

def exact43177RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (1)⟩]

theorem exact43177RawTermsValid :
    exact43177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43177 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38661⟩⟩) exact43177RawTerms .large 43176 .exactZero (none)

def event43178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39528⟩⟩) 0 ⟨38661⟩ 43177

def event43179 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39528⟩⟩) (.authority (.operator))

def exact43180RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (1)⟩]

theorem exact43180RawTermsValid :
    exact43180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43180 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39528⟩⟩) exact43180RawTerms (.finite 8192) 43179 .exactZero (none)

def event43181 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39530⟩⟩) 0 ⟨39040⟩ 34234

def event43182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39530⟩⟩) 1 ⟨39528⟩ 43180

def event43183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39530⟩⟩) (.product (.predecessor 0 43181 .coefficient) (.predecessor 1 43182 .coefficient) (⟨false, false, none, none, none⟩))

def event43184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39530⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩) [⟨.result 43180 .coefficient, false, none⟩])

def event43185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39530⟩⟩) (.product (.result 34234 .summary) (.transfer 43184) (⟨false, false, none, none, none⟩))

def event43186 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39530⟩⟩, .operator (⟨34234, 0⟩, ⟨43180, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (1)⟩)

def event43187 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39530⟩⟩, .operator (⟨34234, 1⟩, ⟨43180, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (-1)⟩)

def event43188 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39530⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39528⟩⟩) ⟨38661⟩ 43177)

def event43189 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39530⟩⟩, .relation 43188 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (-1)⟩)

def exact43190RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (-1)⟩]

theorem exact43190RawTermsValid :
    exact43190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43190 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39530⟩⟩) exact43190RawTerms .large 43183 (.finite 32192736221397252361486566686720) (some (43185))

def event43191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38352⟩⟩) 0 ⟨37501⟩ 951

def event43192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38352⟩⟩) (.authority (.relationPreimageSource ⟨84⟩))

def exact43193RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩, (1)⟩]

theorem exact43193RawTermsValid :
    exact43193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38352⟩⟩) exact43193RawTerms (.finite 5647228698) 43192 .exactZero (none)

def event43194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38354⟩⟩) 0 ⟨38352⟩ 43193

def event43195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38354⟩⟩) 1 ⟨2370⟩ 4

def event43196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38354⟩⟩) (.scale (.predecessor 0 43194 .coefficient) (.value (.predecessor 1 43195 .coefficient)))

def exact43197RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩, (1)⟩]

theorem exact43197RawTermsValid :
    exact43197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43197 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38354⟩⟩) exact43197RawTerms (.finite 5647228698) 43196 .exactZero (none)

def event43198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38355⟩⟩) 0 ⟨11643⟩ 32120

def event43199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38355⟩⟩) 1 ⟨38354⟩ 43197

def event43200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38355⟩⟩) (.product (.predecessor 0 43198 .coefficient) (.predecessor 1 43199 .coefficient) (⟨false, false, none, none, none⟩))

def event43201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38355⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩) [⟨.result 43193 .coefficient, false, none⟩])

def event43202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38355⟩⟩) (.product (.result 32120 .summary) (.transfer 43201) (⟨false, false, none, none, none⟩))

def event43203 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38355⟩⟩, .operator (⟨32120, 0⟩, ⟨43197, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩, (1)⟩)

def event43204 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨38353⟩⟩)

def event43205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event43206 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event43207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event43208 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event43209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event43210 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event43211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event43212 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event43213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 43212

def event43214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 43210

def event43215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 43213 .coefficient) (.value (.predecessor 1 43214 .coefficient)))

def event43216 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event43217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 43216

def event43218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 43208

def event43219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 43217 .coefficient, .predecessor 1 43218 .coefficient])

def event43220 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event43221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 43220

def event43222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 43206

def event43223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 43222 .coefficient))

def event43224 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event43225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37330⟩⟩) 0 ⟨11600⟩ 43224

def event43226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37330⟩⟩) (.authority (.programFamilyFact))

def exact43227RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37330⟩⟩], []⟩, (1)⟩]

theorem exact43227RawTermsValid :
    exact43227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37330⟩⟩) exact43227RawTerms (.finite 42) 43226 .exactZero (none)

def event43228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14016⟩⟩) 0 ⟨11600⟩ 43224

def event43229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14016⟩⟩) (.authority (.programFamilyFact))

def exact43230RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩], []⟩, (1)⟩]

theorem exact43230RawTermsValid :
    exact43230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43230 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14016⟩⟩) exact43230RawTerms (.finite 42) 43229 .exactZero (none)

def event43231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37331⟩⟩) 0 ⟨14016⟩ 43230

def event43232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37331⟩⟩) 1 ⟨37330⟩ 43227

def event43233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37331⟩⟩) (.product (.predecessor 0 43231 .coefficient) (.predecessor 1 43232 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event43234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37331⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], []⟩) [⟨.result 43230 .coefficient, true, some 1⟩, ⟨.result 43227 .coefficient, true, some 1⟩])

def event43235 : Event := .survivorFold (1) 43234

def exact43236RawTerms : List Term := []

theorem exact43236RawTermsValid :
    exact43236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37331⟩⟩) exact43236RawTerms (.finite 1764) 43233 (.finite 1764) (some (43234))

def event43237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37332⟩⟩) 0 ⟨37331⟩ 43236

def event43238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37332⟩⟩) (.identity (.predecessor 0 43237 .coefficient))

def event43239 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37332⟩⟩) (.finite 1764)

def event43240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37500⟩⟩) 0 ⟨37332⟩ 43239

def event43241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37500⟩⟩) (.authority (.programFamilyFact))

def exact43242RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], []⟩, (1)⟩]

theorem exact43242RawTermsValid :
    exact43242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43242 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37500⟩⟩) exact43242RawTerms (.finite 42) 43241 .exactZero (none)

def event43243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37501⟩⟩) 0 ⟨37500⟩ 43242

def event43244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37501⟩⟩) (.identity (.predecessor 0 43243 .coefficient))

def event43245 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37501⟩⟩) (.finite 42)

def event43246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38352⟩⟩) 0 ⟨37501⟩ 43245

def event43247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38352⟩⟩) (.authority (.relationPreimageSource ⟨84⟩))

def exact43248RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩, (1)⟩]

theorem exact43248RawTermsValid :
    exact43248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38352⟩⟩) exact43248RawTerms (.finite 5647228698) 43247 .exactZero (none)

def event43249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact43250RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact43250RawTermsValid :
    exact43250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43250 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact43250RawTerms .large 43249 .exactZero (none)

def event43251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38353⟩⟩) 0 ⟨35⟩ 43250

def event43252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38353⟩⟩) 1 ⟨38352⟩ 43248

def event43253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38353⟩⟩) (.product (.predecessor 0 43251 .coefficient) (.predecessor 1 43252 .coefficient) (⟨false, false, none, none, none⟩))

def event43254 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38353⟩⟩, .operator (⟨43250, 0⟩, ⟨43248, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩, (1)⟩)

def exact43255RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩, (1)⟩]

theorem exact43255RawTermsValid :
    exact43255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43255 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38353⟩⟩) exact43255RawTerms .large 43253 .exactZero (none)

def event43256 : Event := .preFoldPolynomial 43255 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩, (1)⟩] .exactZero none

def exact43257RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩, (1)⟩]

def event43257 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨38353⟩⟩) 43256 exact43257RawTerms .large 43253 .exactZero (none)

def event43258 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨39533⟩⟩)

def event43259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event43260 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event43261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event43262 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event43263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def eventLeaf2688 : Array AnnotatedEvent := #[
  { event := event43008
    frameStart := 42992 },
  { event := event43009
    frameStart := 42992 },
  { event := event43010
    frameStart := 42992 },
  { event := event43011
    frameStart := 42992 },
  { event := event43012
    frameStart := 42992 },
  { event := event43013
    frameStart := 42992 },
  { event := event43014
    frameStart := 42992 },
  { event := event43015
    frameStart := 42992 },
  { event := event43016
    frameStart := 42992 },
  { event := event43017
    frameStart := 42992 },
  { event := event43018
    frameStart := 42992 },
  { event := event43019
    frameStart := 42992 },
  { event := event43020
    frameStart := 42992 },
  { event := event43021
    frameStart := 42992 },
  { event := event43022
    frameStart := 42992 },
  { event := event43023
    frameStart := 42992 }
]

def eventLeaf2689 : Array AnnotatedEvent := #[
  { event := event43024
    frameStart := 42992 },
  { event := event43025
    frameStart := 42992 },
  { event := event43026
    frameStart := 42992 },
  { event := event43027
    frameStart := 42992 },
  { event := event43028
    frameStart := 42992 },
  { event := event43029
    frameStart := 42992 },
  { event := event43030
    frameStart := 42992 },
  { event := event43031
    frameStart := 42992 },
  { event := event43032
    frameStart := 42992 },
  { event := event43033
    frameStart := 42992 },
  { event := event43034
    frameStart := 42992 },
  { event := event43035
    frameStart := 42992 },
  { event := event43036
    frameStart := 42992 },
  { event := event43037
    frameStart := 42992 },
  { event := event43038
    frameStart := 42992 },
  { event := event43039
    frameStart := 42992 }
]

def eventLeaf2690 : Array AnnotatedEvent := #[
  { event := event43040
    frameStart := 42992 },
  { event := event43041
    frameStart := 42992 },
  { event := event43042
    frameStart := 42992 },
  { event := event43043
    frameStart := 42992 },
  { event := event43044
    frameStart := 42992 },
  { event := event43045
    frameStart := 42992 },
  { event := event43046
    frameStart := 43046 },
  { event := event43047
    frameStart := 43046 },
  { event := event43048
    frameStart := 43046 },
  { event := event43049
    frameStart := 43046 },
  { event := event43050
    frameStart := 43046 },
  { event := event43051
    frameStart := 43046 },
  { event := event43052
    frameStart := 43046 },
  { event := event43053
    frameStart := 43046 },
  { event := event43054
    frameStart := 43046 },
  { event := event43055
    frameStart := 43046 }
]

def eventLeaf2691 : Array AnnotatedEvent := #[
  { event := event43056
    frameStart := 43046 },
  { event := event43057
    frameStart := 43046 },
  { event := event43058
    frameStart := 43046 },
  { event := event43059
    frameStart := 43046 },
  { event := event43060
    frameStart := 43046 },
  { event := event43061
    frameStart := 43046 },
  { event := event43062
    frameStart := 43046 },
  { event := event43063
    frameStart := 43046 },
  { event := event43064
    frameStart := 43046 },
  { event := event43065
    frameStart := 43046 },
  { event := event43066
    frameStart := 43046 },
  { event := event43067
    frameStart := 43046 },
  { event := event43068
    frameStart := 43046 },
  { event := event43069
    frameStart := 43046 },
  { event := event43070
    frameStart := 43046 },
  { event := event43071
    frameStart := 43046 }
]

def eventLeaf2692 : Array AnnotatedEvent := #[
  { event := event43072
    frameStart := 43046 },
  { event := event43073
    frameStart := 43046 },
  { event := event43074
    frameStart := 43046 },
  { event := event43075
    frameStart := 43046 },
  { event := event43076
    frameStart := 43046 },
  { event := event43077
    frameStart := 43046 },
  { event := event43078
    frameStart := 43046 },
  { event := event43079
    frameStart := 43046 },
  { event := event43080
    frameStart := 43046 },
  { event := event43081
    frameStart := 43046 },
  { event := event43082
    frameStart := 43046 },
  { event := event43083
    frameStart := 43046 },
  { event := event43084
    frameStart := 43046 },
  { event := event43085
    frameStart := 43046 },
  { event := event43086
    frameStart := 43046 },
  { event := event43087
    frameStart := 43046 }
]

def eventLeaf2693 : Array AnnotatedEvent := #[
  { event := event43088
    frameStart := 43046 },
  { event := event43089
    frameStart := 43046 },
  { event := event43090
    frameStart := 43046 },
  { event := event43091
    frameStart := 43046 },
  { event := event43092
    frameStart := 43046 },
  { event := event43093
    frameStart := 43046 },
  { event := event43094
    frameStart := 43046 },
  { event := event43095
    frameStart := 43046 },
  { event := event43096
    frameStart := 43046 },
  { event := event43097
    frameStart := 43046 },
  { event := event43098
    frameStart := 43046 },
  { event := event43099
    frameStart := 43046 },
  { event := event43100
    frameStart := 43046 },
  { event := event43101
    frameStart := 43046 },
  { event := event43102
    frameStart := 43046 },
  { event := event43103
    frameStart := 43046 }
]

def eventLeaf2694 : Array AnnotatedEvent := #[
  { event := event43104
    frameStart := 43046 },
  { event := event43105
    frameStart := 43046 },
  { event := event43106
    frameStart := 43046 },
  { event := event43107
    frameStart := 43046 },
  { event := event43108
    frameStart := 43046 },
  { event := event43109
    frameStart := 43046 },
  { event := event43110
    frameStart := 43046 },
  { event := event43111
    frameStart := 43046 },
  { event := event43112
    frameStart := 43046 },
  { event := event43113
    frameStart := 43046 },
  { event := event43114
    frameStart := 43046 },
  { event := event43115
    frameStart := 43046 },
  { event := event43116
    frameStart := 43046 },
  { event := event43117
    frameStart := 43046 },
  { event := event43118
    frameStart := 43046 },
  { event := event43119
    frameStart := 43046 }
]

def eventLeaf2695 : Array AnnotatedEvent := #[
  { event := event43120
    frameStart := 43046 },
  { event := event43121
    frameStart := 43046 },
  { event := event43122
    frameStart := 43046 },
  { event := event43123
    frameStart := 43046 },
  { event := event43124
    frameStart := 43046 },
  { event := event43125
    frameStart := 43046 },
  { event := event43126
    frameStart := 43046 },
  { event := event43127
    frameStart := 43046 },
  { event := event43128
    frameStart := 43046 },
  { event := event43129
    frameStart := 43046 },
  { event := event43130
    frameStart := 43046 },
  { event := event43131
    frameStart := 43046 },
  { event := event43132
    frameStart := 43046 },
  { event := event43133
    frameStart := 43046 },
  { event := event43134
    frameStart := 43046 },
  { event := event43135
    frameStart := 43046 }
]

def eventLeaf2696 : Array AnnotatedEvent := #[
  { event := event43136
    frameStart := 43046 },
  { event := event43137
    frameStart := 43046 },
  { event := event43138
    frameStart := 43046 },
  { event := event43139
    frameStart := 43046 },
  { event := event43140
    frameStart := 43046 },
  { event := event43141
    frameStart := 43046 },
  { event := event43142
    frameStart := 43046 },
  { event := event43143
    frameStart := 43046 },
  { event := event43144
    frameStart := 43046 },
  { event := event43145
    frameStart := 43046 },
  { event := event43146
    frameStart := 43046 },
  { event := event43147
    frameStart := 43046 },
  { event := event43148
    frameStart := 43046 },
  { event := event43149
    frameStart := 43046 },
  { event := event43150
    frameStart := 0 },
  { event := event43151
    frameStart := 0 }
]

def eventLeaf2697 : Array AnnotatedEvent := #[
  { event := event43152
    frameStart := 0 },
  { event := event43153
    frameStart := 0 },
  { event := event43154
    frameStart := 0 },
  { event := event43155
    frameStart := 0 },
  { event := event43156
    frameStart := 0 },
  { event := event43157
    frameStart := 0 },
  { event := event43158
    frameStart := 0 },
  { event := event43159
    frameStart := 0 },
  { event := event43160
    frameStart := 0 },
  { event := event43161
    frameStart := 0 },
  { event := event43162
    frameStart := 0 },
  { event := event43163
    frameStart := 0 },
  { event := event43164
    frameStart := 0 },
  { event := event43165
    frameStart := 0 },
  { event := event43166
    frameStart := 0 },
  { event := event43167
    frameStart := 0 }
]

def eventLeaf2698 : Array AnnotatedEvent := #[
  { event := event43168
    frameStart := 0 },
  { event := event43169
    frameStart := 0 },
  { event := event43170
    frameStart := 0 },
  { event := event43171
    frameStart := 0 },
  { event := event43172
    frameStart := 0 },
  { event := event43173
    frameStart := 0 },
  { event := event43174
    frameStart := 0 },
  { event := event43175
    frameStart := 0 },
  { event := event43176
    frameStart := 0 },
  { event := event43177
    frameStart := 0 },
  { event := event43178
    frameStart := 0 },
  { event := event43179
    frameStart := 0 },
  { event := event43180
    frameStart := 0 },
  { event := event43181
    frameStart := 0 },
  { event := event43182
    frameStart := 0 },
  { event := event43183
    frameStart := 0 }
]

def eventLeaf2699 : Array AnnotatedEvent := #[
  { event := event43184
    frameStart := 0 },
  { event := event43185
    frameStart := 0 },
  { event := event43186
    frameStart := 0 },
  { event := event43187
    frameStart := 0 },
  { event := event43188
    frameStart := 0 },
  { event := event43189
    frameStart := 0 },
  { event := event43190
    frameStart := 0 },
  { event := event43191
    frameStart := 0 },
  { event := event43192
    frameStart := 0 },
  { event := event43193
    frameStart := 0 },
  { event := event43194
    frameStart := 0 },
  { event := event43195
    frameStart := 0 },
  { event := event43196
    frameStart := 0 },
  { event := event43197
    frameStart := 0 },
  { event := event43198
    frameStart := 0 },
  { event := event43199
    frameStart := 0 }
]

def eventLeaf2700 : Array AnnotatedEvent := #[
  { event := event43200
    frameStart := 0 },
  { event := event43201
    frameStart := 0 },
  { event := event43202
    frameStart := 0 },
  { event := event43203
    frameStart := 0 },
  { event := event43204
    frameStart := 43204 },
  { event := event43205
    frameStart := 43204 },
  { event := event43206
    frameStart := 43204 },
  { event := event43207
    frameStart := 43204 },
  { event := event43208
    frameStart := 43204 },
  { event := event43209
    frameStart := 43204 },
  { event := event43210
    frameStart := 43204 },
  { event := event43211
    frameStart := 43204 },
  { event := event43212
    frameStart := 43204 },
  { event := event43213
    frameStart := 43204 },
  { event := event43214
    frameStart := 43204 },
  { event := event43215
    frameStart := 43204 }
]

def eventLeaf2701 : Array AnnotatedEvent := #[
  { event := event43216
    frameStart := 43204 },
  { event := event43217
    frameStart := 43204 },
  { event := event43218
    frameStart := 43204 },
  { event := event43219
    frameStart := 43204 },
  { event := event43220
    frameStart := 43204 },
  { event := event43221
    frameStart := 43204 },
  { event := event43222
    frameStart := 43204 },
  { event := event43223
    frameStart := 43204 },
  { event := event43224
    frameStart := 43204 },
  { event := event43225
    frameStart := 43204 },
  { event := event43226
    frameStart := 43204 },
  { event := event43227
    frameStart := 43204 },
  { event := event43228
    frameStart := 43204 },
  { event := event43229
    frameStart := 43204 },
  { event := event43230
    frameStart := 43204 },
  { event := event43231
    frameStart := 43204 }
]

def eventLeaf2702 : Array AnnotatedEvent := #[
  { event := event43232
    frameStart := 43204 },
  { event := event43233
    frameStart := 43204 },
  { event := event43234
    frameStart := 43204 },
  { event := event43235
    frameStart := 43204 },
  { event := event43236
    frameStart := 43204 },
  { event := event43237
    frameStart := 43204 },
  { event := event43238
    frameStart := 43204 },
  { event := event43239
    frameStart := 43204 },
  { event := event43240
    frameStart := 43204 },
  { event := event43241
    frameStart := 43204 },
  { event := event43242
    frameStart := 43204 },
  { event := event43243
    frameStart := 43204 },
  { event := event43244
    frameStart := 43204 },
  { event := event43245
    frameStart := 43204 },
  { event := event43246
    frameStart := 43204 },
  { event := event43247
    frameStart := 43204 }
]

def eventLeaf2703 : Array AnnotatedEvent := #[
  { event := event43248
    frameStart := 43204 },
  { event := event43249
    frameStart := 43204 },
  { event := event43250
    frameStart := 43204 },
  { event := event43251
    frameStart := 43204 },
  { event := event43252
    frameStart := 43204 },
  { event := event43253
    frameStart := 43204 },
  { event := event43254
    frameStart := 43204 },
  { event := event43255
    frameStart := 43204 },
  { event := event43256
    frameStart := 43204 },
  { event := event43257
    frameStart := 43204 },
  { event := event43258
    frameStart := 43258 },
  { event := event43259
    frameStart := 43258 },
  { event := event43260
    frameStart := 43258 },
  { event := event43261
    frameStart := 43258 },
  { event := event43262
    frameStart := 43258 },
  { event := event43263
    frameStart := 43258 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events168
